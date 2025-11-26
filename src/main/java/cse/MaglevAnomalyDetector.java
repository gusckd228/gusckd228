package cse;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class MaglevAnomalyDetector extends Configured implements Tool {

    // ---------------------- 1. Mapper 클래스 ----------------------
    // 입력: <Offset, 한 줄의 CSV 텍스트>, 출력: <결합 키, 전압 값>
    public static class MaglevMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        
        private final static DoubleWritable voltage = new DoubleWritable();
        private Text outputKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            
            String line = value.toString();
            // CSV 헤더 줄 (Window_ID로 시작) 건너뛰기
            if (line.startsWith("Window_ID")) {
                return;
            }

            // --- 파일 이름에서 데이터 상태 (N 또는 A) 추출 ---
            // 처리 중인 파일의 정보를 FileSplit으로 전달받음
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName(); // 예: N_Maglev_01.csv
            String dataState = fileName.substring(0, 1); // 'N' 또는 'A'

            // CSV 파싱
            String[] tokens = line.split(",");
            if (tokens.length < 5) {
                return; // 데이터가 불완전하면 스킵
            }
            
            // CSV 열: [0]Window_ID, [1]Train_ID, [2]Time_ms, [3]CH1_Voltage, [4]CH2_Voltage
            String trainID = tokens[1];
            
            try {
                double ch1Voltage = Double.parseDouble(tokens[3]);
                double ch2Voltage = Double.parseDouble(tokens[4]);
                
                // --- 출력 키 생성: TrainID_상태_채널 ---
                // 예: T001_N_CH1 (Normal의 모든 T001 채널1 데이터 그룹)
                
                // CH1 데이터 출력
                String keyCH1 = trainID + "_" + dataState + "_CH1";
                outputKey.set(keyCH1);
                voltage.set(ch1Voltage);
                context.write(outputKey, voltage);

                // CH2 데이터 출력
                String keyCH2 = trainID + "_" + dataState + "_CH2";
                outputKey.set(keyCH2);
                voltage.set(ch2Voltage);
                context.write(outputKey, voltage);
            } catch (NumberFormatException e) {
                 // 전압 값 파싱 오류 시 스킵
                 return;
            }
        }
    }

    // ---------------------- 2. Reducer 클래스 ----------------------
    // 입력: <결합 키, [전압 리스트]>, 출력: <상태_채널, 표준 편차(Sigma)>
    public static class MaglevReducer extends Reducer<Text, DoubleWritable, Text, Text> {
        
        // 표준 편차 계산 함수 (표본 표준 편차)
        private double calculateStandardDeviation(Iterable<DoubleWritable> values) {
            double sum = 0;
            int count = 0;
            List<Double> data = new ArrayList<>();
            
            for (DoubleWritable val : values) {
                double d = val.get();
                sum += d;
                count++;
                data.add(d);
            }

            // 데이터가 2개 미만이면 표준 편차 계산 불가 (0 반환)
            if (count < 2) return 0.0;
            
            double mean = sum / count;
            double varianceSum = 0;
            
            for (double d : data) {
                varianceSum += Math.pow(d - mean, 2);
            }
            
            // 표본 표준 편차 (Sample Standard Deviation)
            return Math.sqrt(varianceSum / (count - 1)); 
        }

        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            
            double sigma = calculateStandardDeviation(values);
            
            // 출력 형식: <TrainID_상태_채널, 표준편차_값>
            // 예: T001_N_CH1, 0.0001234
            context.write(key, new Text(String.format("%.8f", sigma)));
        }
    }

    // ---------------------- 3. Driver 클래스 (Job 설정) ----------------------
    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("사용법: MaglevAnomalyDetector <입력 경로> <출력 경로>");
            System.exit(-1);
        }
        
        Job job = Job.getInstance(getConf(), "Maglev Anomaly Detector");
        job.setJarByClass(MaglevAnomalyDetector.class);

        // Mapper 및 Reducer 클래스 설정
        job.setMapperClass(MaglevMapper.class);
        job.setReducerClass(MaglevReducer.class);

        // 입/출력 포맷 설정 (CSV 텍스트 입력)
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Mapper 출력 키/값 타입 (Reducer 입력 타입)
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        
        // Reducer 출력 키/값 타입 (최종 출력 타입)
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // 입/출력 경로 설정
        FileInputFormat.addInputPath(job, new Path(args[0])); // 입력 경로
        FileOutputFormat.setOutputPath(job, new Path(args[1])); // 출력 경로

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new MaglevAnomalyDetector(), args);
        System.exit(res);
    }
}
