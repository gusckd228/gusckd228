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

public class MaglevFinalClassifier extends Configured implements Tool {

    // ---------------------- 1. Mapper 클래스 ----------------------
    // Key: 파일 이름 전체, Value: 전압 값 (개별 파일의 모든 데이터를 Reducer로 보냄)
    public static class FinalMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        
        private final static DoubleWritable voltage = new DoubleWritable();
        private Text outputKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            
            String line = value.toString();
            if (line.startsWith("Window_ID")) return; // 헤더 스킵

            // --- 파일 이름 전체를 Key로 사용 ---
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName(); // 예: N_Maglev_01.csv
            outputKey.set(fileName);

            String[] tokens = line.split(",");
            if (tokens.length < 5) return;
            
            try {
                // CH1과 CH2 전압을 모두 Value로 출력
                double ch1Voltage = Double.parseDouble(tokens[3]);
                double ch2Voltage = Double.parseDouble(tokens[4]);
                
                voltage.set(ch1Voltage);
                context.write(outputKey, voltage);
                
                voltage.set(ch2Voltage);
                context.write(outputKey, voltage);
            } catch (NumberFormatException e) {
                 // 전압 값 파싱 오류 시 스킵
            }
        }
    }

    // ---------------------- 2. Reducer 클래스 ----------------------
    public static class FinalReducer extends Reducer<Text, DoubleWritable, Text, Text> {
        
        // --- 💡 임계값 정의: 0.03882 * 0.95 = 0.03688 ---
        private static final double SIGMA_THRESHOLD = 0.03688; 

        // 표준 편차 계산 함수 (이전 Job에서 사용한 동일 함수)
        private double calculateStandardDeviation(Iterable<DoubleWritable> values) {
            double sum = 0;
            int count = 0;
            List<Double> data = new ArrayList<>();
            for (DoubleWritable val : values) {
                data.add(val.get());
                sum += val.get();
                count++;
            }
            if (count < 2) return 0.0;
            double mean = sum / count;
            double varianceSum = 0;
            for (double d : data) {
                varianceSum += Math.pow(d - mean, 2);
            }
            return Math.sqrt(varianceSum / (count - 1));
        }

        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> values, Context context)
                throws IOException, InterruptedException {
            
            double fileSigma = calculateStandardDeviation(values);
            String finalResult;
            
            if (fileSigma < SIGMA_THRESHOLD) {
                // 임계값보다 낮으면 (정상 시그마보다 작으면) 비정상으로 판별
                finalResult = "ANOMALY_DETECTED (Sigma: " + String.format("%.8f", fileSigma) + " < " + SIGMA_THRESHOLD + ")";
            } else {
                finalResult = "NORMAL_Operation (Sigma: " + String.format("%.8f", fileSigma) + ")";
            }
            
            context.write(key, new Text(finalResult));
        }
    }

    // ---------------------- 3. Driver 클래스 (Job 설정) ----------------------
    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("사용법: MaglevFinalClassifier <입력 경로> <출력 경로>");
            System.exit(-1);
        }
        
        Job job = Job.getInstance(getConf(), "Maglev Final Classifier");
        job.setJarByClass(MaglevFinalClassifier.class);
        job.setMapperClass(FinalMapper.class);
        job.setReducerClass(FinalReducer.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new MaglevFinalClassifier(), args);
        System.exit(res);
    }
}