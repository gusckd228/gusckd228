package cse;
import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
public class WordCount extends Configured implements Tool{
		public static void main(String[] args) throws Exception {
			ToolRunner.run(new WordCount(), args);
		}
		public static class WCMapper extends Mapper<Object, Text, Text, IntWritable>{
			Text word = new Text();
			IntWritable one = new IntWritable(1);
			@Override protected void map(Object key, Text value,
						Mapper<Object, Text, Text, IntWritable>.Context context)
						throws IOException, InterruptedException
		{
				StringTokenizer st = new StringTokenizer(value.toString());
				while(st.hasMoreTokens()) {
					word.set(st.nextToken());
					context.write(word, one);
				}
		}
}
public static class WCReducer extends
			Reducer<Text, IntWritable, Text, IntWritable>
{
		IntWritable oval = new IntWritable();
		@Override protected void reduce(Text key, Iterable<IntWritable>
		values,
			Reducer<Text, IntWritable, Text, IntWritable>.Context context)
					throws IOException, InterruptedException
	{
		int sum = 0;
		for(IntWritable value : values)
		{
			sum += value.get();
		} oval.set(sum);
		context.write(key, oval);
	}
}
	public int run(String[] args) throws Exception {
		Job myjob = Job.getInstance(getConf());
		myjob.setJarByClass(WordCount.class);
		myjob.setMapperClass(WCMapper.class);
		myjob.setReducerClass(WCReducer.class);
		myjob.setMapOutputKeyClass(Text.class);
		myjob.setMapOutputValueClass(IntWritable.class);
		myjob.setOutputFormatClass(TextOutputFormat.class);
		myjob.setInputFormatClass(TextInputFormat.class);
		FileInputFormat.addInputPath(myjob, new Path(args[0]));
		FileOutputFormat.setOutputPath(myjob, new
		Path(args[0]).suffix(".out"));
		myjob.waitForCompletion(true);
		return 0;
	}


}
