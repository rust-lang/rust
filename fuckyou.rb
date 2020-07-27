Dir.glob('src/test/mir-opt/**/*.rs').each do |f|
  puts f
  t = File.read(f)
  b = File.basename(f, ".rs")
  t.gsub!(/\/\/ EMIT_MIR rustc/, "// EMIT_MIR " + b)
  File.open(f, "w") { |f| f.puts t }
end
