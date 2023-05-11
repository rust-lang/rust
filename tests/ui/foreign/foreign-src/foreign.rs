// run-pass



pub fn main() {
    libc.puts(rustrt.str_buf("hello, extern world 1"));
    libc.puts(rustrt.str_buf("hello, extern world 2"));
    libc.puts(rustrt.str_buf("hello, extern world 3"));
}
