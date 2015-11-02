use fmt;
use sys::stdio;
use io::Write;

pub fn dumb_print(args: fmt::Arguments) {
    let _ = stdio::stderr().map(|mut stderr| stderr.write_fmt(args));
}
