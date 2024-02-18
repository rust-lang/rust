use std::fs::OpenOptions;
#[allow(unused_must_use)]
#[warn(clippy::suspicious_open_options)]
fn main() {
    OpenOptions::new().create(true).open("foo.txt");
    //~^ ERROR: file opened with `create`, but `truncate` behavior not defined
}
