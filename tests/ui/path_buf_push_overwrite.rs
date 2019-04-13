// run-rustfix
use std::path::PathBuf;

fn main() {
    let mut x = PathBuf::from("/foo");
    x.push("/bar");
}
