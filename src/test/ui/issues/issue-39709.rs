// run-pass
#![allow(unused_macros)]
fn main() {
    println!("{}", { macro_rules! x { ($(t:tt)*) => {} } 33 });
}
