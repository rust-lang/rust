// run-pass
#![allow(unused_macros)]
macro_rules! m {
    ($e:expr) => {
        macro_rules! n { () => { $e } }
    }
}

fn main() {
    m!(foo!());
}
