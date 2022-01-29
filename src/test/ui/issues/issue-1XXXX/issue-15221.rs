// run-pass
#![allow(path_statements)]
// pretty-expanded FIXME #23616

macro_rules! inner {
    ($e:pat ) => ($e)
}

macro_rules! outer {
    ($e:pat ) => (inner!($e))
}

fn main() {
    let outer!(g1) = 13;
    g1;
}
