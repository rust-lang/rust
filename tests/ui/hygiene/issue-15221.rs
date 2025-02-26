//@ run-pass
#![allow(path_statements)]

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
