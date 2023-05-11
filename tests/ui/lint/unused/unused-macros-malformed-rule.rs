#![deny(unused_macros)]

macro_rules! foo { //~ ERROR: unused macro definition
    (v) => {};
    () => 0; //~ ERROR: macro rhs must be delimited
}

macro_rules! bar {
    (v) => {};
    () => 0; //~ ERROR: macro rhs must be delimited
}

fn main() {
    bar!(v);
}
