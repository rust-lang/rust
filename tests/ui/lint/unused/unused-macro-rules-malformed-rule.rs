#![deny(unused_macro_rules)]

macro_rules! foo {
    (v) => {};
    (w) => {};
    () => 0; //~ ERROR: macro rhs must be delimited
}

fn main() {
    foo!(v);
}
