// Tests literals in attributes.

//@ pp-exact

#![feature(rustc_attrs)]

fn main() {
    #![rustc_dummy("hi", 1, 2, 1.012, pi = 3.14, bye, name("John"))]
    #[rustc_dummy = 8]
    fn f() {}

    #[rustc_dummy(1, 2, 3)]
    fn g() {}
}
