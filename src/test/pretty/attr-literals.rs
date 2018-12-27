// pp-exact
// Tests literals in attributes.

#![feature(custom_attribute)]

fn main() {
    #![hello("hi", 1, 2, 1.012, pi = 3.14, bye, name("John"))]
    #[align = 8]
    fn f() { }

    #[vector(1, 2, 3)]
    fn g() { }
}
