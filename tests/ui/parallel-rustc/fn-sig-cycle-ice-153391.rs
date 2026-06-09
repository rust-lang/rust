// Regression test for #153391.
//
//@ edition:2024
//@ compile-flags: -Z threads=16
//@ compare-output-by-lines
//@ ignore-test (#142063)

trait A {
    fn g() -> B;
    //~^ ERROR expected a type, found a trait
}

trait B {
    fn bar(&self, x: &A);
    //~^ ERROR expected a type, found a trait
}

fn main() {}
