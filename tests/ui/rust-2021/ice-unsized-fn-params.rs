//@ edition:2021
// Test that it doesn't trigger an ICE when using an unsized fn params.
// https://github.com/rust-lang/rust/issues/120241

trait B {
    fn f(a: A) -> A;
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: expected a type, found a trait
}

trait A {
    fn g(b: B) -> B;
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: expected a type, found a trait
}

fn main() {}
