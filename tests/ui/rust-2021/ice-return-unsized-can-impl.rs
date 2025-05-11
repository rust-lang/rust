// Doesn't trigger ICE when returning unsized trait that can be impl
// issue https://github.com/rust-lang/rust/issues/120482
//@ edition:2021

trait B {
    fn bar(&self, x: &Self);
}

trait A {
    fn g(new: B) -> B;
    //~^ ERROR: expected a type, found a trait
    //~| ERROR: expected a type, found a trait
}

fn main() {}
