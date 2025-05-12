#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

union Union {
    x: usize,
}

trait Trait {}
impl Trait for Union {}

fn bar(_: dyn* Trait) {}

fn main() {
    bar(Union { x: 0usize });
    //~^ ERROR `Union` needs to have the same ABI as a pointer
}
