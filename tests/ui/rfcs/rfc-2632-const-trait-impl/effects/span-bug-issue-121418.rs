#![feature(const_trait_impl)]
#![feature(effects)] //~ WARN the feature `effects` is incomplete

struct S;
trait T {}

impl const dyn T {
    //~^ ERROR inherent impls cannot be `const`
    pub const fn new() -> std::sync::Mutex<dyn T> {}
    //~^ ERROR mismatched types
    //~| ERROR cannot be known at compilation time
}

fn main() {}
