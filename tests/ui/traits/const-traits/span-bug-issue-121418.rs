#![feature(const_trait_impl)]

struct S;
trait T {}

const impl dyn T {
    pub const fn new() -> std::sync::Mutex<dyn T> {}
    //~^ ERROR mismatched types
    //~| ERROR cannot be known at compilation time
    //~| ERROR redundant `const` fn marker in const impl
}

fn main() {}
