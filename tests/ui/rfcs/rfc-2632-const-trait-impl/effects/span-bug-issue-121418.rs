#![feature(const_trait_impl)]
#![feature(effects)]

struct S;
trait T {}

impl const dyn T {
    //~^ ERROR inherent impls cannot be `const`
    //~| ERROR the const parameter `host` is not constrained by the impl trait, self type, or
    pub const fn new() -> std::sync::Mutex<dyn T> {}
}

fn main() {}
