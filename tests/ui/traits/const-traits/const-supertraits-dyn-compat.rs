#![feature(const_trait_impl)]

const trait Super {}

// Not ok
const trait Unconditionally: const Super {}
fn test() {
    let _: &dyn Unconditionally;
    //~^ ERROR the trait `Unconditionally` is not dyn compatible
}

// Okay
const trait Conditionally: [const] Super {}
fn test2() {
    let _: &dyn Conditionally;
}

fn main() {}
