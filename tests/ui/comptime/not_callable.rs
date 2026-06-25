#![feature(comptime, const_trait_impl)]

#[comptime]
fn foo() {}

fn main() {
    // Ok
    const { foo() };
    // Not Ok
    foo(); //~ ERROR: comptime fns can only be called at compile time
}

const fn bar() {
    // Not Ok
    foo(); //~ ERROR: comptime fns can only be called at compile time
}

#[comptime]
fn baz() {
    // Ok
    foo();
}
