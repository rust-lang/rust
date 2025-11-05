#![feature(rustc_attrs, const_trait_impl)]

#[rustc_comptime]
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

#[rustc_comptime]
fn baz() {
    // TODO: Should be allowed
    foo(); //~ ERROR: comptime fns can only be called at compile time
}
