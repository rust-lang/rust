#![feature(rustc_attrs, const_trait_impl)]

#[rustc_comptime]
const fn foo() {}

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
const fn baz() {
    // Ok
    foo();
}
