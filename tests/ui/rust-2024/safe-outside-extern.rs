//@ revisions: gated ungated
#![cfg_attr(gated, feature(unsafe_extern_blocks))]

safe fn foo() {}
//~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier
//[ungated]~| ERROR: unsafe extern {}` blocks and `safe` keyword are experimental [E0658]

safe static FOO: i32 = 1;
//~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier
//[ungated]~| ERROR: unsafe extern {}` blocks and `safe` keyword are experimental [E0658]

trait Foo {
    safe fn foo();
    //~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier
    //[ungated]~| ERROR: unsafe extern {}` blocks and `safe` keyword are experimental [E0658]
}

impl Foo for () {
    safe fn foo() {}
    //~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier
    //[ungated]~| ERROR: unsafe extern {}` blocks and `safe` keyword are experimental [E0658]
}

type FnPtr = safe fn(i32, i32) -> i32;
//~^ ERROR: function pointers cannot be declared with `safe` safety qualifier
//[ungated]~| ERROR: unsafe extern {}` blocks and `safe` keyword are experimental [E0658]

fn main() {}
