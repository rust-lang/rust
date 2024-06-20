//@ revisions: gated ungated
#![cfg_attr(gated, feature(unsafe_extern_blocks))]

safe fn foo() {}
//~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier

safe static FOO: i32 = 1;
//~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier

trait Foo {
    safe fn foo();
    //~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier
}

impl Foo for () {
    safe fn foo() {}
    //~^ ERROR: items outside of `unsafe extern { }` cannot be declared with `safe` safety qualifier
}

type FnPtr = safe fn(i32, i32) -> i32;
//~^ ERROR: function pointers cannot be declared with `safe` safety qualifier

fn main() {}
