#![feature(fn_delegation)]
#![allow(incomplete_features)]

fn main() {
    fn foo<const N: dyn for<'a> Foo>() {}
    //~^ ERROR: cannot find trait `Foo` in this scope
    reuse foo as bar;
}
