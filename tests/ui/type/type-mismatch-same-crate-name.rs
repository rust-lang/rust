//@ aux-build:crate_a1.rs
//@ aux-build:crate_a2.rs

// This tests the extra note reported when a type error deals with
// seemingly identical types.
// The main use case of this error is when there are two crates imported
// with the same name, causing a type mismatch. Here, we simulate that error
// using block-scoped aliased `extern crate` declarations.
// This is *not* the same case as two different crate versions in the
// dependency tree. That is tested in `tests/run-make/crate-loading/`.

fn main() {
    let foo2 = {extern crate crate_a2 as a; a::Foo};
    let bar2 = {extern crate crate_a2 as a; a::bar()};
    {
        extern crate crate_a1 as a;
        a::try_foo(foo2);
        //~^ ERROR mismatched types
        //~| NOTE expected `crate_a1::Foo`, found `crate_a2::Foo`
        //~| NOTE arguments to this function are incorrect
        //~| NOTE `crate_a2::Foo` and `crate_a1::Foo` have similar names, but are actually distinct types
        //~| NOTE `crate_a2::Foo` is defined in crate `crate_a2`
        //~| NOTE `crate_a1::Foo` is defined in crate `crate_a1`
        //~| NOTE function defined here
        a::try_bar(bar2);
        //~^ ERROR mismatched types
        //~| NOTE expected trait `crate_a1::Bar`, found trait `crate_a2::Bar`
        //~| NOTE arguments to this function are incorrect
        //~| NOTE expected struct `Box<(dyn crate_a1::Bar + 'static)>`
        //~| NOTE    found struct `Box<dyn crate_a2::Bar>`
        //~| NOTE function defined here
    }
}
