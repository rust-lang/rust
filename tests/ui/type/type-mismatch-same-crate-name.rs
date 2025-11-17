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
        //~| NOTE expected `main::a::Foo`, found a different `main::a::Foo`
        //~| NOTE arguments to this function are incorrect
        //~| NOTE there are multiple different versions of crate `crate_a1` in the dependency graph
        //~| NOTE function defined here
        //~| HELP you can use `cargo tree` to explore your dependency tree
        a::try_bar(bar2);
        //~^ ERROR mismatched types
        //~| NOTE expected trait `main::a::Bar`, found a different trait `main::a::Bar`
        //~| NOTE arguments to this function are incorrect
        //~| NOTE there are multiple different versions of crate `crate_a1` in the dependency graph
        //~| NOTE function defined here
        //~| HELP you can use `cargo tree` to explore your dependency tree
    }
}
