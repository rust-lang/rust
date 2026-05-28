#![feature(extern_item_impls)]
// EIIs cannot be used in statement position.
// This is also a regression test for an ICE (https://github.com/rust-lang/rust/issues/149980).

fn main() {
    struct Bar;

    #[eii]
    //~^ ERROR `#[eii]` is only valid on functions
    impl Bar {}


    // Even on functions, eiis in statement position are rejected
    #[eii]
    //~^ ERROR `#[eii]` can only be used on functions inside a module
    fn foo() {}
}
