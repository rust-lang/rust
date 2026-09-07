// Regression test for <https://github.com/rust-lang/rust/issues/146084>.

//@ compile-flags: --crate-type=lib

struct S<const N: [()] = { loop {} }>;
//~^ ERROR the size for values of type `[()]` cannot be known at compilation time
//~| ERROR `[()]` is forbidden as the type of a const generic parameter
