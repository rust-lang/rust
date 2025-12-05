//@ check-pass
//@ compile-flags: --crate-type lib
#![deny(rustdoc)]
//~^ WARNING removed: use `rustdoc::all`
#![deny(rustdoc::all)] // has no effect when run with rustc directly
