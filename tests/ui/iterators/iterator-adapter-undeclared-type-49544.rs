//@ aux-build:iterator-adapter-undeclared-type-49544.rs
//@ check-pass

extern crate iterator_adapter_undeclared_type_49544 as lib;
use lib::foo;

fn main() {
    let _ = foo();
}

// https://github.com/rust-lang/rust/issues/49544
