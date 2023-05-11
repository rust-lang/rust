//! This is a minimal reproducer for the ICE in https://github.com/rust-lang/rust-clippy/pull/6179.
//! The ICE is mainly caused by using `hir_ty_to_ty`. See the discussion in the PR for details.

#![warn(clippy::use_self)]
#![allow(dead_code, clippy::let_with_type_underscore)]

struct Foo;

impl Foo {
    fn new() -> Self {
        impl Foo {
            fn bar() {}
        }

        let _: _ = 1;

        Self {}
    }
}

fn main() {}
