//@ known-bug: rust-lang/rust#126680
//@ compile-flags: -Zvalidate-mir
#![feature(type_alias_impl_trait)]
type Bar = impl std::fmt::Display;

use std::path::Path;

struct A {
    pub func: fn(check: Bar, b: Option<&Path>),
}

#[define_opaque(Bar)]
fn foo() -> A {
    A {
        func: |check, b| {
            if check {
                ()
            } else if let Some(_) = b.and_then(|p| p.parent()) {
                ()
            }
        },
    }
}

fn main() {}
