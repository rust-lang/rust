//@ aux-build:match_non_exhaustive_lib.rs
//@ dont-require-annotations: NOTE

#![deny(unreachable_patterns)]

#[non_exhaustive]
//~^ NOTE `#[non_exhaustive]` does not affect exhaustiveness checking within the defining crate
//~^^ NOTE `#[non_exhaustive]` does not affect exhaustiveness checking within the defining crate
enum E1 { A, B }

extern crate match_non_exhaustive_lib;
use match_non_exhaustive_lib::E2;

fn foo(e: E1) {
    match e {
        E1::A | E1::B => {}
        _ => {}
        //~^ ERROR unreachable pattern
    }
}

fn bar(e: Option<E1>) {
    match e {
        Some(E1::A | E1::B | _) => {}
        //~^ ERROR unreachable pattern
        None => {}
    }
}

fn baz(e: E2) {
    match e {
        E2::A | E2::B => {}
        _ => {}
    }
}

fn main() {}
