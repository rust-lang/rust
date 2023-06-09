// run-pass
#![allow(non_camel_case_types)]

// aux-build:namespaced_enum_emulate_flat.rs

// pretty-expanded FIXME #23616

extern crate namespaced_enum_emulate_flat;

use namespaced_enum_emulate_flat::{Foo, A, B, C};
use namespaced_enum_emulate_flat::nest::{Bar, D, E, F};

fn _f(f: Foo) {
    match f {
        A | B(_) | C { .. } => {}
    }
}

fn _f2(f: Bar) {
    match f {
        D | E(_) | F { .. } => {}
    }
}

pub fn main() {}
