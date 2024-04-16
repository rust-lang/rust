//@ known-bug: #120600
#![feature(never_type, never_type_fallback)]

enum E { Bar(!) }

fn f(a: &E, b: &E) {
    match (a, b) {
        (E::Bar(a), E::Bar(b)) => { *a == *b; }
        _ => {}
    }
}

pub fn main() {}
