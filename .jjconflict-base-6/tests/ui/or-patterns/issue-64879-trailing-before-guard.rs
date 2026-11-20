// In this regression test we check that a trailing `|` in an or-pattern just
// before the `if` token of a `match` guard will receive parser recovery with
// an appropriate error message.
//@ run-rustfix
#![allow(dead_code)]

enum E { A, B }

fn main() {
    match E::A {
        E::A |
        E::B | //~ ERROR a trailing `|` is not allowed in an or-pattern
        if true => {
            let _recovery_witness: i32 = 0u32; //~ ERROR mismatched types
        }
        _ => {}
    }
}
