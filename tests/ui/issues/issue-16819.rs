//@ run-pass
#![allow(unused_variables)]
// `#[cfg]` on struct field permits empty unusable struct

struct S {
    #[cfg(FALSE)]
    a: int,
}

fn main() {
    let s = S {};
}
