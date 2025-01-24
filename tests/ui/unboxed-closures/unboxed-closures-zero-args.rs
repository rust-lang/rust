//@ run-pass
#![allow(unused_mut)]

fn main() {
    let mut zero = || {};
    let () = zero();
}
