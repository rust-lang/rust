//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ ignore-cross-compile
//
//@ compile-flags: -Z sanitizer=address -O -g
//
//@ run-fail
//@ check-run-results

use std::hint::black_box;

fn main() {
    let xs = [0, 1, 2, 3];
    // Avoid optimizing everything out.
    let xs = black_box(xs.as_ptr());
    let code = unsafe { *xs.offset(4) };
    std::process::exit(code);
}
