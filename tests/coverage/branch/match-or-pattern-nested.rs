#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

use core::hint::black_box;

fn foo(a: i32, b: i32) {
    match black_box((a, b)) {
        (1, 2 | 3) | (2, 4 | 5) => {}
        _ => {}
    }
}

#[coverage(off)]
fn main() {
    foo(1, 2);
    foo(1, 99);
    foo(2, 5);
}
