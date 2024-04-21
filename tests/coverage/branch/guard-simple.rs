#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

use core::hint::black_box;

fn never_taken() {
    match black_box(false) {
        _ if black_box(false) => {}
        _ if black_box(false) => {}
        _ => {}
    }
}

#[coverage(off)]
fn main() {
    never_taken();
}
