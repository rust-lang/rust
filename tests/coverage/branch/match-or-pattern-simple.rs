#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

use core::hint::black_box;

fn foo(a: i32) {
    match black_box(a) {
        1 | 2 => {
            consume(1);
        }
        _ => {
            consume(2);
        }
    }
}

#[coverage(off)]
fn consume<T>(x: T) {
    core::hint::black_box(x);
}

#[coverage(off)]
fn main() {
    foo(1);
    foo(3);
    foo(4);
}
