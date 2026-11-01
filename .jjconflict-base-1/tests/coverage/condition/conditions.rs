#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=condition
//@ llvm-cov-flags: --show-branches=count

use core::hint::black_box;

fn simple_assign(a: bool) {
    let x = a;
    black_box(x);
}

fn assign_and(a: bool, b: bool) {
    let x = a && b;
    black_box(x);
}

fn assign_or(a: bool, b: bool) {
    let x = a || b;
    black_box(x);
}

fn assign_3_or_and(a: bool, b: bool, c: bool) {
    let x = a || b && c;
    black_box(x);
}

fn assign_3_and_or(a: bool, b: bool, c: bool) {
    let x = a && b || c;
    black_box(x);
}

fn foo(a: bool) -> bool {
    black_box(a)
}

fn func_call(a: bool, b: bool) {
    foo(a && b);
}

#[coverage(off)]
fn main() {
    simple_assign(true);
    simple_assign(false);

    assign_and(true, false);
    assign_and(true, true);
    assign_and(false, false);

    assign_or(true, false);
    assign_or(true, true);
    assign_or(false, false);

    assign_3_or_and(true, false, false);
    assign_3_or_and(true, true, false);
    assign_3_or_and(false, false, true);
    assign_3_or_and(false, true, true);

    assign_3_and_or(true, false, false);
    assign_3_and_or(true, true, false);
    assign_3_and_or(false, false, true);
    assign_3_and_or(false, true, true);

    func_call(true, false);
    func_call(true, true);
    func_call(false, false);
}
