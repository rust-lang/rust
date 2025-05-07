#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

// This test ensures that boolean expressions that are not inside control flow
// decisions are correctly instrumented.

use core::hint::black_box;

fn assign_and(a: bool, b: bool) {
    let x = a && b;
    black_box(x);
}

fn assign_or(a: bool, b: bool) {
    let x = a || b;
    black_box(x);
}

fn assign_3(a: bool, b: bool, c: bool) {
    let x = a || b && c;
    black_box(x);
}

fn assign_3_bis(a: bool, b: bool, c: bool) {
    let x = a && b || c;
    black_box(x);
}

fn right_comb_tree(a: bool, b: bool, c: bool, d: bool, e: bool) {
    let x = a && (b && (c && (d && (e))));
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
    assign_and(true, false);
    assign_and(true, true);
    assign_and(false, false);

    assign_or(true, false);
    assign_or(true, true);
    assign_or(false, false);

    assign_3(true, false, false);
    assign_3(true, true, false);
    assign_3(false, false, true);
    assign_3(false, true, true);

    assign_3_bis(true, false, false);
    assign_3_bis(true, true, false);
    assign_3_bis(false, false, true);
    assign_3_bis(false, true, true);

    right_comb_tree(false, false, false, true, true);
    right_comb_tree(true, false, false, true, true);
    right_comb_tree(true, true, true, true, true);

    func_call(true, false);
    func_call(true, true);
    func_call(false, false);
}
