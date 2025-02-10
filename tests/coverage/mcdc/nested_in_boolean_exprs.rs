#![feature(coverage_attribute)]
//@ edition: 2021
//@ min-llvm-version: 19
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

use core::hint::black_box;

fn assign_nested_if(a: bool, b: bool, c: bool) {
    let x = a || if b && c { false } else { true };
    black_box(x);
}

fn foo(a: bool) -> bool {
    black_box(a)
}

fn assign_nested_func_call(a: bool, b: bool, c: bool) {
    let x = a || foo(b && c);
    black_box(x);
}

fn func_call_nested_if(a: bool, b: bool, c: bool) {
    let x = foo(a || if b && c { false } else { true });
    black_box(x);
}

fn func_call_with_unary_not(a: bool, b: bool) {
    let x = a || foo(!b);
    black_box(x);
}

#[coverage(off)]
fn main() {
    assign_nested_if(true, false, true);
    assign_nested_if(false, false, true);
    assign_nested_if(false, true, true);

    assign_nested_func_call(true, false, true);
    assign_nested_func_call(false, false, true);
    assign_nested_func_call(false, true, true);

    func_call_nested_if(true, false, true);
    func_call_nested_if(false, false, true);
    func_call_nested_if(false, true, true);

    func_call_with_unary_not(true, false);
    func_call_with_unary_not(false, true);
}
