#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

fn accept_7_conditions(bool_arr: [bool; 7]) {
    let [a, b, c, d, e, f, g] = bool_arr;
    if a && b && c && d && e && f && g {
        core::hint::black_box("hello");
    }
}

#[coverage(off)]
fn main() {
    accept_7_conditions([false; 7]);
    accept_7_conditions([true; 7]);
}
