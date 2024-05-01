#![feature(coverage_attribute)]
//@ edition: 2021
//@ min-llvm-version: 18
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

fn good() {
    let [a, b, c, d, e, f] = <[bool; 6]>::default();
    if a && b && c && d && e && f {
        core::hint::black_box("hello");
    }
}

fn bad() {
    let [a, b, c, d, e, f, g] = <[bool; 7]>::default();
    if a && b && c && d && e && f && g {
        core::hint::black_box("hello");
    }
}

#[coverage(off)]
fn main() {
    good();
    bad();
}
