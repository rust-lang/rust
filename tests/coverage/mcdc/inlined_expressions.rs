#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=mcdc -Copt-level=z -Cllvm-args=--inline-threshold=0
//@ llvm-cov-flags: --show-branches=count --show-mcdc

#[inline(always)]
fn inlined_instance(a: bool, b: bool) -> bool {
    a && b
}

#[coverage(off)]
fn main() {
    let _ = inlined_instance(true, false);
    let _ = inlined_instance(false, true);
    let _ = inlined_instance(true, true);
}
