#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

fn while_cond() {
    no_merge!();

    let mut a = 8;
    while a > 0 {
        a -= 1;
    }
}

fn while_cond_not() {
    no_merge!();

    let mut a = 8;
    while !(a == 0) {
        a -= 1;
    }
}

fn while_op_and() {
    no_merge!();

    let mut a = 8;
    let mut b = 4;
    while a > 0 && b > 0 {
        a -= 1;
        b -= 1;
    }
}

fn while_op_or() {
    no_merge!();

    let mut a = 4;
    let mut b = 8;
    while a > 0 || b > 0 {
        a -= 1;
        b -= 1;
    }
}

#[coverage(off)]
fn main() {
    while_cond();
    while_cond_not();
    while_op_and();
    while_op_or();
}
