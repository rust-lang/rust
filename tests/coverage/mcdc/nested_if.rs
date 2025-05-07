#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

fn nested_if_in_condition(a: bool, b: bool, c: bool) {
    if a && if b || c { true } else { false } {
        say("yes");
    } else {
        say("no");
    }
}

fn doubly_nested_if_in_condition(a: bool, b: bool, c: bool, d: bool) {
    if a && if b || if c && d { true } else { false } { false } else { true } {
        say("yes");
    } else {
        say("no");
    }
}

fn nested_single_condition_decision(a: bool, b: bool) {
    // Decision with only 1 decision should not be instrumented by MCDC because
    // branch-coverage is equivalent to MCDC coverage in this case, and we don't
    // want to waste bitmap space for this.
    if a && if b { false } else { true } {
        say("yes");
    } else {
        say("no");
    }
}

fn nested_in_then_block_in_condition(a: bool, b: bool, c: bool, d: bool, e: bool) {
    if a && if b || c { if d && e { true } else { false } } else { false } {
        say("yes");
    } else {
        say("no");
    }
}

#[coverage(off)]
fn main() {
    nested_if_in_condition(true, false, false);
    nested_if_in_condition(true, true, true);
    nested_if_in_condition(true, false, true);
    nested_if_in_condition(false, true, true);

    doubly_nested_if_in_condition(true, false, false, true);
    doubly_nested_if_in_condition(true, true, true, true);
    doubly_nested_if_in_condition(true, false, true, true);
    doubly_nested_if_in_condition(false, true, true, true);

    nested_single_condition_decision(true, true);
    nested_single_condition_decision(true, false);
    nested_single_condition_decision(false, false);

    nested_in_then_block_in_condition(false, false, false, false, false);
    nested_in_then_block_in_condition(true, false, false, false, false);
    nested_in_then_block_in_condition(true, true, false, false, false);
    nested_in_then_block_in_condition(true, false, true, false, false);
    nested_in_then_block_in_condition(true, false, true, true, false);
    nested_in_then_block_in_condition(true, false, true, false, true);
    nested_in_then_block_in_condition(true, false, true, true, true);
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}
