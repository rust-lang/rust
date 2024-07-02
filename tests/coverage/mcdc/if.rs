#![feature(coverage_attribute)]
//@ edition: 2021
//@ min-llvm-version: 18
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

fn check_neither(a: bool, b: bool) {
    if a && b {
        say("a and b");
    } else {
        say("not both");
    }
}

fn check_a(a: bool, b: bool) {
    if a && b {
        say("a and b");
    } else {
        say("not both");
    }
}

fn check_b(a: bool, b: bool) {
    if a && b {
        say("a and b");
    } else {
        say("not both");
    }
}

fn check_both(a: bool, b: bool) {
    if a && b {
        say("a and b");
    } else {
        say("not both");
    }
}

fn check_tree_decision(a: bool, b: bool, c: bool) {
    // This expression is intentionally written in a way
    // where 100% branch coverage indicates 100% mcdc coverage.
    if a && (b || c) {
        say("pass");
    } else {
        say("reject");
    }
}

fn check_not_tree_decision(a: bool, b: bool, c: bool) {
    // Contradict to `check_tree_decision`,
    // 100% branch coverage of this expression does not mean indicates 100% mcdc coverage.
    if (a || b) && c {
        say("pass");
    } else {
        say("reject");
    }
}

fn nested_if(a: bool, b: bool, c: bool) {
    if a || b {
        say("a or b");
        if b && c {
            say("b and c");
        }
    } else {
        say("neither a nor b");
    }
}

#[coverage(off)]
fn main() {
    check_neither(false, false);
    check_neither(false, true);

    check_a(true, true);
    check_a(false, true);

    check_b(true, true);
    check_b(true, false);

    check_both(false, true);
    check_both(true, true);
    check_both(true, false);

    check_tree_decision(false, true, true);
    check_tree_decision(true, true, false);
    check_tree_decision(true, false, false);
    check_tree_decision(true, false, true);

    check_not_tree_decision(false, true, true);
    check_not_tree_decision(true, true, false);
    check_not_tree_decision(true, false, false);
    check_not_tree_decision(true, false, true);

    nested_if(true, false, true);
    nested_if(true, true, true);
    nested_if(true, true, false);
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}
