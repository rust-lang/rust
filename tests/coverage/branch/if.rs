#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

fn branch_not(a: bool) {
    no_merge!();

    if a {
        say("a")
    }
    if !a {
        say("not a");
    }
    if !!a {
        say("not not a");
    }
    if !!!a {
        say("not not not a");
    }
}

fn branch_not_as(a: bool) {
    no_merge!();

    if !(a as bool) {
        say("not (a as bool)");
    }
    if !!(a as bool) {
        say("not not (a as bool)");
    }
    if !!!(a as bool) {
        say("not not (a as bool)");
    }
}

fn branch_and(a: bool, b: bool) {
    no_merge!();

    if a && b {
        say("both");
    } else {
        say("not both");
    }
}

fn branch_or(a: bool, b: bool) {
    no_merge!();

    if a || b {
        say("either");
    } else {
        say("neither");
    }
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}

#[coverage(off)]
fn main() {
    for a in [false, true, true] {
        branch_not(a);
        branch_not_as(a);
    }

    for a in [false, true, true, true, true] {
        for b in [false, true, true] {
            branch_and(a, b);
            branch_or(a, b);
        }
    }
}
