#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

fn let_else(value: Option<&str>) {
    no_merge!();

    let Some(x) = value else {
        say("none");
        return;
    };

    say(x);
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}

#[coverage(off)]
fn main() {
    let_else(Some("x"));
    let_else(Some("x"));
    let_else(None);
}

// FIXME(#124118) Actually instrument let-else for branch coverage.
