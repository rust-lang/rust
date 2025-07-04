#![feature(coverage_attribute)]
//@ edition: 2024
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

fn if_let(input: Option<&str>) {
    no_merge!();

    if let Some(x) = input {
        say(x);
    } else {
        say("none");
    }
    say("done");
}

fn if_let_chain(a: Option<&str>, b: Option<&str>) {
    if let Some(x) = a
        && let Some(y) = b
    {
        say(x);
        say(y);
    } else {
        say("not both");
    }
    say("done");
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}

#[coverage(off)]
fn main() {
    if_let(Some("x"));
    if_let(Some("x"));
    if_let(None);

    for _ in 0..8 {
        if_let_chain(Some("a"), Some("b"));
    }
    for _ in 0..4 {
        if_let_chain(Some("a"), None);
    }
    for _ in 0..2 {
        if_let_chain(None, Some("b"));
    }
    if_let_chain(None, None);
}

// FIXME(#124118) Actually instrument if-let and let-chains for branch coverage.
