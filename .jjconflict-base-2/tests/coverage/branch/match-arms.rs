#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

// Tests for branch coverage of various kinds of match arms.

// Helper macro to prevent start-of-function spans from being merged into
// spans on the lines we care about.
macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

#[derive(Clone, Copy, Debug)]
enum Enum {
    A(u32),
    B(u32),
    C(u32),
    D(u32),
}

fn match_arms(value: Enum) {
    no_merge!();

    match value {
        Enum::D(d) => consume(d),
        Enum::C(c) => consume(c),
        Enum::B(b) => consume(b),
        Enum::A(a) => consume(a),
    }

    consume(0);
}

fn or_patterns(value: Enum) {
    no_merge!();

    match value {
        Enum::D(x) | Enum::C(x) => consume(x),
        Enum::B(y) | Enum::A(y) => consume(y),
    }

    consume(0);
}

fn guards(value: Enum, cond: bool) {
    no_merge!();

    match value {
        Enum::D(d) if cond => consume(d),
        Enum::C(c) if cond => consume(c),
        Enum::B(b) if cond => consume(b),
        Enum::A(a) if cond => consume(a),
        _ => consume(0),
    }

    consume(0);
}

#[coverage(off)]
fn consume<T>(x: T) {
    core::hint::black_box(x);
}

#[coverage(off)]
fn main() {
    #[coverage(off)]
    fn call_everything(e: Enum) {
        match_arms(e);
        or_patterns(e);
        for cond in [false, false, true] {
            guards(e, cond);
        }
    }

    call_everything(Enum::A(0));
    for b in 0..2 {
        call_everything(Enum::B(b));
    }
    for c in 0..4 {
        call_everything(Enum::C(c));
    }
    for d in 0..8 {
        call_everything(Enum::D(d));
    }
}

// FIXME(#124118) Actually instrument match arms for branch coverage.
