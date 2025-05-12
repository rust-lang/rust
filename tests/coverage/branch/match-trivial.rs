#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

// When instrumenting match expressions for branch coverage, make sure we don't
// cause an ICE or produce weird coverage output for matches with <2 arms.

// Helper macro to prevent start-of-function spans from being merged into
// spans on the lines we care about.
macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

enum Uninhabited {}
enum Trivial {
    Value,
}

fn _uninhabited(x: Uninhabited) {
    no_merge!();

    match x {}

    consume("done");
}

fn trivial(x: Trivial) {
    no_merge!();

    match x {
        Trivial::Value => consume("trivial"),
    }

    consume("done");
}

#[coverage(off)]
fn consume<T>(x: T) {
    core::hint::black_box(x);
}

#[coverage(off)]
fn main() {
    trivial(Trivial::Value);
}
