#![feature(coverage_attribute)]
//@ test-mir-pass: InstrumentCoverage
//@ compile-flags: -Cinstrument-coverage -Zno-profiler-runtime -Zcoverage-options=branch
// skip-filecheck

enum Enum {
    A(u32),
    B(u32),
    C(u32),
    D(u32),
}

// EMIT_MIR branch_match_arms.main.InstrumentCoverage.diff
fn main() {
    match Enum::A(0) {
        Enum::D(d) => consume(d),
        Enum::C(c) => consume(c),
        Enum::B(b) => consume(b),
        Enum::A(a) => consume(a),
    }
}

#[inline(never)]
#[coverage(off)]
fn consume(x: u32) {
    core::hint::black_box(x);
}
