// Ensures -Zmir-opt-level=2 (specifically, inlining) is not allowed with -Zinstrument-coverage.
// Regression test for issue #80060.
//
// needs-profiler-support
// build-pass
// compile-flags: -Zmir-opt-level=2 -Zinstrument-coverage
#[inline(never)]
fn foo() {}

pub fn baz() {
    bar();
}

#[inline(always)]
fn bar() {
    foo();
}

fn main() {
    bar();
}
