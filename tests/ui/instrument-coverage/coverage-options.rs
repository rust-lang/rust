//@ revisions: block branch condition bad
//@ compile-flags -Cinstrument-coverage -Zno-profiler-runtime

//@ [block] check-pass
//@ [block] compile-flags: -Zcoverage-options=block

//@ [branch] check-pass
//@ [branch] compile-flags: -Zcoverage-options=branch

//@ [condition] check-pass
//@ [condition] compile-flags: -Zcoverage-options=condition

//@ [bad] check-fail
//@ [bad] compile-flags: -Zcoverage-options=bad

fn main() {}

//[bad]~? ERROR incorrect value `bad` for unstable option `coverage-options`
