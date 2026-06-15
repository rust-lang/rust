//@ revisions: no_unstable block branch condition
//@ compile-flags: -Zno-profiler-runtime

//@ [no_unstable] check-fail
//@ [no_unstable] compile-flags: -Cinstrument-coverage=single-byte

//@ [block] check-pass
//@ [block] compile-flags: -Cinstrument-coverage=single-byte -Zunstable-options
//@ [block] compile-flags: -Zcoverage-options=block

//@ [branch] check-fail
//@ [branch] compile-flags: -Cinstrument-coverage=single-byte -Zunstable-options
//@ [branch] compile-flags: -Zcoverage-options=branch

//@ [condition] check-fail
//@ [condition] compile-flags: -Cinstrument-coverage=single-byte -Zunstable-options
//@ [condition] compile-flags: -Zcoverage-options=condition

fn main() {}

//[no_unstable]~? ERROR `-C instrument-coverage=single-byte` requires `-Z unstable-options`
//[branch]~? ERROR `-C instrument-coverage=single-byte` is not compatible with branch or condition coverage
//[condition]~? ERROR `-C instrument-coverage=single-byte` is not compatible with branch or condition coverage
