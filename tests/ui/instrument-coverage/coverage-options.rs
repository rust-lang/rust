//@ revisions: branch no-branch bad
//@ compile-flags -Cinstrument-coverage -Zno-profiler-runtime

//@ [branch] check-pass
//@ [branch] compile-flags: -Zcoverage-options=branch

//@ [no-branch] check-pass
//@ [no-branch] compile-flags: -Zcoverage-options=no-branch

//@ [bad] check-fail
//@ [bad] compile-flags: -Zcoverage-options=bad

fn main() {}
