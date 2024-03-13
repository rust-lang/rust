//@ needs-profiler-support
//@ revisions: branch no-branch bad
//@ compile-flags -Cinstrument-coverage

//@ [branch] check-pass
//@ [branch] compile-flags: -Zcoverage-options=branch

//@ [no-branch] check-pass
//@ [no-branch] compile-flags: -Zcoverage-options=no-branch

//@ [bad] check-fail
//@ [bad] compile-flags: -Zcoverage-options=bad

fn main() {}
