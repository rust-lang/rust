//@ needs-profiler-support
//@ revisions: branch no-branch bad
//@ compile-flags -Cinstrument-coverage

//@ [branch] check-pass
//@ [branch] compile-flags: -Zcoverage-options=branch

//@ [no-branch] check-pass
//@ [no-branch] compile-flags: -Zcoverage-options=no-branch

//@ [mcdc] check-pass
//@ [mcdc] compile-flags: -Zcoverage-options=mcdc

//@ [bad] check-fail
//@ [bad] compile-flags: -Zcoverage-options=bad

//@ [conflict] check-fail
//@ [conflict] compile-flags: -Zcoverage-options=no-branch,mcdc

fn main() {}
