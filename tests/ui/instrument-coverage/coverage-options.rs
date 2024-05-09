//@ needs-profiler-support
//@ revisions: block branch mcdc bad
//@ compile-flags -Cinstrument-coverage

//@ [block] check-pass
//@ [block] compile-flags: -Zcoverage-options=block

//@ [branch] check-pass
//@ [branch] compile-flags: -Zcoverage-options=branch

//@ [mcdc] check-pass
//@ [mcdc] compile-flags: -Zcoverage-options=mcdc

//@ [bad] check-fail
//@ [bad] compile-flags: -Zcoverage-options=bad

fn main() {}
