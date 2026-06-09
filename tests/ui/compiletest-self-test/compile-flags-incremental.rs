//@ revisions: good bad bad-space
//@ check-pass

//@[bad] compile-flags: -Cincremental=true
//@[bad] should-fail

//@[bad-space] compile-flags:  -C  incremental=dir
//@[bad-space] should-fail

fn main() {}

// Tests should not try to manually enable incremental compilation with
// `-Cincremental`, because that typically results in stray directories being
// created in the repository root.
//
// Instead, use the `//@ incremental` directive, which instructs compiletest
// to handle the details of passing `-Cincremental` with a fresh directory.
