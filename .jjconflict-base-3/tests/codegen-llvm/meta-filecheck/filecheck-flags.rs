// Arguments provided via `filecheck-flags` should be passed to `filecheck`.

//@ revisions: good bad
//@ [good] filecheck-flags: --check-prefix=CUSTOM
//@ [bad] should-fail

// CUSTOM: main
fn main() {}
