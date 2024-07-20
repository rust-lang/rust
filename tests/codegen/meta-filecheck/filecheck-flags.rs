// Arguments provided via `filecheck-flags` should be passed to `filecheck`.

//@ revisions: good bad
//@ [good] filecheck-flags: --check-prefix=CHECK-CUSTOM
//@ [bad] should-fail

// CHECK-CUSTOM: main
fn main() {}
