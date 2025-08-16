// The default filecheck prefix is `CHECK`, others need to be specified

//@ revisions: GOOD BAD
//@ [BAD] should-fail
//@ [BAD] filecheck-flags: --check-prefixes=BAD

// CHECK: main
// BAD: text that should not match
fn main() {}
