// The current revision name is registered as a filecheck prefix.

//@ revisions: good bad
//@ [bad] should-fail

// CHECK-GOOD: main
// CHECK-BAD: text that should not match
fn main() {}
