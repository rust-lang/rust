// The current revision name is registered as a filecheck prefix.

//@ revisions: GOOD BAD
//@ [BAD] should-fail

// GOOD: main
// BAD: text that should not match
fn main() {}
