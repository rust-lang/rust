//
// Test pretty printing of macro with braces but without terminating semicolon,
// this used to panic before fix.

//@ pretty-compare-only
//@ pp-exact

fn main() { b! {} c }
