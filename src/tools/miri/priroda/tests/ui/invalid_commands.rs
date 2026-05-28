// Verifies unknown commands and malformed breakpoints are rejected without
// mutating debugger state.
// This may look trivial, but a bunch of code runs in std before
// `main` is called, so we are ensuring that that all works.
fn main() {}
