// Verifies EOF exits the debugger loop cleanly without requiring an explicit
// quit command.
// This may look trivial, but a bunch of code runs in std before
// `main` is called, so we are ensuring that that all works.
fn main() {}
