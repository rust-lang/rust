// Verifies Priroda can start on the simplest passing Rust program and accept
// a scripted `quit` command.
// This may look trivial, but a bunch of code runs in std before
// `main` is called, so we are ensuring that that all works.
fn main() {}
