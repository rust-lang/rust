// Verifies every current spelling of MIR-instruction stepping advances
// execution and reports a location.
// This may look trivial, but a bunch of code runs in std before
// `main` is called, so we are ensuring that that all works.
fn main() {}
