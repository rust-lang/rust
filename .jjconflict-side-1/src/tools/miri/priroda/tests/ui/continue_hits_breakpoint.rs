// Verifies continue stops when execution reaches a registered source-location
// breakpoint.
// This may look trivial, but a bunch of code runs in std before
// `main` is called, so we are ensuring that that all works.
fn main() {
    let _value = 0;
}
