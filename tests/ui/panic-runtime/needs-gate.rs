// gate-test-needs_panic_runtime
// gate-test-panic_runtime

#![panic_runtime] //~ ERROR: is an experimental feature
#![needs_panic_runtime] //~ ERROR: is an experimental feature

fn main() {}
