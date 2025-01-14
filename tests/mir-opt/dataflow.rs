// skip-filecheck
// Test graphviz dataflow output
//@ compile-flags: -Z dump-mir=main -Z dump-mir-dataflow

// EMIT_MIR dataflow.main.maybe_init.borrowck.dot
fn main() {}
