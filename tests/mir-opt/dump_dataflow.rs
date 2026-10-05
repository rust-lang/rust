//@ skip-filecheck
// Test graphviz dataflow output
//@ compile-flags: -Z dump-mir=main -Z dump-mir-dataflow

// EMIT_MIR dump_dataflow.main.maybe_uninit.borrowck.dot
fn main() {}
