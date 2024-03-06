// skip-filecheck
// Test graphviz output
//@ compile-flags: -Z dump-mir-graphviz

// EMIT_MIR graphviz.main.built.after.dot
fn main() {}
