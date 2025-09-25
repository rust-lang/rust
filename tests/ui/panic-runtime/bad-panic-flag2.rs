//@ compile-flags:-C panic

fn main() {}

//~? ERROR codegen option `panic` requires either `unwind`, `abort`, or `immediate-abort`
