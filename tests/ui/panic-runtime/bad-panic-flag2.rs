//@ compile-flags:-C panic

fn main() {}

//~? ERROR codegen option `panic` requires either `unwind` or `abort`
