//@ compile-flags:-C panic=foo

fn main() {}

//~? ERROR incorrect value `foo` for codegen option `panic` - either `unwind` or `abort` was expected
