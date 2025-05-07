//@ rustc-env:RUST_MIN_STACK=banana

fn main() {}

//~? ERROR `RUST_MIN_STACK` should be a number of bytes, but was "banana"
