//@ compile-flags: -C lto -C embed-bitcode=no

fn main() {}

//~? ERROR options `-C embed-bitcode=no` and `-C lto` are incompatible
