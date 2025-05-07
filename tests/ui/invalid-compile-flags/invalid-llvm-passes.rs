//@ build-fail
//@ compile-flags: -Cpasses=unknown-pass

fn main() {}

//~? ERROR failed to run LLVM passes: unknown pass name 'unknown-pass'
