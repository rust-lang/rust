//@ build-fail
//@ compile-flags: -C lto -C prefer-dynamic

fn main() {}

//~? ERROR cannot prefer dynamic linking when performing LTO
