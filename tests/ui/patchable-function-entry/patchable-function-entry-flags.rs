//@ compile-flags: -Z patchable-function-entry=1,2

fn main() {}

//~? ERROR incorrect value `1,2` for unstable option `patchable-function-entry`
