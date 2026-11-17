//! `-Z raise-optimize-speed` rejects a setting it does not know.

//@ compile-flags: -Zraise-optimize-speed=modules,sideways

//~? ERROR incorrect value `modules,sideways` for unstable option `raise-optimize-speed`

fn main() {}
