//@ run-pass
#![allow(dead_code)]

struct Empty;

// This used to cause an ICE
#[allow(improper_c_fn_definitions)]
extern "C" fn ice(_a: Empty) {}

fn main() {
}
