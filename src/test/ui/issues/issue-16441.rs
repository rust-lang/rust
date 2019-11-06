// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

struct Empty;

// This used to cause an ICE
#[allow(improper_ctypes)]
extern "C" fn ice(_a: Empty) {}

fn main() {
}
