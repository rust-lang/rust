// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

struct Empty;

// This used to cause an ICE
extern "C" fn ice(_a: Empty) {}

fn main() {
}
