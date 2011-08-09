// Fails periodically, probably with RUST_THREADS > 1
// xfail-stage1
// xfail-stage2
// xfail-stage3

use std;
import std::sio;

fn main() {
  let cx: sio::ctx = sio::new();
  sio::destroy(cx);
}
