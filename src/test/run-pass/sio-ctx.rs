// Seems to hang for long periods, probably with RUST_THREADS > 1. Issue #810
// xfail-stage1
// xfail-stage2
// xfail-stage3

use std;
import std::sio;

fn main() {
  let cx: sio::ctx = sio::new();
  sio::destroy(cx);
}
