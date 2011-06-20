/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3

use std;

import std::str;

type ctx = chan[int];

fn iotask(ctx cx, str ip) {
  assert(str::eq(ip, "localhost"));
}

fn main() {
  let port[int] p = port();
  spawn iotask(chan(p), "localhost");
}
