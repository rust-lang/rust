/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

use std;

import std::str;

type ctx = chan[int];

fn iotask(cx: ctx, ip: str) { assert (str::eq(ip, "localhost")); }

fn main() { let p: port[int] = port(); spawn iotask(chan(p), "localhost"); }