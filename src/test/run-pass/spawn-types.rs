/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

use std;

import std::str;
import std::comm;
import std::task;

type ctx = comm::chan<int>;

fn# iotask(&&args: (ctx, str)) {
    let (cx, ip) = args;
    assert (str::eq(ip, "localhost"));
}

fn main() {
    let p = comm::port::<int>();
    task::spawn((comm::chan(p), "localhost"), iotask);
}
