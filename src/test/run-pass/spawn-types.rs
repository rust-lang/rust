/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

use std;

import std::str;
import std::comm;
import std::task;

type ctx = comm::_chan<int>;

fn iotask(cx: ctx, ip: str) { assert (str::eq(ip, "localhost")); }

fn main() {
    let p = comm::mk_port<int>();
    task::_spawn(bind iotask(p.mk_chan(), "localhost"));
}
