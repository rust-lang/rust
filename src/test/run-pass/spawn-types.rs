/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

use std;

import str;
import comm;
import task;

type ctx = comm::chan<int>;

fn iotask(cx: ctx, ip: str) {
    assert (str::eq(ip, "localhost"));
}

fn main() {
    let p = comm::port::<int>();
    let ch = comm::chan(p);
    task::spawn {|| iotask(ch, "localhost"); };
}
