/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

use std;


type ctx = comm::Chan<int>;

fn iotask(cx: ctx, ip: ~str) {
    assert (ip == ~"localhost");
}

fn main() {
    let p = comm::Port::<int>();
    let ch = comm::Chan(p);
    task::spawn(|| iotask(ch, ~"localhost") );
}
