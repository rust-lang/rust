// -*- rust -*-

// Regression test for circular_buffer initialization

use std;

import std.option;
import std._uint;
import std._vec;

// 12-byte unit for the channel buffer. Assuming that the default
// buffer size needs to hold 8 units, then the minimum buffer size
// needs to be 96. That's not a power of two so needs to be rounded up.
type record = rec(i32 val1, i32 val2, i32 val3);

impure fn worker(chan[record] channel) {
    let record val = rec(val1=0i32, val2=0i32, val3=0i32);
    channel <| val;
}

impure fn main() {
    let port[record] myport = port();
    auto mychan = chan(myport);

    auto temp = spawn worker(mychan);
    auto val <- myport;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
