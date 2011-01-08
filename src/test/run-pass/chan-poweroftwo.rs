// -*- rust -*-

// Regression tests for circular_buffer when using a unit
// that has a size that is not a power of two

use std;

import std.option;
import std._uint;
import std._vec;

// A 12-byte unit to send over the channel
type record = rec(i32 val1, i32 val2, i32 val3);

// Assuming that the default buffer size needs to hold 8 units,
// then the minimum buffer size needs to be 96. That's not a
// power of two so needs to be rounded up. Don't trigger any
// assertions.
impure fn test_init() {
    let port[record] myport = port();
    auto mychan = chan(myport);

    let record val = rec(val1=0i32, val2=0i32, val3=0i32);

    mychan <| val;
}

// Dump lots of items into the channel so it has to grow.
// Don't trigger any assertions.
impure fn test_grow() {
    let port[record] myport = port();
    auto mychan = chan(myport);

    let record val = rec(val1=0i32, val2=0i32, val3=0i32);

    for each (uint i in _uint.range(0u, 100u)) {
        mychan <| val;
    }
}

// Don't allow the buffer to shrink below it's original size
impure fn test_shrink() {
    let port[i8] myport = port();
    auto mychan = chan(myport);

    mychan <| 0i8;
    auto x <- myport;
}

impure fn main() {
    test_init();
    test_grow();
    test_shrink();
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
