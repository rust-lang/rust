// -*- rust -*-

// Regression tests for circular_buffer when using a unit
// that has a size that is not a power of two
use std;
import std::option;
import std::uint;
import std::comm;
import std::comm::mk_port;
import std::comm::send;

// A 12-byte unit to send over the channel
type record = {val1: u32, val2: u32, val3: u32};


// Assuming that the default buffer size needs to hold 8 units,
// then the minimum buffer size needs to be 96. That's not a
// power of two so needs to be rounded up. Don't trigger any
// assertions.
fn test_init() {
    let myport = mk_port::<record>();
    let mychan = myport.mk_chan();
    let val: record = {val1: 0u32, val2: 0u32, val3: 0u32};
    send(mychan, val);
}


// Dump lots of items into the channel so it has to grow.
// Don't trigger any assertions.
fn test_grow() {
    let myport: comm::_port<record> = comm::mk_port();
    let mychan = myport.mk_chan();
    for each i: uint  in uint::range(0u, 100u) {
        let val: record = {val1: 0u32, val2: 0u32, val3: 0u32};
        comm::send(mychan, val);
    }
}


// Don't allow the buffer to shrink below it's original size
fn test_shrink1() {
    let myport = comm::mk_port::<i8>();
    let mychan = myport.mk_chan();
    send(mychan, 0i8);
    let x = myport.recv();
}

fn test_shrink2() {
    let myport = mk_port::<record>();
    let mychan = myport.mk_chan();
    for each i: uint  in uint::range(0u, 100u) {
        let val: record = {val1: 0u32, val2: 0u32, val3: 0u32};
        send(mychan, val);
    }
    for each i: uint  in uint::range(0u, 100u) { let x = myport.recv(); }
}


// Test rotating the buffer when the unit size is not a power of two
fn test_rotate() {
    let myport = mk_port::<record>();
    let mychan = myport.mk_chan();
    for each i: uint  in uint::range(0u, 100u) {
        let val = {val1: i as u32, val2: i as u32, val3: i as u32};
        send(mychan, val);
        let x = myport.recv();
        assert (x.val1 == i as u32);
        assert (x.val2 == i as u32);
        assert (x.val3 == i as u32);
    }
}


// Test rotating and growing the buffer when
// the unit size is not a power of two
fn test_rotate_grow() {
    let myport = mk_port::<record>();
    let mychan = myport.mk_chan();
    for each j: uint  in uint::range(0u, 10u) {
        for each i: uint  in uint::range(0u, 10u) {
            let val: record =
                {val1: i as u32, val2: i as u32, val3: i as u32};
            send(mychan, val);
        }
        for each i: uint  in uint::range(0u, 10u) {
            let x = myport.recv();
            assert (x.val1 == i as u32);
            assert (x.val2 == i as u32);
            assert (x.val3 == i as u32);
        }
    }
}

fn main() {
    test_init();
    test_grow();
    test_shrink1();
    test_shrink2();
    test_rotate();
    test_rotate_grow();
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
