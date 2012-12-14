// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Regression tests for circular_buffer when using a unit
// that has a size that is not a power of two

// A 12-byte unit to core::oldcomm::send over the channel
type record = {val1: u32, val2: u32, val3: u32};


// Assuming that the default buffer size needs to hold 8 units,
// then the minimum buffer size needs to be 96. That's not a
// power of two so needs to be rounded up. Don't trigger any
// assertions.
fn test_init() {
    let myport = core::oldcomm::Port();
    let mychan = core::oldcomm::Chan(&myport);
    let val: record = {val1: 0u32, val2: 0u32, val3: 0u32};
    core::oldcomm::send(mychan, val);
}


// Dump lots of items into the channel so it has to grow.
// Don't trigger any assertions.
fn test_grow() {
    let myport = core::oldcomm::Port();
    let mychan = core::oldcomm::Chan(&myport);
    for uint::range(0u, 100u) |i| {
        let val: record = {val1: 0u32, val2: 0u32, val3: 0u32};
        core::oldcomm::send(mychan, val);
    }
}


// Don't allow the buffer to shrink below it's original size
fn test_shrink1() {
    let myport = core::oldcomm::Port();
    let mychan = core::oldcomm::Chan(&myport);
    core::oldcomm::send(mychan, 0i8);
    let x = core::oldcomm::recv(myport);
}

fn test_shrink2() {
    let myport = core::oldcomm::Port();
    let mychan = core::oldcomm::Chan(&myport);
    for uint::range(0u, 100u) |_i| {
        let val: record = {val1: 0u32, val2: 0u32, val3: 0u32};
        core::oldcomm::send(mychan, val);
    }
    for uint::range(0u, 100u) |_i| { let x = core::oldcomm::recv(myport); }
}


// Test rotating the buffer when the unit size is not a power of two
fn test_rotate() {
    let myport = core::oldcomm::Port();
    let mychan = core::oldcomm::Chan(&myport);
    for uint::range(0u, 100u) |i| {
        let val = {val1: i as u32, val2: i as u32, val3: i as u32};
        core::oldcomm::send(mychan, val);
        let x = core::oldcomm::recv(myport);
        assert (x.val1 == i as u32);
        assert (x.val2 == i as u32);
        assert (x.val3 == i as u32);
    }
}


// Test rotating and growing the buffer when
// the unit size is not a power of two
fn test_rotate_grow() {
    let myport = core::oldcomm::Port::<record>();
    let mychan = core::oldcomm::Chan(&myport);
    for uint::range(0u, 10u) |j| {
        for uint::range(0u, 10u) |i| {
            let val: record =
                {val1: i as u32, val2: i as u32, val3: i as u32};
            core::oldcomm::send(mychan, val);
        }
        for uint::range(0u, 10u) |i| {
            let x = core::oldcomm::recv(myport);
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
// End:
