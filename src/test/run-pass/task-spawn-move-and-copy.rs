// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let p = comm::Port::<uint>();
    let ch = comm::Chan(&p);

    let x = ~1;
    let x_in_parent = ptr::addr_of(&(*x)) as uint;

    let y = ~2;
    let y_in_parent = ptr::addr_of(&(*y)) as uint;

    task::spawn(fn~(copy ch, copy y, move x) {
        let x_in_child = ptr::addr_of(&(*x)) as uint;
        comm::send(ch, x_in_child);

        let y_in_child = ptr::addr_of(&(*y)) as uint;
        comm::send(ch, y_in_child);
    });
    // Ensure last-use analysis doesn't move y to child.
    let _q = y;

    let x_in_child = comm::recv(p);
    assert x_in_parent == x_in_child;

    let y_in_child = comm::recv(p);
    assert y_in_parent != y_in_child;
}
