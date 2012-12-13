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


extern mod std;

fn sub(parent: oldcomm::Chan<int>, id: int) {
    if id == 0 {
        oldcomm::send(parent, 0);
    } else {
        let p = oldcomm::Port();
        let ch = oldcomm::Chan(&p);
        let child = task::spawn(|| sub(ch, id - 1) );
        let y = oldcomm::recv(p);
        oldcomm::send(parent, y + 1);
    }
}

fn main() {
    let p = oldcomm::Port();
    let ch = oldcomm::Chan(&p);
    let child = task::spawn(|| sub(ch, 200) );
    let y = oldcomm::recv(p);
    debug!("transmission complete");
    log(debug, y);
    assert (y == 200);
}
