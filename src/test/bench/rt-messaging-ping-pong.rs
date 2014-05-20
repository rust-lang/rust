// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::os;
use std::uint;

// This is a simple bench that creates M pairs of tasks. These
// tasks ping-pong back and forth over a pair of streams. This is a
// cannonical message-passing benchmark as it heavily strains message
// passing and almost nothing else.

fn ping_pong_bench(n: uint, m: uint) {

    // Create pairs of tasks that pingpong back and forth.
    fn run_pair(n: uint) {
        // Create a stream A->B
        let (atx, arx) = channel::<()>();
        // Create a stream B->A
        let (btx, brx) = channel::<()>();

        spawn(proc() {
            let (tx, rx) = (atx, brx);
            for _ in range(0, n) {
                tx.send(());
                rx.recv();
            }
        });

        spawn(proc() {
            let (tx, rx) = (btx, arx);
            for _ in range(0, n) {
                rx.recv();
                tx.send(());
            }
        });
    }

    for _ in range(0, m) {
        run_pair(n)
    }
}



fn main() {

    let args = os::args();
    let args = args.as_slice();
    let n = if args.len() == 3 {
        from_str::<uint>(args[1].as_slice()).unwrap()
    } else {
        10000
    };

    let m = if args.len() == 3 {
        from_str::<uint>(args[2].as_slice()).unwrap()
    } else {
        4
    };

    ping_pong_bench(n, m);

}
