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

use std::sync::mpsc::channel;
use std::env;
use std::thread;

// This is a simple bench that creates M pairs of tasks. These
// tasks ping-pong back and forth over a pair of streams. This is a
// canonical message-passing benchmark as it heavily strains message
// passing and almost nothing else.

fn ping_pong_bench(n: usize, m: usize) {

    // Create pairs of tasks that pingpong back and forth.
    fn run_pair(n: usize) {
        // Create a channel: A->B
        let (atx, arx) = channel();
        // Create a channel: B->A
        let (btx, brx) = channel();

        let guard_a = thread::spawn(move|| {
            let (tx, rx) = (atx, brx);
            for _ in 0..n {
                tx.send(()).unwrap();
                rx.recv().unwrap();
            }
        });

        let guard_b = thread::spawn(move|| {
            let (tx, rx) = (btx, arx);
            for _ in 0..n {
                rx.recv().unwrap();
                tx.send(()).unwrap();
            }
        });

        guard_a.join().ok();
        guard_b.join().ok();
    }

    for _ in 0..m {
        run_pair(n)
    }
}



fn main() {
    let mut args = env::args();
    let (n, m) = if args.len() == 3 {
        let n = args.nth(1).unwrap().parse::<usize>().unwrap();
        let m = args.next().unwrap().parse::<usize>().unwrap();
        (n, m)
    } else {
        (10000, 4)
    };

    ping_pong_bench(n, m);

}
