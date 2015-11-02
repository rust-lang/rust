// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Interfaces to the operating system provided random number
//! generators.

pub use sys::rand::OsRng;

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use sync::mpsc::channel;
    use rand::Rng;
    use super::OsRng;
    use thread;

    #[test]
    fn test_os_rng() {
        let mut r = OsRng::new().unwrap();

        r.next_u32();
        r.next_u64();

        let mut v = [0; 1000];
        r.fill_bytes(&mut v);
    }

    #[test]
    fn test_os_rng_tasks() {

        let mut txs = vec!();
        for _ in 0..20 {
            let (tx, rx) = channel();
            txs.push(tx);

            thread::spawn(move|| {
                // wait until all the threads are ready to go.
                rx.recv().unwrap();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OsRng::new().unwrap();
                thread::yield_now();
                let mut v = [0; 1000];

                for _ in 0..100 {
                    r.next_u32();
                    thread::yield_now();
                    r.next_u64();
                    thread::yield_now();
                    r.fill_bytes(&mut v);
                    thread::yield_now();
                }
            });
        }

        // start all the threads
        for tx in &txs {
            tx.send(()).unwrap();
        }
    }
}
