// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

type RingBuffer = Vec<f64> ;
type SamplesFn = proc(samples: &RingBuffer):Send;

enum Msg
{
    GetSamples(~str, SamplesFn), // sample set name, callback which receives samples
}

fn foo(name: ~str, samples_chan: Sender<Msg>) {
    task::spawn(proc() {
        let mut samples_chan = samples_chan;
        let callback: SamplesFn = proc(buffer) {
            for i in range(0u, buffer.len()) {
                println!("{}: {}", i, *buffer.get(i))
            }
        };
        samples_chan.send(GetSamples(name.clone(), callback));
    });
}

pub fn main() {}
