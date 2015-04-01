// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(std_misc)]

use std::thread;
use std::sync::mpsc::Sender;
use std::thunk::Invoke;

type RingBuffer = Vec<f64> ;
type SamplesFn = Box<FnMut(&RingBuffer) + Send>;

enum Msg
{
    GetSamples(String, SamplesFn), // sample set name, callback which receives samples
}

fn foo(name: String, samples_chan: Sender<Msg>) {
    let _t = thread::scoped(move|| {
        let mut samples_chan = samples_chan;

        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        let callback: SamplesFn = Box::new(move |buffer| {
            for i in 0..buffer.len() {
                println!("{}: {}", i, buffer[i])
            }
        });

        samples_chan.send(Msg::GetSamples(name.clone(), callback));
    });
}

pub fn main() {}
