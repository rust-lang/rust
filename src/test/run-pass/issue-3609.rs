extern mod extra;

use std::comm::Chan;
use std::task;

type RingBuffer = ~[float];
type SamplesFn = ~fn(samples: &RingBuffer);

enum Msg
{
    GetSamples(~str, SamplesFn), // sample set name, callback which receives samples
}

fn foo(name: ~str, samples_chan: Chan<Msg>) {
    do task::spawn
    {
        let callback: SamplesFn =
            |buffer|
            {
                foreach i in range(0u, buffer.len()) {
                    error!("%?: %f", i, buffer[i])
                }
            };
        samples_chan.send(GetSamples(name.clone(), callback));
    };
}

pub fn main() {}
