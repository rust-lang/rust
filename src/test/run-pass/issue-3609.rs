extern mod extra;

use std::comm::Chan;
use std::task;

type RingBuffer = ~[f64];
type SamplesFn = proc(samples: &RingBuffer);

enum Msg
{
    GetSamples(~str, SamplesFn), // sample set name, callback which receives samples
}

fn foo(name: ~str, samples_chan: Chan<Msg>) {
    do task::spawn
    {
        let mut samples_chan = samples_chan;
        let callback: SamplesFn = proc(buffer) {
            for i in range(0u, buffer.len()) {
                error!("{}: {}", i, buffer[i])
            }
        };
        samples_chan.send(GetSamples(name.clone(), callback));
    };
}

pub fn main() {}
