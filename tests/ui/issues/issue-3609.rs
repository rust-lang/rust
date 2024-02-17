//@ check-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_mut)]
use std::thread;
use std::sync::mpsc::Sender;

type RingBuffer = Vec<f64> ;
type SamplesFn = Box<dyn FnMut(&RingBuffer) + Send>;

enum Msg
{
    GetSamples(String, SamplesFn), // sample set name, callback which receives samples
}

fn foo(name: String, samples_chan: Sender<Msg>) {
    thread::spawn(move|| {
        let mut samples_chan = samples_chan;

        let callback: SamplesFn = Box::new(move |buffer| {
            for i in 0..buffer.len() {
                println!("{}: {}", i, buffer[i])
            }
        });

        samples_chan.send(Msg::GetSamples(name.clone(), callback));
    }).join();
}

pub fn main() {}
