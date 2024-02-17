//@ run-pass
//@ aux-build:trait_superkinds_in_metadata.rs

// Tests "capabilities" granted by traits with super-builtin-kinds,
// even when using them cross-crate.


extern crate trait_superkinds_in_metadata;

use std::sync::mpsc::{channel, Sender, Receiver};
use trait_superkinds_in_metadata::{RequiresRequiresShareAndSend, RequiresShare};

#[derive(PartialEq, Debug)]
struct X<T>(T);

impl <T: Sync> RequiresShare for X<T> { }
impl <T: Sync+Send> RequiresRequiresShareAndSend for X<T> { }

fn foo<T: RequiresRequiresShareAndSend + 'static>(val: T, chan: Sender<T>) {
    chan.send(val).unwrap();
}

pub fn main() {
    let (tx, rx): (Sender<X<isize>>, Receiver<X<isize>>) = channel();
    foo(X(31337), tx);
    assert_eq!(rx.recv().unwrap(), X(31337));
}
