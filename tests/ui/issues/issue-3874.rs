//@ build-pass
#![allow(dead_code)]

enum PureCounter { PureCounterVariant(usize) }

fn each<F>(thing: PureCounter, blk: F) where F: FnOnce(&usize) {
    let PureCounter::PureCounterVariant(ref x) = thing;
    blk(x);
}

pub fn main() {}
