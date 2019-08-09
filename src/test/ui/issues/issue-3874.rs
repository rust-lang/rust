// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

enum PureCounter { PureCounterVariant(usize) }

fn each<F>(thing: PureCounter, blk: F) where F: FnOnce(&usize) {
    let PureCounter::PureCounterVariant(ref x) = thing;
    blk(x);
}

pub fn main() {}
