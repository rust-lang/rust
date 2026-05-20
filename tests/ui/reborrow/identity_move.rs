//@ run-pass

#![feature(reborrow)]

use std::marker::Reborrow;

struct Thing<'a>(&'a mut usize);

impl<'a> Reborrow for Thing<'a> {}

impl Drop for Thing<'_> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut drops = 0;

    {
        let thing = Thing(&mut drops);
        let _moved: Thing<'_> = thing;
    }

    assert_eq!(drops, 1);
}
