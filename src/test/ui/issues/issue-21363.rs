// check-pass
// pretty-expanded FIXME #23616

#![feature(no_prelude)]
#![no_prelude]

trait Iterator {
    type Item;
    fn dummy(&self) { }
}

impl<'a, T> Iterator for &'a mut (dyn Iterator<Item=T> + 'a) {
    type Item = T;
}

fn main() {}
