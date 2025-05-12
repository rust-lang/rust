//@ check-pass

#![allow(unused)]
struct Mappable;

impl Mappable {
    pub fn map(&self) {}
}

fn main() {
    let m = Mappable {};
    m.map();
}
