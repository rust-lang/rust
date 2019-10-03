#![allow(unused, clippy::unused_self)]
struct Mappable {}

impl Mappable {
    pub fn map(&self) {}
}

fn main() {
    let m = Mappable {};
    m.map();
}
