//@ check-pass

trait Database: Restriction<Inner = u32> {}

trait Restriction {
    type Inner;
}

struct Test {}

impl Database for Test {}
impl Restriction for Test {
    type Inner = u32;
}

fn main() {
    let t = Test {};
    let x: &dyn Database<Inner = _> = &t;
}
