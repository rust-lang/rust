// compile-flags: -Z borrowck=mir

// run-pass

use std::io::Result;

struct Foo {}

pub trait FakeRead {
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize>;
}

impl FakeRead for Foo {
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize> {
        Ok(4)
    }
}

fn main() {
    let mut a = Foo {};
    let mut v = Vec::new();
    a.read_to_end(&mut v);
}
