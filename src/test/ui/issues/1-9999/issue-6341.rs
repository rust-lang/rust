// check-pass
// pretty-expanded FIXME #23616

#[derive(PartialEq)]
struct A { x: usize }

impl Drop for A {
    fn drop(&mut self) {}
}

pub fn main() {}
