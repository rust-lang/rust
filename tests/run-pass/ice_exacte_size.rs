#![deny(clippy)]

#[allow(dead_code)]
struct Foo;

impl Iterator for Foo {
    type Item = ();

    #[allow(unimplemented)]
    fn next(&mut self) -> Option<()> {
        let _ = self.len() == 0;
        unimplemented!()
    }
}

impl ExactSizeIterator for Foo {}

fn main() {}
