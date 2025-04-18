//@ check-pass

// Test for https://github.com/rust-lang/rust-clippy/issues/1336

struct Foo;

impl Iterator for Foo {
    type Item = ();

    fn next(&mut self) -> Option<()> {
        let _ = self.len() == 0;
        unimplemented!()
    }
}

impl ExactSizeIterator for Foo {}

fn main() {}
