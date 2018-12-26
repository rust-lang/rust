// run-pass
#![allow(dead_code)]
// Regression test for #20582. This test caused an ICE related to
// inconsistent region erasure in codegen.

// pretty-expanded FIXME #23616

struct Foo<'a> {
    buf: &'a[u8]
}

impl<'a> Iterator for Foo<'a> {
    type Item = &'a[u8];

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        Some(self.buf)
    }
}

fn main() {
}
