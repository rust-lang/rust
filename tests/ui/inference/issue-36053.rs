//@ run-pass
// Regression test for #36053. ICE was caused due to obligations being
// added to a special, dedicated fulfillment cx during a
// probe. Problem seems to be related to the particular definition of
// `FusedIterator` in std but I was not able to isolate that into an
// external crate.

use std::iter::FusedIterator;

struct Thing<'a>(#[allow(dead_code)] &'a str);
impl<'a> Iterator for Thing<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        None
    }
}

impl<'a> FusedIterator for Thing<'a> {}

fn main() {
    Thing("test").fuse().filter(|_| true).count();
}
