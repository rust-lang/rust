// Regression test for <https://github.com/rust-lang/rust/issues/5500>,
// check that you can take a reference to the never type.
//
//@ edition:2015..2021
//@ check-pass

struct TrieMapIterator<'a> {
    node: &'a usize
}

fn main() {
    let a = 5;
    let _iter = TrieMapIterator{node: &a};
    _iter.node = &panic!()
}
