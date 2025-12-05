//@ check-pass
//
// rust-lang/rust#74933: Lifetime error when indexing with borrowed index

use std::ops::{Index, IndexMut};

struct S(V);
struct K<'a>(&'a ());
struct V;

impl<'a> Index<&'a K<'a>> for S {
    type Output = V;

    fn index(&self, _: &'a K<'a>) -> &V {
        &self.0
    }
}

impl<'a> IndexMut<&'a K<'a>> for S {
    fn index_mut(&mut self, _: &'a K<'a>) -> &mut V {
        &mut self.0
    }
}

impl V {
    fn foo(&mut self) {}
}

fn test(s: &mut S, k: &K<'_>) {
    s[k] = V;
    s[k].foo();
}

fn main() {
    let mut s = S(V);
    let k = K(&());
    test(&mut s, &k);
}
