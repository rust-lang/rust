//@ revisions: rpass1

// Regression test that `infer_outlives_predicates` can be
// used with incremental without an ICE.

struct Foo<'a, T> {
  x: &'a T
}

fn main() { }
