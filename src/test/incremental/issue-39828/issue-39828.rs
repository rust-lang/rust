// Regression test for #39828. If you make use of a module that
// consists only of generics, no code is generated, just a dummy
// module. The reduced graph consists of a single node (for that
// module) with no inputs. Since we only serialize edges, when we
// reload, we would consider that node dirty since it is not recreated
// (it is not the target of any edges).

// revisions:rpass1 rpass2
// aux-build:generic.rs

extern crate generic;
fn main() { }
