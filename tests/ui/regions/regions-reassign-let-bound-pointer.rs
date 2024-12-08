//@ run-pass
#![allow(unused_assignments)]
#![allow(unused_variables)]
// Check that the type checker permits us to reassign `z` which
// started out with a longer lifetime and was reassigned to a shorter
// one (it should infer to be the intersection).


fn foo(x: &isize) {
    let a = 1;
    let mut z = x;
    z = &a;
}

pub fn main() {
    foo(&1);
}
