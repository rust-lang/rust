//@ build-fail
//
// When `large_assignments` is escalated to a hard error inside a generic function, the error must
// be attributed to the instantiation that triggered it, producing a "while instantiating" note.
// This exercises the post-monomorphization error bubbling out of the move check.
#![feature(large_assignments)]
#![move_size_limit = "1000"]
#![deny(large_assignments)]
#![allow(unused)]

struct Data([u8; 9999]);

fn main() {
    instantiate::<u8>();
}

fn instantiate<T>() {
    let data = Data([100; 9999]); //~ ERROR large_assignments
}
