#![deny(unclear_local_imports)]

mod localmod {
    pub struct S;
    pub struct T;
}

// Not a local import, so no lint.
use std::cell::Cell;

// Implicitly local import, gets lint.
use localmod::S; //~ERROR: unclear

// Explicitly local import, no lint.
use self::localmod::T;

fn main() {}
