// Just check if we don't get an ICE for the _S type.

use std::cell::Cell;
use std::mem;

pub struct S {
    s: Cell<usize>
}

pub type _S = [usize; 0 - (mem::size_of::<S>() != 4) as usize];
