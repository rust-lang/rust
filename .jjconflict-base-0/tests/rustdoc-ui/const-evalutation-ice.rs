// Just check we don't get an ICE for `N`.

use std::cell::Cell;
use std::mem;

pub struct S {
    s: Cell<usize>,
}

pub const N: usize = 0 - (mem::size_of::<S>() != 400) as usize;
//~^ ERROR overflow
