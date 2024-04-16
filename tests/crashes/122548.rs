//@ known-bug: #122548
#![feature(const_mut_refs)]
#![feature(const_refs_to_static)]

use std::cell::UnsafeCell;

struct Meh {
    x: &'static UnsafeCell<i32>,
}

const MUH: Meh = Meh {
    x: &mut *(&READONLY as *const _ as *mut _),
};

static READONLY: i32 = 0;

pub fn main() {}
