//@ known-bug: rust-lang/rust#127266
#![feature(const_mut_refs)]
#![feature(const_refs_to_static)]

struct Meh {
    x: &'static dyn UnsafeCell,
}

const MUH: Meh = Meh {
    x: &mut *(READONLY as *mut _),
};

static READONLY: i32 = 0;

trait UnsafeCell<'a> {}

pub fn main() {}
