//@ known-bug: rust-lang/rust#126666
#![feature(const_mut_refs)]
#![feature(const_refs_to_static)]
#![feature(object_safe_for_dispatch)]

struct Meh {
    x: &'static dyn UnsafeCell,
}

const MUH: Meh = Meh {
    x: &mut *(&READONLY as *const _ as *mut _),
};

static READONLY: i32 = 0;

trait UnsafeCell<'a> {}

pub fn main() {}
