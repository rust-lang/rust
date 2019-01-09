// no-prefer-dynamic

// This would previously leak the Box<Trait> because we wouldn't
// schedule cleanups when auto borrowing trait objects.
// This program should be valgrind clean.

#![feature(box_syntax)]

static mut DROP_RAN: bool = false;

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { DROP_RAN = true; }
    }
}


trait Trait { fn dummy(&self) { } }
impl Trait for Foo {}

pub fn main() {
    {
        let _x: &Trait = &*(box Foo as Box<Trait>);
    }
    unsafe {
        assert!(DROP_RAN);
    }
}
