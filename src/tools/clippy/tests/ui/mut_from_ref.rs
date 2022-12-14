#![allow(unused, clippy::needless_lifetimes)]
#![warn(clippy::mut_from_ref)]

struct Foo;

impl Foo {
    fn this_wont_hurt_a_bit(&self) -> &mut Foo {
        unsafe { unimplemented!() }
    }
}

trait Ouch {
    fn ouch(x: &Foo) -> &mut Foo;
}

impl Ouch for Foo {
    fn ouch(x: &Foo) -> &mut Foo {
        unsafe { unimplemented!() }
    }
}

fn fail(x: &u32) -> &mut u16 {
    unsafe { unimplemented!() }
}

fn fail_lifetime<'a>(x: &'a u32, y: &mut u32) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

fn fail_double<'a, 'b>(x: &'a u32, y: &'a u32, z: &'b mut u32) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

// this is OK, because the result borrows y
fn works<'a>(x: &u32, y: &'a mut u32) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

// this is also OK, because the result could borrow y
fn also_works<'a>(x: &'a u32, y: &'a mut u32) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

unsafe fn also_broken(x: &u32) -> &mut u32 {
    unimplemented!()
}

fn without_unsafe(x: &u32) -> &mut u32 {
    unimplemented!()
}

fn main() {
    //TODO
}
