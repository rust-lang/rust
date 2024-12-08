#![allow(unused, clippy::needless_lifetimes, clippy::needless_pass_by_ref_mut)]
#![warn(clippy::mut_from_ref)]

struct Foo;

impl Foo {
    fn this_wont_hurt_a_bit(&self) -> &mut Foo {
        //~^ ERROR: mutable borrow from immutable input(s)
        unsafe { unimplemented!() }
    }
}

trait Ouch {
    fn ouch(x: &Foo) -> &mut Foo;
    //~^ ERROR: mutable borrow from immutable input(s)
}

impl Ouch for Foo {
    fn ouch(x: &Foo) -> &mut Foo {
        unsafe { unimplemented!() }
    }
}

fn fail(x: &u32) -> &mut u16 {
    //~^ ERROR: mutable borrow from immutable input(s)
    unsafe { unimplemented!() }
}

fn fail_lifetime<'a>(x: &'a u32, y: &mut u32) -> &'a mut u32 {
    //~^ ERROR: mutable borrow from immutable input(s)
    unsafe { unimplemented!() }
}

fn fail_double<'a, 'b>(x: &'a u32, y: &'a u32, z: &'b mut u32) -> &'a mut u32 {
    //~^ ERROR: mutable borrow from immutable input(s)
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
    //~^ ERROR: mutable borrow from immutable input(s)
    unimplemented!()
}

fn without_unsafe(x: &u32) -> &mut u32 {
    unimplemented!()
}

fn main() {
    //TODO
}
