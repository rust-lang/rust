#![allow(
    unused,
    clippy::needless_lifetimes,
    clippy::needless_pass_by_ref_mut,
    clippy::redundant_allocation,
    clippy::boxed_local
)]
#![warn(clippy::mut_from_ref)]

struct Foo;

impl Foo {
    fn this_wont_hurt_a_bit(&self) -> &mut Foo {
        //~^ mut_from_ref

        unsafe { unimplemented!() }
    }
}

trait Ouch {
    fn ouch(x: &Foo) -> &mut Foo;
    //~^ mut_from_ref
}

impl Ouch for Foo {
    fn ouch(x: &Foo) -> &mut Foo {
        unsafe { unimplemented!() }
    }
}

fn fail(x: &u32) -> &mut u16 {
    //~^ mut_from_ref

    unsafe { unimplemented!() }
}

fn fail_lifetime<'a>(x: &'a u32, y: &mut u32) -> &'a mut u32 {
    //~^ mut_from_ref

    unsafe { unimplemented!() }
}

fn fail_double<'a, 'b>(x: &'a u32, y: &'a u32, z: &'b mut u32) -> &'a mut u32 {
    //~^ mut_from_ref

    unsafe { unimplemented!() }
}

fn fail_tuples<'a>(x: (&'a u32, &'a u32)) -> &'a mut u32 {
    //~^ mut_from_ref

    unsafe { unimplemented!() }
}

fn fail_box<'a>(x: Box<&'a u32>) -> &'a mut u32 {
    //~^ mut_from_ref

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

fn works_tuples<'a>(x: (&'a u32, &'a mut u32)) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

fn works_box<'a>(x: &'a u32, y: Box<&'a mut u32>) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

struct RefMut<'a>(&'a mut u32);

fn works_parameter<'a>(x: &'a u32, y: RefMut<'a>) -> &'a mut u32 {
    unsafe { unimplemented!() }
}

unsafe fn also_broken(x: &u32) -> &mut u32 {
    //~^ mut_from_ref

    unimplemented!()
}

fn without_unsafe(x: &u32) -> &mut u32 {
    unimplemented!()
}

fn main() {
    //TODO
}
