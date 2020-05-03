// compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(const_err)]

use std::cell::UnsafeCell;

// a test demonstrating what things we could allow with a smarter const qualification

// this is fine because is not possible to mutate through an immutable reference.
static FOO: &&mut u32 = &&mut 42;

// this is fine because accessing an immutable static `BAR` is equivalent to accessing `*&BAR`
// which puts the mutable reference behind an immutable one.
static BAR: &mut () = &mut ();

struct Foo<T>(T);

// this is fine for the same reason as `BAR`.
static BOO: &mut Foo<()> = &mut Foo(());

struct Meh {
    x: &'static UnsafeCell<i32>,
}

unsafe impl Sync for Meh {}

static MEH: Meh = Meh {
    x: &UnsafeCell::new(42),
};

// this is fine for the same reason as `BAR`.
static OH_YES: &mut i32 = &mut 42;

fn main() {
    unsafe {
        *MEH.x.get() = 99;
    }
    *OH_YES = 99; //~ ERROR cannot assign to `*OH_YES`, as `OH_YES` is an immutable static item
}
