// compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(const_err)]

use std::cell::UnsafeCell;

// a test demonstrating what things we could allow with a smarter const qualification

static FOO: &&mut u32 = &&mut 42;

static BAR: &mut () = &mut ();

struct Foo<T>(T);

static BOO: &mut Foo<()> = &mut Foo(());

struct Meh {
    x: &'static UnsafeCell<i32>,
}

unsafe impl Sync for Meh {}

static MEH: Meh = Meh {
    x: &UnsafeCell::new(42),
};

static OH_YES: &mut i32 = &mut 42;

fn main() {
    unsafe {
        *MEH.x.get() = 99; //~ WARN skipping const checks
        //~^ WARN skipping const checks
    }
    *OH_YES = 99; //~ ERROR cannot assign to `*OH_YES`, as `OH_YES` is an immutable static item
    //~^ WARN skipping const checks
}
