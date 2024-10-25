// Regression test for #130521.

#![feature(dyn_compatible_for_dispatch)]

trait Cap<'a> {}
struct Vtable(dyn Cap);
//~^ ERROR missing lifetime specifier

union Transmute {
    t: u64,
    u: &'static Vtable,
}

const G: &dyn Copy = unsafe { Transmute { t: 1 }.u };
//~^ ERROR the trait `Copy` cannot be made into an object
//~| ERROR evaluation of constant value failed

fn main() {}
