//@ known-bug: #130521

#![feature(dyn_compatible_for_dispatch)]
struct Vtable(dyn Cap<'static>);

trait Cap<'a> {}

union Transmute {
    t: u128,
    u: &'static Vtable,
}

const G: &Copy = unsafe { Transmute { t: 1 }.u };
