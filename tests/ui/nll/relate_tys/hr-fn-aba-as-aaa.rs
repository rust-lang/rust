// Test that the NLL `relate_tys` code correctly deduces that a
// function returning always its first argument can be upcast to one
// that returns either first or second argument.
//
//@ check-pass
//@ compile-flags:-Zno-leak-check

#![allow(dropping_copy_types)]

fn make_it() -> for<'a, 'b> fn(&'a u32, &'b u32) -> &'a u32 {
    panic!()
}

fn main() {
    let a: for<'a> fn(&'a u32, &'a u32) -> &'a u32 = make_it();
    drop(a);
}
