// Variant of #117151 when the overflow comes entirely from subtype predicates.

#![allow(unreachable_code)]

use std::ptr;

fn main() {
    // Give x and y completely unconstrained types. Using a function call
    // or `as` cast would create a well-formed predicate.
    let x = return;
    let y = return;
    let mut w = (x, y);
    //~^ ERROR overflow assigning `_` to `*const _`
    // Avoid creating lifetimes, `Sized` bounds or function calls.
    let a = (ptr::addr_of!(y), ptr::addr_of!(x));
    w = a;
}
