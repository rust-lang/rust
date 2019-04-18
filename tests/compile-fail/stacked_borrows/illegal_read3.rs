// A callee may not read the destination of our `&mut` without us noticing.
// Thise code got carefully checked to not introduce any reborrows
// that are not explicit in the source. Let's hope the compiler does not break this later!

#![feature(untagged_unions)]

use std::mem;

fn main() {
    let mut x: i32 = 15;
    let xref1 = &mut x;
    let xref1_sneaky: usize = unsafe { mem::transmute_copy(&xref1) };
    // Derived from `xref1`, so using raw value is still ok, ...
    let xref2 = &mut *xref1;
    callee(xref1_sneaky);
    // ... though any use of it will invalidate our ref.
    let _val = *xref2;
    //~^ ERROR: borrow stack
}

fn callee(xref1: usize) {
    // Transmuting through a union to avoid retagging.
    union UsizeToRef {
        from: usize,
        to: &'static mut i32,
    }
    let xref1 = UsizeToRef { from: xref1 };
    // Doing the deref and the transmute (through the union) in the same place expression
    // should avoid retagging.
    let _val = unsafe { *xref1.to };
}
