// A callee may not read the destination of our `&mut` without us noticing.
// This code got carefully checked to not introduce any reborrows
// that are not explicit in the source. Let's hope the compiler does not break this later!

use std::mem;

union HiddenRef {
    // We avoid retagging at this type, and we only read, so shared vs mutable does not matter.
    r: &'static i32,
}

fn main() {
    let mut x: i32 = 15;
    let xref1 = &mut x;
    let xref1_sneaky: HiddenRef = unsafe { mem::transmute_copy(&xref1) };
    // Derived from `xref1`, so using raw value is still ok, ...
    let xref2 = &mut *xref1;
    callee(xref1_sneaky);
    // ... though any use of it will invalidate our ref.
    let _val = *xref2;
    //~^ ERROR: /read access .* tag does not exist in the borrow stack/
}

fn callee(xref1: HiddenRef) {
    // Doing the deref and the transmute (through the union) in the same place expression
    // should avoid retagging.
    let _val = unsafe { *xref1.r };
}
