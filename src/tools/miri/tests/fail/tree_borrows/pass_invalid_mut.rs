//@compile-flags: -Zmiri-tree-borrows
// This test is the TB counterpart to fail/stacked_borrows/pass_invalid_mut,
// but the SB version passes TB without error.
// An additional write access is inserted so that this test properly fails.

// Make sure that we cannot use a `&mut` whose parent got invalidated.
// fail/both_borrows/pass_invalid_shr is already checking a forbidden read,
// so the new thing that this tests is a forbidden write.
fn foo(nope: &mut i32) {
    *nope = 31; //~ ERROR: /write access through .* is forbidden/
}

fn main() {
    let x = &mut 42;
    let xraw = x as *mut _;
    let xref = unsafe { &mut *xraw };
    *xref = 18; // activate xref
    let _val = unsafe { *xraw }; // invalidate xref for writing
    foo(xref);
}
