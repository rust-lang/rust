//@compile-flags: -Zmiri-tree-borrows
// This test is the TB counterpart to fail/stacked_borrows/return_invalid_mut,
// but the SB version passes TB without error.
// An additional write access is inserted so that this test properly fails.

// Make sure that we cannot use a returned `&mut` that got already invalidated.
// fail/both_borrows/return_invalid_shr is already testing that we cannot return
// a reference invalidated for reading, so the new thing that we test here
// is the case where the return value cannot be used for writing.
fn foo(x: &mut (i32, i32)) -> &mut i32 {
    let xraw = x as *mut (i32, i32);
    let ret = unsafe { &mut (*xraw).1 };
    *ret = *ret; // activate
    let _val = unsafe { *xraw }; // invalidate xref for writing
    ret
}

fn main() {
    let arg = &mut (1, 2);
    let ret = foo(arg);
    *ret = 3; //~ ERROR: /write access through .* is forbidden/
}
