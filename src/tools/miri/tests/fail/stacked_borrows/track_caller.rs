// This is a copy of illegal_read1.rs, but with #[track_caller] on the test.
// This test only checks that our diagnostics do not display the contents of callee.

#[rustfmt::skip] // rustfmt bug: https://github.com/rust-lang/rustfmt/issues/5391
fn main() {
    let mut x = 15;
    let xraw = &mut x as *mut _;
    let xref = unsafe { &mut *xraw }; // derived from raw, so using raw is still ok...
    callee(xraw);
    let _val = *xref; // ...but any use of raw will invalidate our ref.
    //~^ ERROR: /read access .* tag does not exist in the borrow stack/
}

#[track_caller]
fn callee(xraw: *mut i32) {
    let _val = unsafe { *xraw };
}
