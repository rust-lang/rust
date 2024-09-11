fn main() {
    let x = &[0i32; 2];
    let x = x.as_ptr().wrapping_add(1);
    // If the `!0` is interpreted as `isize`, it is just `-1` and hence harmless.
    // However, this is unsigned arithmetic, so really this is `usize::MAX` and hence UB.
    unsafe { x.byte_add(!0).read() }; //~ERROR: does not fit in an `isize`
}
