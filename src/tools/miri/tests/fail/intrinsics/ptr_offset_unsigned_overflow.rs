fn main() {
    let x = &[0i32; 2];
    let x = x.as_ptr().wrapping_add(1);
    // If `usize::MAX` is interpreted as `isize`, it is just `-1` and hence harmless.
    let _ = unsafe { x.byte_offset(usize::MAX as isize) };
    // However, `byte_add` uses unsigned arithmetic, so really this is `usize::MAX` and hence UB.
    let _ = unsafe { x.byte_add(usize::MAX) }; //~ERROR: does not fit in an `isize`
}
