// error-pattern:bounds check

fn main() {

    // This should cause a bounds-check failure, but may not if we do our
    // bounds checking by comparing the scaled index to the vector's
    // address-bounds, since we've scaled the index to wrap around to the
    // address of the 0th cell in the array (even though the index is
    // huge).

    let x = ~[1u,2u,3u];
    do vec::as_imm_buf(x) |p, _len| {
        let base = p as uint;                     // base = 0x1230 say
        let idx = base / sys::size_of::<uint>();  // idx  = 0x0246 say
        error!("ov1 base = 0x%x", base);
        error!("ov1 idx = 0x%x", idx);
        error!("ov1 sizeof::<uint>() = 0x%x", sys::size_of::<uint>());
        error!("ov1 idx * sizeof::<uint>() = 0x%x",
               idx * sys::size_of::<uint>());

        // This should fail.
        error!("ov1 0x%x",  x[idx]);
    }
}
