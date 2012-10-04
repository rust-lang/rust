// xfail-test
// error-pattern:index out of bounds

fn main() {
    let x = ~[1u,2u,3u];

    // This should cause a bounds-check failure, but may not if we do our
    // bounds checking by comparing a scaled index value to the vector's
    // length (in bytes), because the scaling of the index will cause it to
    // wrap around to a small number.

    let idx = uint::max_value & !(uint::max_value >> 1u);
    error!("ov2 idx = 0x%x", idx);

    // This should fail.
    error!("ov2 0x%x",  x[idx]);
}
