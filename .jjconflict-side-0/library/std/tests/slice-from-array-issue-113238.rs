// This intends to use the unsizing coercion from array to slice, but it only
// works if we resolve `<&[u8]>::from` as the reflexive `From<T> for T`. In
// #113238, we found that gimli had added its own `From<EndianSlice> for &[u8]`
// that affected all `std/backtrace` users.
#[test]
fn slice_from_array() {
    let _ = <&[u8]>::from(&[]);
}
