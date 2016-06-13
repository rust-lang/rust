fn main() {
    let x: *const u8 = &1;
    let y: *const u8 = &2;
    if x < y { //~ ERROR: attempted to do math or a comparison on pointers into different allocations
        unreachable!()
    }
}
