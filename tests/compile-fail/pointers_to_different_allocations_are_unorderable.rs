fn main() {
    let x: *const u8 = &1;
    let y: *const u8 = &2;
    if x < y { //~ ERROR attempted to do invalid arithmetic on pointers
        unreachable!()
    }
}
