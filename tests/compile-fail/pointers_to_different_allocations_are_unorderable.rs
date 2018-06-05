fn main() {
    let x: *const u8 = &1;
    let y: *const u8 = &2;
    if x < y { //~ ERROR constant evaluation error [E0080]
    //~^ NOTE attempted to do invalid arithmetic on pointers
        unreachable!()
    }
}
