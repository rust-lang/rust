fn main() {
    let x: u8 = 256;
    //~^ error: literal out of range for `u8`

    for _ in 0..256u8 {}
    //~^ error: range endpoint is out of range for `u8`
}
