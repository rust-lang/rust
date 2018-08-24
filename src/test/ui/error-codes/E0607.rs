fn main() {
    let v = 0 as *const u8;
    v as *const [u8]; //~ ERROR E0607
}
