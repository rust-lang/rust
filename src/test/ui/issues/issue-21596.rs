fn main() {
    let x = 8u8;
    let z: *const u8 = &x;
    println!("{}", z.to_string());  //~ ERROR E0599
}
