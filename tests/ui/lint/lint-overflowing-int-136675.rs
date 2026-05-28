fn main() {
    if let -129 = 0i8 {} //~ ERROR literal out of range for `i8`
    let x: i8 = -129; //~ ERROR literal out of range for `i8`
}
