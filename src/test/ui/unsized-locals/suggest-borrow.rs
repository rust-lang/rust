fn main() {
    let x: [u8] = vec!(1, 2, 3)[..]; //~ ERROR E0277
    let x: &[u8] = vec!(1, 2, 3)[..]; //~ ERROR E0308
    let x: [u8] = &vec!(1, 2, 3)[..]; //~ ERROR E0308
    //~^ ERROR E0277
    let x: &[u8] = &vec!(1, 2, 3)[..];
}
