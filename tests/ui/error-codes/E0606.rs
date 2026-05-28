fn main() {
    let x = &(&0u8 as u8); //~ ERROR E0606
    x as u8; //~ ERROR casting `&u8` as `u8` is invalid [E0606]
}
