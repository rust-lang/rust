fn main() {
    let x = 0u8;
    x as Vec<u8>; //~ ERROR E0605

    let v = std::ptr::null::<u8>();
    v as &u8; //~ ERROR E0605
}
