fn main() {
    let v = core::ptr::null::<u8>();
    v as *const [u8]; //~ ERROR E0607
}
