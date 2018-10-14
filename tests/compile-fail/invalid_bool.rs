fn main() {
    let _b = unsafe { std::mem::transmute::<u8, bool>(2) }; //~ ERROR encountered 2, but expected something in the range 0..=1
}
