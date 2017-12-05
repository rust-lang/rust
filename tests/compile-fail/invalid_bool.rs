fn main() {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) }; //~ ERROR: invalid boolean value read
    if b { unreachable!() } else { unreachable!() }
}
