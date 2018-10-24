// Validation makes this fail in the wrong place
// compile-flags: -Zmiri-disable-validation

fn main() {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    let _x = b == true; //~ ERROR invalid boolean value read
}
