// Validation makes this fail in the wrong place
// compile-flags: -Zmiri-disable-validation

fn main() {
    assert!(std::char::from_u32(-1_i32 as u32).is_none());
    let c = unsafe { std::mem::transmute::<i32, char>(-1) };
    let _x = c == 'x'; //~ ERROR tried to interpret an invalid 32-bit value as a char
}
