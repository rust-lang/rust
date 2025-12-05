#![crate_type="lib"]

enum E { E0 = 0, E1 = 1 }
const E0_U8: u8 = E::E0 as u8;
const E1_U8: u8 = E::E1 as u8;

pub fn go<T>() {
    match 0 {
        E0_U8 => (),
        E1_U8 => (),
        _ => (),
    }
}
