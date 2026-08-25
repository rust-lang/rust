//@ run-pass
#[allow(improper_ctypes_definitions)]
pub extern "C" fn tuple2() -> (u16, u8) {
    (1, 2)
}

#[allow(improper_ctypes_definitions)]
pub extern "C" fn tuple3() -> (u8, u8, u8) {
    (1, 2, 3)
}

pub fn test2() -> u8 {
    tuple2().1
}

pub fn test3() -> u8 {
    tuple3().2
}

fn main() {
    assert_eq!(test2(), 2);
    assert_eq!(test3(), 3);
}
