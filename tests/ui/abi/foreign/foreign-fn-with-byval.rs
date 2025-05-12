//@ run-pass
#![allow(improper_ctypes, improper_ctypes_definitions)]

#[derive(Copy, Clone)]
pub struct S {
    x: u64,
    y: u64,
    z: u64,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    pub fn get_x(x: S) -> u64;
    pub fn get_y(x: S) -> u64;
    pub fn get_z(x: S) -> u64;
}

#[inline(never)]
fn indirect_call(func: unsafe extern "C" fn(s: S) -> u64, s: S) -> u64 {
    unsafe { func(s) }
}

fn main() {
    let s = S { x: 1, y: 2, z: 3 };
    assert_eq!(s.x, indirect_call(get_x, s));
    assert_eq!(s.y, indirect_call(get_y, s));
    assert_eq!(s.z, indirect_call(get_z, s));
}
