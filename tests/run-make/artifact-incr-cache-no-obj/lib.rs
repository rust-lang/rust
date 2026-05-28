#![crate_name = "foo"]

#[inline(never)]
pub fn add(a: u32, b: u32) -> u32 {
    a + b
}
