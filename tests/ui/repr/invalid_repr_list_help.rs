#![crate_type = "lib"]

#[repr(uwu)] //~ERROR: unrecognized representation hint
pub struct OwO;

#[repr(uwu = "a")] //~ERROR: unrecognized representation hint
pub struct OwO2(i32);

#[repr(uwu(4))] //~ERROR: unrecognized representation hint
pub struct OwO3 {
    x: i32,
}

#[repr(uwu, u8)] //~ERROR: unrecognized representation hint
pub enum OwO4 {
    UwU = 1,
}

#[repr(uwu)] //~ERROR: unrecognized representation hint
#[doc(owo)]  //~ERROR: unknown `doc` attribute
pub struct Owo5;
