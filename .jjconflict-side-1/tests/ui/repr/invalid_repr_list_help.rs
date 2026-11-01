#![deny(invalid_doc_attributes)]
#![crate_type = "lib"]

#[repr(uwu)] //~ERROR: malformed `repr` attribute input
pub struct OwO;

#[repr(uwu = "a")] //~ERROR: malformed `repr` attribute input
pub struct OwO2(i32);

#[repr(uwu(4))] //~ERROR: malformed `repr` attribute input
pub struct OwO3 {
    x: i32,
}

#[repr(uwu, u8)] //~ERROR: malformed `repr` attribute input
pub enum OwO4 {
    UwU = 1,
}

#[repr(uwu)] //~ERROR: malformed `repr` attribute input
#[doc(owo)]  //~ERROR: unknown `doc` attribute
pub struct Owo5;
