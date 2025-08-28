//@ run-pass
#![allow(dead_code)]

// Passing enums by value

#[repr(C)]
pub enum PoorQualityAnyEnum {
   None = 0,
   Int = 1,
   Long = 2,
   Float = 17,
   Double = 18,
}

mod bindgen {
    use super::PoorQualityAnyEnum;

    extern "C" {
        pub fn printf(v: PoorQualityAnyEnum);
    }
}

pub fn main() {}
