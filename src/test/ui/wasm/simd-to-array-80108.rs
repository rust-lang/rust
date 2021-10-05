// only-wasm32
// build-pass
#![feature(repr_simd)]

// Regression test for #80108


#[repr(simd)]
pub struct Vector([i32; 4]);

impl Vector {
    pub const fn to_array(self) -> [i32; 4] {
        self.0
    }
}
