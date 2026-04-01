#[repr(simd)] //~ ERROR SIMD types are experimental
struct RGBA {
    rgba: [f32; 4],
}

pub fn main() {}
