// pretty-expanded FIXME #23616

#[repr(simd)] //~ ERROR SIMD types are experimental
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32
}

pub fn main() {}
