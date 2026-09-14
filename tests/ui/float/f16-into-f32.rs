//@ build-pass
#![feature(f16, f32_from_f16)]
#![allow(unused)]

// Check that float conversions work, specifically a {float} literal that normally would fall back
// to an f64 but due to the Into bound here falls back to f32. Also test that the lint is emitted in
// the correct location, and can be `expect`ed or `allow`ed.
fn convert(x: impl Into<f32>) -> f32 {
    x.into()
}

pub fn main() {
    let _ = convert(1.0f32);
    let _ = convert(1.0f16);
    #[expect(float_literal_f32_fallback)]
    let _ = convert(1.0);
}
