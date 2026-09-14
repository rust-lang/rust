pub fn sqrtf(x: f32) -> f32 {
    core::arch::wasm32::f32_sqrt(x)
}

pub fn sqrt(x: f64) -> f64 {
    core::arch::wasm32::f64_sqrt(x)
}
