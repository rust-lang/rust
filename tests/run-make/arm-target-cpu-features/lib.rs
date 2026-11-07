#![no_std]

#[no_mangle]
pub fn vadd_f32_q(x: &mut [f32; 4], y: &[f32; 4]) {
    for i in 0..4 {
        x[i] += y[i];
    }
}

#[no_mangle]
pub fn vadd_f64(x: f64, y: f64) -> f64 {
    x + y
}
