#[no_mangle]
pub extern "C" fn __floatdisf(x: i64) -> f32 {
    x as f32
}

#[no_mangle]
pub extern "C" fn __floatdidf(x: i64) -> f64 {
    x as f64
}
