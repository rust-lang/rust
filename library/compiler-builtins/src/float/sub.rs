use float::Float;

intrinsics! {
    pub extern "C" fn __subsf3(a: f32, b: f32) -> f32 {
        a + f32::from_repr(b.repr() ^ f32::sign_mask())
    }

    pub extern "C" fn __subdf3(a: f64, b: f64) -> f64 {
        a + f64::from_repr(b.repr() ^ f64::sign_mask())
    }
}
