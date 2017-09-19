extern crate stdsimd;

#[no_mangle]
pub fn umulx_u32(x: u32, y: u32) -> (u32, u32) {
    stdsimd::vendor::_mulx_u32(x, y)
}

#[no_mangle]
pub fn umulx_u64(x: u64, y: u64) -> (u64, u64) {
    stdsimd::vendor::_mulx_u64(x, y)
}
