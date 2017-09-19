extern crate stdsimd;

#[no_mangle]
pub fn andn_u32(x: u32, y: u32) -> u32 {
    stdsimd::vendor::_andn_u32(x, y)
}

#[no_mangle]
pub fn andn_u64(x: u64, y: u64) -> u64 {
    stdsimd::vendor::_andn_u64(x, y)
}

