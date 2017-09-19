extern crate stdsimd;

#[no_mangle]
pub fn tzmsk_u32(x: u32) -> u32 {
    stdsimd::vendor::_tzmsk_u32(x)
}

#[no_mangle]
pub fn tzmsk_u64(x: u64) -> u64 {
    stdsimd::vendor::_tzmsk_u64(x)
}
