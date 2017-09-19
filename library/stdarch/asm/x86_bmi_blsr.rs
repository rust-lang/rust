extern crate stdsimd;

#[no_mangle]
pub fn blsr_u32(x: u32) -> u32 {
    stdsimd::vendor::_blsr_u32(x)
}

#[no_mangle]
pub fn blsr_u64(x: u64) -> u64 {
    stdsimd::vendor::_blsr_u64(x)
}
