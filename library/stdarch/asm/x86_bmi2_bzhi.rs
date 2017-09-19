extern crate stdsimd;

#[no_mangle]
pub fn bzhi_u32(x: u32, mask: u32) -> u32 {
    stdsimd::vendor::_bzhi_u32(x, mask)
}

#[no_mangle]
pub fn bzhi_u64(x: u64, mask: u64) -> u64 {
    stdsimd::vendor::_bzhi_u64(x, mask)
}
