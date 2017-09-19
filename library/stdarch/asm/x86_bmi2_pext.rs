extern crate stdsimd;

#[no_mangle]
pub fn pext_u32(x: u32, mask: u32) -> u32 {
    stdsimd::vendor::_pext_u32(x, mask)
}

#[no_mangle]
pub fn pext_u64(x: u64, mask: u64) -> u64 {
    stdsimd::vendor::_pext_u64(x, mask)
}
