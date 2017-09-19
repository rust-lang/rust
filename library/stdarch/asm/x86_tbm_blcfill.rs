extern crate stdsimd;

#[no_mangle]
pub fn blcfill_u32(x: u32) -> u32 {
    stdsimd::vendor::_blcfill_u32(x)
}

#[no_mangle]
pub fn blcfill_u64(x: u64) -> u64 {
    stdsimd::vendor::_blcfill_u64(x)
}
