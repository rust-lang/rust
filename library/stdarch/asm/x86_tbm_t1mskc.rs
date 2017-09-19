extern crate stdsimd;

#[no_mangle]
pub fn t1mskc_u32(x: u32) -> u32 {
    stdsimd::vendor::_t1mskc_u32(x)
}

#[no_mangle]
pub fn t1mskc_u64(x: u64) -> u64 {
    stdsimd::vendor::_t1mskc_u64(x)
}
