extern crate stdsimd;

#[no_mangle]
pub fn popcnt_u32(x: u32) -> u32 {
    stdsimd::vendor::_popcnt32(x)
}

#[no_mangle]
pub fn popcnt_u64(x: u64) -> u64 {
    stdsimd::vendor::_popcnt64(x)
}
