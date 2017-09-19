extern crate stdsimd;

#[no_mangle]
pub fn bextr_u32(x: u32, y: u32, z: u32) -> u32 {
    stdsimd::vendor::_bextr_u32(x, y, z)
}

#[no_mangle]
pub fn bextr_u64(x: u64, y: u64, z: u64) -> u64 {
    stdsimd::vendor::_bextr_u64(x, y, z)
}

#[no_mangle]
pub fn bextr2_u32(x: u32, y: u32) -> u32 {
    stdsimd::vendor::_bextr2_u32(x, y)
}

#[no_mangle]
pub fn bextr2_u64(x: u64, y: u64) -> u64 {
    stdsimd::vendor::_bextr2_u64(x, y)
}
