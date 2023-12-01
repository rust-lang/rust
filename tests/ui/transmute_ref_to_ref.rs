//@no-rustfix

#![deny(clippy::transmute_ptr_to_ptr)]
#![allow(dead_code)]

fn main() {
    unsafe {
        let single_u64: &[u64] = &[0xDEAD_BEEF_DEAD_BEEF];
        let bools: &[bool] = unsafe { std::mem::transmute(single_u64) };
        //~^ ERROR: transmute from a reference to a reference
        let a: &[u32] = &[0x12345678, 0x90ABCDEF, 0xFEDCBA09, 0x87654321];
        let b: &[u8] = unsafe { std::mem::transmute(a) };
        //~^ ERROR: transmute from a reference to a reference
        let bytes = &[1u8, 2u8, 3u8, 4u8] as &[u8];
        let alt_slice: &[u32] = unsafe { core::mem::transmute(bytes) };
        //~^ ERROR: transmute from a reference to a reference
    }
}
