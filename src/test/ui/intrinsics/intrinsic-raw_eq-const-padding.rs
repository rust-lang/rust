#![feature(core_intrinsics)]
#![feature(const_intrinsic_raw_eq)]
#![deny(const_err)]

const BAD_RAW_EQ_CALL: bool = unsafe {
    std::intrinsics::raw_eq(&(1_u8, 2_u16), &(1_u8, 2_u16))
//~^ ERROR evaluation of constant value failed
};

pub fn main() {
}
