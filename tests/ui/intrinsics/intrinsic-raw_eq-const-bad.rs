#![feature(core_intrinsics)]
#![feature(const_intrinsic_raw_eq)]

const RAW_EQ_PADDING: bool = unsafe {
    std::intrinsics::raw_eq(&(1_u8, 2_u16), &(1_u8, 2_u16))
//~^ ERROR evaluation of constant value failed
//~| requires initialized memory
};

const RAW_EQ_PTR: bool = unsafe {
    std::intrinsics::raw_eq(&(&0), &(&1))
//~^ ERROR evaluation of constant value failed
//~| `raw_eq` on bytes with provenance
};

pub fn main() {
}
