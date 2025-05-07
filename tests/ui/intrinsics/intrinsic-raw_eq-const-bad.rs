#![feature(core_intrinsics)]

const RAW_EQ_PADDING: bool = unsafe {
    std::intrinsics::raw_eq(&(1_u8, 2_u16), &(1_u8, 2_u16))
//~^ ERROR evaluation of constant value failed
//~| NOTE requires initialized memory
};

const RAW_EQ_PTR: bool = unsafe {
    std::intrinsics::raw_eq(&(&0), &(&1))
//~^ ERROR evaluation of constant value failed
//~| NOTE unable to turn pointer into integer
};

const RAW_EQ_NOT_ALIGNED: bool = unsafe {
    let arr = [0u8; 4];
    let aref = &*arr.as_ptr().cast::<i32>();
    std::intrinsics::raw_eq(aref, aref)
//~^ ERROR evaluation of constant value failed
//~| NOTE alignment
};

pub fn main() {
}
