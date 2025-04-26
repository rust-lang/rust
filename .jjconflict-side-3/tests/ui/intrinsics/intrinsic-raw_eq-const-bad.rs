//@ normalize-stderr: "[[:xdigit:]]{2} __ ([[:xdigit:]]{2}\s){2}" -> "HEX_DUMP"
#![feature(core_intrinsics)]

const RAW_EQ_PADDING: bool = unsafe {
    std::intrinsics::raw_eq(&(1_u8, 2_u16), &(1_u8, 2_u16))
//~^ ERROR requires initialized memory
};

const RAW_EQ_PTR: bool = unsafe {
    std::intrinsics::raw_eq(&(&0), &(&1))
//~^ ERROR unable to turn pointer into integer
};

const RAW_EQ_NOT_ALIGNED: bool = unsafe {
    let arr = [0u8; 4];
    let aref = &*arr.as_ptr().cast::<i32>();
    std::intrinsics::raw_eq(aref, aref)
//~^ ERROR alignment
};

pub fn main() {
}
