const WRITE: () = unsafe {
    *std::ptr::null_mut() = 0;
    //~^ ERROR dereferencing raw pointers in constants is unstable
    //~| HELP add `#![feature(const_raw_ptr_deref)]` to the crate attributes to enable
};

fn main() {}
