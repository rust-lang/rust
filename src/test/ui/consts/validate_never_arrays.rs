#![feature(const_raw_ptr_deref, never_type)]

const FOO: &[!; 1] = unsafe { &*(1_usize as *const [!; 1]) }; //~ ERROR undefined behavior

fn main() {}
