//@ check-pass
#![deny(improper_ctypes)]
#![feature(generic_nonzero)]

pub struct Error(std::num::NonZero<u32>);

extern "Rust" {
    fn foo(dest: &mut [u8]) -> Result<(), Error>;
}

fn main() {
    let _ = unsafe { foo(&mut []) };
}
