use std::libc::c_int;
use std::libc;

pub struct Fd(c_int);

impl Drop for Fd {
    fn finalize(&self) {
        unsafe {
            libc::close(**self);
        }
    }
}

pub fn main() {
}
