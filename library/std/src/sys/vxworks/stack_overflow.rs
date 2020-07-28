#![cfg_attr(test, allow(dead_code))]

use self::imp::{drop_handler, make_handler};

pub use self::imp::cleanup;
pub use self::imp::init;

pub struct Handler {
    _data: *mut libc::c_void,
}

impl Handler {
    pub unsafe fn new() -> Handler {
        make_handler()
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            drop_handler(self);
        }
    }
}

mod imp {
    use crate::ptr;

    pub unsafe fn init() {}

    pub unsafe fn cleanup() {}

    pub unsafe fn make_handler() -> super::Handler {
        super::Handler { _data: ptr::null_mut() }
    }

    pub unsafe fn drop_handler(_handler: &mut super::Handler) {}
}
