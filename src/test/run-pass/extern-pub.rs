use std::libc;

extern {
    pub unsafe fn debug_get_stk_seg() -> *libc::c_void;
}

pub fn main() {
}
