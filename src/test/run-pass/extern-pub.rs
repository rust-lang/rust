use std::libc;

extern {
    pub unsafe fn free(p: *libc::c_void);
}

pub fn main() {
}
