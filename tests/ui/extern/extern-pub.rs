//@ run-pass

extern "C" {
    pub fn free(p: *mut std::ffi::c_void);
}

pub fn main() {}
