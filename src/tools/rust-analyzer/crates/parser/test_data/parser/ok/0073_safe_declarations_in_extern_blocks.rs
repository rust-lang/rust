unsafe extern {
    // sqrt (from libm) may be called with any `f64`
    pub safe fn sqrt(x: f64) -> f64;

    // strlen (from libc) requires a valid pointer,
    // so we mark it as being an unsafe fn
    pub unsafe fn strlen(p: *const c_char) -> usize;

    // this function doesn't say safe or unsafe, so it defaults to unsafe
    pub fn free(p: *mut core::ffi::c_void);

    pub safe static mut COUNTER: i32;

    pub unsafe static IMPORTANT_BYTES: [u8; 256];

    pub safe static LINES: SyncUnsafeCell<i32>;
}
