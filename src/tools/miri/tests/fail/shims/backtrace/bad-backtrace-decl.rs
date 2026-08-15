extern "Rust" {
    fn miri_backtrace_size(flags: u64) -> usize;
    fn miri_get_backtrace(flags: u64, buf: *mut *mut ());
    fn miri_resolve_frame(ptr: *mut (), flags: u64);
}

fn main() {
    let size = unsafe { miri_backtrace_size(0) };
    let mut frames = vec![std::ptr::null_mut(); size];
    unsafe { miri_get_backtrace(1, frames.as_mut_ptr()) };
    for frame in frames.iter() {
        unsafe {
            miri_resolve_frame(*frame, 0); //~ ERROR: Undefined Behavior: bad declaration of miri_resolve_frame - should return a struct with 5 fields
        }
    }
}
