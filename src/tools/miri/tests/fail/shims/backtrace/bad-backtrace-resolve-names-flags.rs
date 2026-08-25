extern "Rust" {
    fn miri_backtrace_size(flags: u64) -> usize;
    fn miri_get_backtrace(flags: u64, buf: *mut *mut ());
    fn miri_resolve_frame_names(ptr: *mut (), flags: u64, name_buf: *mut u8, filename_buf: *mut u8);
}

fn main() {
    unsafe {
        let mut buf = vec![std::ptr::null_mut(); miri_backtrace_size(0)];

        miri_get_backtrace(1, buf.as_mut_ptr());

        // miri_resolve_frame_names will error from an invalid backtrace before it will from invalid flags
        miri_resolve_frame_names(buf[0], 2, std::ptr::null_mut(), std::ptr::null_mut()); //~ ERROR:  unsupported operation: unknown `miri_resolve_frame_names` flags 2
    }
}
