#[repr(C)]
struct MiriFrame {
    name_len: usize,
    filename_len: usize,
    lineno: u32,
    colno: u32,
    fn_ptr: *mut (),
}

extern "Rust" {
    fn miri_backtrace_size(flags: u64) -> usize;
    fn miri_get_backtrace(flags: u64, buf: *mut *mut ());
    fn miri_resolve_frame(ptr: *mut (), flags: u64) -> MiriFrame;
}

fn main() {
    unsafe {
        let mut buf = vec![std::ptr::null_mut(); miri_backtrace_size(0)];

        miri_get_backtrace(1, buf.as_mut_ptr());

        // miri_resolve_frame will error from an invalid backtrace before it will from invalid flags
        miri_resolve_frame(buf[0], 2); //~ ERROR:  unsupported operation: unknown `miri_resolve_frame` flags 2
    }
}
