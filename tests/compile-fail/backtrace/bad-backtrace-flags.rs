extern "Rust" {
    fn miri_get_backtrace(flags: u64, buf: *mut *mut ());
}

fn main() {
    unsafe {
        miri_get_backtrace(2, 0 as *mut _); //~ ERROR  unsupported operation: unknown `miri_get_backtrace` flags 2
    }
}
