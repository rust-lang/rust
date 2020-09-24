extern "Rust" {
    fn miri_resolve_frame(ptr: *mut (), flags: u64);
}

fn main() {
    unsafe {
        miri_resolve_frame(0 as *mut _, 1); //~ ERROR  unsupported operation: unknown `miri_resolve_frame` flags 1
    }
}
