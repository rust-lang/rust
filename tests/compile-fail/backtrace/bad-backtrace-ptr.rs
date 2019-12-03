extern "Rust" {
    fn miri_resolve_frame(version: u8, ptr: *mut ());
}

fn main() {
    unsafe {
        miri_resolve_frame(0, 0 as *mut _); //~ ERROR Undefined Behavior: Expected a pointer
    }
}
