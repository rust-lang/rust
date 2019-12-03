extern "Rust" {
    fn miri_resolve_frame(version: u8, ptr: *mut ());
}

fn main() {
    unsafe {
        miri_resolve_frame(1, 0 as *mut _); //~ ERROR  Undefined Behavior: Unknown `miri_resolve_frame` version 1
    }
}
