extern "Rust" {
    fn miri_get_backtrace() -> Box<[*mut ()]>;
    fn miri_resolve_frame(version: u8, ptr: *mut ());
}

fn main() {
    let frames = unsafe { miri_get_backtrace() };
    for frame in frames.into_iter() {
        unsafe {
            miri_resolve_frame(0, *frame); //~ ERROR Undefined Behavior: Bad declaration of miri_resolve_frame - should return a struct with 4 fields
        }
    }
}
