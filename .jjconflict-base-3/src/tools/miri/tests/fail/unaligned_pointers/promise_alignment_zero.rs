#[path = "../../utils/mod.rs"]
mod utils;

fn main() {
    let buffer = [0u32; 128];
    unsafe { utils::miri_promise_symbolic_alignment(buffer.as_ptr().cast(), 0) };
    //~^ERROR: alignment must be a power of 2
}
