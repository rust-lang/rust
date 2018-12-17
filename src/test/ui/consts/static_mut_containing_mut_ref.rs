// compile-pass

static mut STDERR_BUFFER_SPACE: [u8; 42] = [0u8; 42];

pub static mut STDERR_BUFFER: *mut [u8] = unsafe { &mut STDERR_BUFFER_SPACE };

fn main() {}
