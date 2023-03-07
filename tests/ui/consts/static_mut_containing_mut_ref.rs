// build-pass (FIXME(62277): could be check-pass?)

static mut STDERR_BUFFER_SPACE: [u8; 42] = [0u8; 42];

pub static mut STDERR_BUFFER: *mut [u8] = unsafe { &mut STDERR_BUFFER_SPACE };

fn main() {}
