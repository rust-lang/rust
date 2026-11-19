//@ignore-target: windows # No libc IO on Windows

fn main() {
    let mut buf = [0u8; 4];
    // Reading from stdin blocks or returns EOF, but we first check the buffer.
    unsafe { libc::read(0, buf.as_mut_ptr().cast(), 5) };
    //~^ ERROR: only 4 bytes from the end of the allocation
}
