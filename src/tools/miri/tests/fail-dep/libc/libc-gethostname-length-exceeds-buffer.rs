//@ignore-target: windows # No libc

fn main() {
    let mut name = [0u8; 5];
    unsafe {
        libc::gethostname(name.as_mut_ptr().cast(), 6); //~ERROR: memory access
    }
}
