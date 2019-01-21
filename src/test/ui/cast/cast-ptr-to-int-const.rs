// gate-test-const_raw_ptr_to_usize_cast

fn main() {
    const X: u32 = unsafe {
        main as u32 //~ ERROR casting pointers to integers in constants is unstable
    };
    const Y: u32 = 0;
    const Z: u32 = unsafe {
        &Y as *const u32 as u32 //~ ERROR is unstable
    };
}
