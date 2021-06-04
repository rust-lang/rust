fn main() {
    const X: usize = unsafe {
        main as usize //~ ERROR casting pointers to integers in constants is unstable
    };
    const Y: u32 = 0;
    const Z: usize = unsafe {
        &Y as *const u32 as usize //~ ERROR is unstable
    };
}

const fn test() -> usize {
    &0 as *const i32 as usize //~ ERROR is unstable
}
