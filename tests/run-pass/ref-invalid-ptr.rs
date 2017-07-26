fn main() {
    let x = 2usize as *const u32;
    let _y = unsafe { &*x as *const u32 };

    let x = 0usize as *const u32;
    let _y = unsafe { &*x as *const u32 };
}
