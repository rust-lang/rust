// FIXME: validation disabled because it checks these references too eagerly.
// compile-flags: -Zmiri-disable-validation

fn main() {
    let x = 2usize as *const u32;
    // This is not aligned, but we immediately cast it to a raw ptr so that must be ok.
    let _y = unsafe { &*x as *const u32 };

    let x = 0usize as *const u32;
    // This is NULL, but we immediately cast it to a raw ptr so that must be ok.
    let _y = unsafe { &*x as *const u32 };
}
