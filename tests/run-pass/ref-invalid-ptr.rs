// FIXME validation disabled because it checks these references too eagerly
// compile-flags: -Zmiri-disable-validation

fn main() {
    let x = 2usize as *const u32;
    // this is not aligned, but we immediately cast it to a raw ptr so that must be okay
    let _y = unsafe { &*x as *const u32 };

    let x = 0usize as *const u32;
    // this is NULL, but we immediately cast it to a raw ptr so that must be okay
    let _y = unsafe { &*x as *const u32 };
}
