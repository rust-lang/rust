fn main() {
    let x = 13;
    let y = &x;
    let z = &y as *const &i32 as *const u8;
    // the deref fails, because we are reading only a part of the pointer
    let _val = unsafe { *z }; //~ ERROR unable to turn pointer into raw bytes
}
