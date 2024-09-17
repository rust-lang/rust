//@only-target: linux
fn main() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    // This will pass when blocking is implemented.
    let flags = libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    // Write u64 - 1.
    let mut sized_8_data: [u8; 8] = (u64::MAX - 1).to_ne_bytes();
    let res: i64 = unsafe {
        libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
    };
    assert_eq!(res, 8);

    // Write 1.
    sized_8_data = 1_u64.to_ne_bytes();
    // Write 1 to the counter.
    let _res: i64 = unsafe {
        libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap() //~ERROR: blocking is unsupported
    };
}
