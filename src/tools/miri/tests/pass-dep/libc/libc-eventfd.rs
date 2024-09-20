//@only-target: linux
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-preemption-rate=0

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::thread;

fn main() {
    test_read_write();
    test_race();
}

fn read_bytes<const N: usize>(fd: i32, buf: &mut [u8; N]) -> i32 {
    let res: i32 = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), N).try_into().unwrap() };
    return res;
}

fn write_bytes<const N: usize>(fd: i32, data: [u8; N]) -> i32 {
    let res: i32 =
        unsafe { libc::write(fd, data.as_ptr() as *const libc::c_void, N).try_into().unwrap() };
    return res;
}

fn test_read_write() {
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    // Write 1 to the counter.
    let res = write_bytes(fd, sized_8_data);
    assert_eq!(res, 8);

    // Read 1 from the counter.
    let mut buf: [u8; 8] = [0; 8];
    let res = read_bytes(fd, &mut buf);
    // Read returns number of bytes has been read, which is always 8.
    assert_eq!(res, 8);
    // Check the value of counter read.
    let counter = u64::from_ne_bytes(buf);
    assert_eq!(counter, 1);

    // After read, the counter is currently 0, read counter 0 should fail with return
    // value -1.
    let mut buf: [u8; 8] = [0; 8];
    let res = read_bytes(fd, &mut buf);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EAGAIN));
    assert_eq!(res, -1);

    // Write with supplied buffer bigger than 8 bytes should be allowed.
    let sized_9_data: [u8; 9];
    if cfg!(target_endian = "big") {
        // Adjust the data based on the endianness of host system.
        sized_9_data = [0, 0, 0, 0, 0, 0, 0, 1, 0];
    } else {
        sized_9_data = [1, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    let res = write_bytes(fd, sized_9_data);
    assert_eq!(res, 8);

    // Read with supplied buffer smaller than 8 bytes should fail with return
    // value -1.
    let mut buf: [u8; 7] = [1; 7];
    let res = read_bytes(fd, &mut buf);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);

    // Write with supplied buffer smaller than 8 bytes should fail with return
    // value -1.
    let size_7_data: [u8; 7] = [1; 7];
    let res = write_bytes(fd, size_7_data);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);

    // Read with supplied buffer bigger than 8 bytes should be allowed.
    let mut buf: [u8; 9] = [1; 9];
    let res = read_bytes(fd, &mut buf);
    assert_eq!(res, 8);

    // Write u64::MAX should fail.
    let u64_max_bytes: [u8; 8] = [255; 8];
    let res = write_bytes(fd, u64_max_bytes);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);
}

fn test_race() {
    static mut VAL: u8 = 0;
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 8] = [0; 8];
        let res = read_bytes(fd, &mut buf);
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1);
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    let data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = write_bytes(fd, data);
    // write returns number of bytes written, which is always 8.
    assert_eq!(res, 8);
    thread::yield_now();
    thread1.join().unwrap();
}
