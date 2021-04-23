// Aligned to two bytes
const DATA: [u16; 2] = [u16::from_ne_bytes([0x01, 0x23]), u16::from_ne_bytes([0x45, 0x67])];

const fn unaligned_ptr() -> *const u16 {
    // Since DATA.as_ptr() is aligned to two bytes, adding 1 byte to that produces an unaligned *const u16
    unsafe { (DATA.as_ptr() as *const u8).add(1) as *const u16 }
}

#[test]
fn read() {
    use core::ptr;

    const FOO: i32 = unsafe { ptr::read(&42 as *const i32) };
    assert_eq!(FOO, 42);

    const ALIGNED: i32 = unsafe { ptr::read_unaligned(&42 as *const i32) };
    assert_eq!(ALIGNED, 42);

    const UNALIGNED_PTR: *const u16 = unaligned_ptr();

    const UNALIGNED: u16 = unsafe { ptr::read_unaligned(UNALIGNED_PTR) };
    assert_eq!(UNALIGNED, u16::from_ne_bytes([0x23, 0x45]));
}

#[test]
fn const_ptr_read() {
    const FOO: i32 = unsafe { (&42 as *const i32).read() };
    assert_eq!(FOO, 42);

    const ALIGNED: i32 = unsafe { (&42 as *const i32).read_unaligned() };
    assert_eq!(ALIGNED, 42);

    const UNALIGNED_PTR: *const u16 = unaligned_ptr();

    const UNALIGNED: u16 = unsafe { UNALIGNED_PTR.read_unaligned() };
    assert_eq!(UNALIGNED, u16::from_ne_bytes([0x23, 0x45]));
}

#[test]
fn mut_ptr_read() {
    const FOO: i32 = unsafe { (&42 as *const i32 as *mut i32).read() };
    assert_eq!(FOO, 42);

    const ALIGNED: i32 = unsafe { (&42 as *const i32 as *mut i32).read_unaligned() };
    assert_eq!(ALIGNED, 42);

    const UNALIGNED_PTR: *mut u16 = unaligned_ptr() as *mut u16;

    const UNALIGNED: u16 = unsafe { UNALIGNED_PTR.read_unaligned() };
    assert_eq!(UNALIGNED, u16::from_ne_bytes([0x23, 0x45]));
}

#[test]
fn write() {
    use core::ptr;

    const fn write_aligned() -> i32 {
        let mut res = 0;
        unsafe {
            ptr::write(&mut res as *mut _, 42);
        }
        res
    }
    const ALIGNED: i32 = write_aligned();
    assert_eq!(ALIGNED, 42);

    const fn write_unaligned() -> [u16; 2] {
        let mut two_aligned = [0u16; 2];
        unsafe {
            let unaligned_ptr = (two_aligned.as_mut_ptr() as *mut u8).add(1) as *mut u16;
            ptr::write_unaligned(unaligned_ptr, u16::from_ne_bytes([0x23, 0x45]));
        }
        two_aligned
    }
    const UNALIGNED: [u16; 2] = write_unaligned();
    assert_eq!(UNALIGNED, [u16::from_ne_bytes([0x00, 0x23]), u16::from_ne_bytes([0x45, 0x00])]);
}

#[test]
fn mut_ptr_write() {
    const fn aligned() -> i32 {
        let mut res = 0;
        unsafe {
            (&mut res as *mut i32).write(42);
        }
        res
    }
    const ALIGNED: i32 = aligned();
    assert_eq!(ALIGNED, 42);

    const fn write_unaligned() -> [u16; 2] {
        let mut two_aligned = [0u16; 2];
        unsafe {
            let unaligned_ptr = (two_aligned.as_mut_ptr() as *mut u8).add(1) as *mut u16;
            unaligned_ptr.write_unaligned(u16::from_ne_bytes([0x23, 0x45]));
        }
        two_aligned
    }
    const UNALIGNED: [u16; 2] = write_unaligned();
    assert_eq!(UNALIGNED, [u16::from_ne_bytes([0x00, 0x23]), u16::from_ne_bytes([0x45, 0x00])]);
}
