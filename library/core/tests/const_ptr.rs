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
