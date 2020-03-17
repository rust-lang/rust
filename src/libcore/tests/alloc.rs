use core::alloc::Layout;
use core::ptr::NonNull;

#[test]
fn const_unchecked_layout() {
    const SIZE: usize = 0x2000;
    const ALIGN: usize = 0x1000;
    const LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(SIZE, ALIGN) };
    const DANGLING: NonNull<u8> = LAYOUT.dangling();
    assert_eq!(LAYOUT.size(), SIZE);
    assert_eq!(LAYOUT.align(), ALIGN);
    assert_eq!(Some(DANGLING), NonNull::new(ALIGN as *mut u8));
}
