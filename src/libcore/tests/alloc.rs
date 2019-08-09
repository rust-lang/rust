use core::alloc::Layout;

#[test]
fn const_unchecked_layout() {
    const SIZE: usize = 0x2000;
    const ALIGN: usize = 0x1000;
    const LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(SIZE, ALIGN) };
    assert_eq!(LAYOUT.size(), SIZE);
    assert_eq!(LAYOUT.align(), ALIGN);
}
