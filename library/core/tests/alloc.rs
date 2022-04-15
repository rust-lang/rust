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

#[test]
fn layout_debug_shows_log2_of_alignment() {
    // `Debug` is not stable, but here's what it does right now
    let layout = Layout::from_size_align(24576, 8192).unwrap();
    let s = format!("{:?}", layout);
    assert_eq!(s, "Layout { size: 24576, align: 8192 (1 << 13) }");
}

// Running this normally doesn't do much, but it's also run in Miri, which
// will double-check that these are allowed by the validity invariants.
#[test]
fn layout_accepts_all_valid_alignments() {
    for align in 0..usize::BITS {
        let layout = Layout::from_size_align(0, 1_usize << align).unwrap();
        assert_eq!(layout.align(), 1_usize << align);
    }
}
