use core::alloc::Layout;
use core::ptr::{self, NonNull};

#[test]
fn const_unchecked_layout() {
    const SIZE: usize = 0x2000;
    const ALIGN: usize = 0x1000;
    const LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(SIZE, ALIGN) };
    const DANGLING: NonNull<u8> = LAYOUT.dangling();
    assert_eq!(LAYOUT.size(), SIZE);
    assert_eq!(LAYOUT.align(), ALIGN);
    assert_eq!(Some(DANGLING), NonNull::new(ptr::invalid_mut(ALIGN)));
}

#[test]
fn layout_debug_shows_log2_of_alignment() {
    // `Debug` is not stable, but here's what it does right now
    let layout = Layout::from_size_align(24576, 8192).unwrap();
    let s = format!("{:?}", layout);
    assert_eq!(s, "Layout { size: 24576, align: 8192 (1 << 13) }");
}

#[test]
fn layout_rejects_invalid_combinations() {
    assert!(Layout::from_size_align(3, 3).is_err()); // bad align
    assert!(Layout::from_size_align(1 << (usize::BITS - 1), 1).is_err()); // bad size
    assert!(Layout::from_size_align(isize::MAX as usize, 2).is_err()); // fails round-up
    assert!(Layout::from_size_align(1, 1 << (usize::BITS - 1)).is_err()); // fails round-up
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

#[test]
fn layout_accepts_various_valid_sizes() {
    for shift in 1..usize::BITS {
        let layout = Layout::from_size_align(usize::MAX >> shift, 1).unwrap();
        assert_eq!(layout.size(), usize::MAX >> shift);
    }
}
