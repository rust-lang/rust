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
    assert_eq!(Some(DANGLING), NonNull::new(ptr::without_provenance_mut(ALIGN)));
}

#[test]
fn layout_round_up_to_align_edge_cases() {
    const MAX_SIZE: usize = isize::MAX as usize;

    for shift in 0..usize::BITS {
        let align = 1_usize << shift;
        let edge = (MAX_SIZE + 1) - align;
        let low = edge.saturating_sub(10);
        let high = edge.saturating_add(10);
        assert!(Layout::from_size_align(low, align).is_ok());
        assert!(Layout::from_size_align(high, align).is_err());
        for size in low..=high {
            assert_eq!(
                Layout::from_size_align(size, align).is_ok(),
                size.next_multiple_of(align) <= MAX_SIZE,
            );
        }
    }
}

#[test]
fn layout_array_edge_cases() {
    for_type::<i64>();
    for_type::<[i32; 0b10101]>();
    for_type::<[u8; 0b1010101]>();

    // Make sure ZSTs don't lead to divide-by-zero
    assert_eq!(Layout::array::<()>(usize::MAX).unwrap(), Layout::from_size_align(0, 1).unwrap());

    fn for_type<T>() {
        const MAX_SIZE: usize = isize::MAX as usize;

        let edge = (MAX_SIZE + 1) / size_of::<T>();
        let low = edge.saturating_sub(10);
        let high = edge.saturating_add(10);
        assert!(Layout::array::<T>(low).is_ok());
        assert!(Layout::array::<T>(high).is_err());
        for n in low..=high {
            assert_eq!(Layout::array::<T>(n).is_ok(), n * size_of::<T>() <= MAX_SIZE);
        }
    }
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
