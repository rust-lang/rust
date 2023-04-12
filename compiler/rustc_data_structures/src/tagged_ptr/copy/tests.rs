use std::ptr;

use crate::tagged_ptr::{CopyTaggedPtr, Pointer, Tag};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Tag2 {
    B00 = 0b00,
    B01 = 0b01,
    B10 = 0b10,
    B11 = 0b11,
}

unsafe impl Tag for Tag2 {
    const BITS: usize = 2;

    fn into_usize(self) -> usize {
        self as _
    }

    unsafe fn from_usize(tag: usize) -> Self {
        const B00: usize = Tag2::B00 as _;
        const B01: usize = Tag2::B01 as _;
        const B10: usize = Tag2::B10 as _;
        const B11: usize = Tag2::B11 as _;
        match tag {
            B00 => Tag2::B00,
            B01 => Tag2::B01,
            B10 => Tag2::B10,
            B11 => Tag2::B11,
            _ => unreachable!(),
        }
    }
}

#[test]
fn smoke() {
    let value = 12u32;
    let reference = &value;
    let tag = Tag2::B01;

    let ptr = tag_ptr(reference, tag);

    assert_eq!(ptr.tag(), tag);
    assert_eq!(*ptr, 12);
    assert!(ptr::eq(ptr.pointer(), reference));

    let copy = ptr;

    let mut ptr = ptr;
    ptr.set_tag(Tag2::B00);
    assert_eq!(ptr.tag(), Tag2::B00);

    assert_eq!(copy.tag(), tag);
    assert_eq!(*copy, 12);
    assert!(ptr::eq(copy.pointer(), reference));
}

/// Helper to create tagged pointers without specifying `COMPARE_PACKED` if it does not matter.
fn tag_ptr<P: Pointer, T: Tag>(ptr: P, tag: T) -> CopyTaggedPtr<P, T, true> {
    CopyTaggedPtr::new(ptr, tag)
}
