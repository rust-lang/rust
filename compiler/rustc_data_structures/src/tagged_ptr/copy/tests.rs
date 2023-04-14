use std::ptr;

use crate::tagged_ptr::{CopyTaggedPtr, Pointer, Tag, Tag2};

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
