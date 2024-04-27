use std::{ptr, sync::Arc};

use crate::tagged_ptr::{Pointer, Tag, Tag2, TaggedPtr};

#[test]
fn smoke() {
    let value = 12u32;
    let reference = &value;
    let tag = Tag2::B01;

    let ptr = tag_ptr(reference, tag);

    assert_eq!(ptr.tag(), tag);
    assert_eq!(*ptr, 12);

    let clone = ptr.clone();
    assert_eq!(clone.tag(), tag);
    assert_eq!(*clone, 12);

    let mut ptr = ptr;
    ptr.set_tag(Tag2::B00);
    assert_eq!(ptr.tag(), Tag2::B00);

    assert_eq!(clone.tag(), tag);
    assert_eq!(*clone, 12);
    assert!(ptr::eq(&*ptr, &*clone))
}

#[test]
fn boxed() {
    let value = 12u32;
    let boxed = Box::new(value);
    let tag = Tag2::B01;

    let ptr = tag_ptr(boxed, tag);

    assert_eq!(ptr.tag(), tag);
    assert_eq!(*ptr, 12);

    let clone = ptr.clone();
    assert_eq!(clone.tag(), tag);
    assert_eq!(*clone, 12);

    let mut ptr = ptr;
    ptr.set_tag(Tag2::B00);
    assert_eq!(ptr.tag(), Tag2::B00);

    assert_eq!(clone.tag(), tag);
    assert_eq!(*clone, 12);
    assert!(!ptr::eq(&*ptr, &*clone))
}

#[test]
fn arclones() {
    let value = 12u32;
    let arc = Arc::new(value);
    let tag = Tag2::B01;

    let ptr = tag_ptr(arc, tag);

    assert_eq!(ptr.tag(), tag);
    assert_eq!(*ptr, 12);

    let clone = ptr.clone();
    assert!(ptr::eq(&*ptr, &*clone))
}

/// Helper to create tagged pointers without specifying `COMPARE_PACKED` if it does not matter.
fn tag_ptr<P: Pointer, T: Tag>(ptr: P, tag: T) -> TaggedPtr<P, T, true> {
    TaggedPtr::new(ptr, tag)
}
