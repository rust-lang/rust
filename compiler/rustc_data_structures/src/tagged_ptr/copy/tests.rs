use std::ptr;

use crate::stable_hasher::{HashStable, StableHasher};
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

#[test]
fn stable_hash_hashes_as_tuple() {
    let hash_packed = {
        let mut hasher = StableHasher::new();
        tag_ptr(&12, Tag2::B11).hash_stable(&mut (), &mut hasher);

        hasher.finalize()
    };

    let hash_tupled = {
        let mut hasher = StableHasher::new();
        (&12, Tag2::B11).hash_stable(&mut (), &mut hasher);
        hasher.finalize()
    };

    assert_eq!(hash_packed, hash_tupled);
}

/// Helper to create tagged pointers without specifying `COMPARE_PACKED` if it does not matter.
fn tag_ptr<P: Pointer, T: Tag>(ptr: P, tag: T) -> CopyTaggedPtr<P, T, true> {
    CopyTaggedPtr::new(ptr, tag)
}
