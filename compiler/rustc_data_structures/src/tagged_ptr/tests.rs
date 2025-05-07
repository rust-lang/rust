use std::ptr;

use rustc_hashes::Hash128;

use super::*;
use crate::stable_hasher::{HashStable, StableHasher};

/// A tag type used in [`TaggedRef`] tests.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Tag2 {
    B00 = 0b00,
    B01 = 0b01,
    B10 = 0b10,
    B11 = 0b11,
}

unsafe impl Tag for Tag2 {
    const BITS: u32 = 2;

    fn into_usize(self) -> usize {
        self as _
    }

    unsafe fn from_usize(tag: usize) -> Self {
        match tag {
            0b00 => Tag2::B00,
            0b01 => Tag2::B01,
            0b10 => Tag2::B10,
            0b11 => Tag2::B11,
            _ => unreachable!(),
        }
    }
}

impl<HCX> crate::stable_hasher::HashStable<HCX> for Tag2 {
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut crate::stable_hasher::StableHasher) {
        (*self as u8).hash_stable(hcx, hasher);
    }
}

#[test]
fn smoke() {
    let value = 12u32;
    let reference = &value;
    let tag = Tag2::B01;

    let ptr = TaggedRef::new(reference, tag);

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
        TaggedRef::new(&12, Tag2::B11).hash_stable(&mut (), &mut hasher);
        hasher.finish::<Hash128>()
    };

    let hash_tupled = {
        let mut hasher = StableHasher::new();
        (&12, Tag2::B11).hash_stable(&mut (), &mut hasher);
        hasher.finish::<Hash128>()
    };

    assert_eq!(hash_packed, hash_tupled);
}

/// Test that `new` does not compile if there is not enough alignment for the
/// tag in the pointer.
///
/// ```compile_fail,E0080
/// use rustc_data_structures::tagged_ptr::{TaggedRef, Tag};
///
/// #[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// enum Tag2 { B00 = 0b00, B01 = 0b01, B10 = 0b10, B11 = 0b11 };
///
/// unsafe impl Tag for Tag2 {
///     const BITS: u32 = 2;
///
///     fn into_usize(self) -> usize { todo!() }
///     unsafe fn from_usize(tag: usize) -> Self { todo!() }
/// }
///
/// let value = 12u16;
/// let reference = &value;
/// let tag = Tag2::B01;
///
/// let _ptr = TaggedRef::<_, _, true>::new(reference, tag);
/// ```
// For some reason miri does not get the compile error
// probably it `check`s instead of `build`ing?
#[cfg(not(miri))]
const _: () = ();
