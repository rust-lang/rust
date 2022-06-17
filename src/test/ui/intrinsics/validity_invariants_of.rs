// run-pass
// revisions: disabled normal sanitized
// [disabled]compile-flags: -Zno-validity-invariant-checks
// [sanitized]compile-flags: -Z sanitizer=memory
#![feature(core_intrinsics, const_intrinsic_validity_invariants_of)]

use std::mem::MaybeUninit;
use std::intrinsics::{Invariant, InvariantSize, validity_invariants_of, assert_validity_of};
use std::num::NonZeroU16;

#[repr(C)]
struct MyStruct {
    a: NonZeroU16,
    b: u8,
    c: bool,
}

const MY_STRUCT_INVARIANTS: &'static [Invariant] = validity_invariants_of::<MyStruct>();
const OPTION_INVARIANTS: &'static [Invariant] = validity_invariants_of::<Option<fn()>>();

fn main() {
    let mut invs: Vec<Invariant> = MY_STRUCT_INVARIANTS.to_vec();

    invs.sort_by_key(|x| x.offset);

    if cfg!(disabled) {
        assert_eq!(&invs, &[]);
    } else if cfg!(normal) {
        assert_eq!(&invs, &[
            Invariant {
                offset: 0,
                size: InvariantSize::U16,
                valid_range_start: 1,
                valid_range_end: 65535
            },
            Invariant {
                offset: 3,
                size: InvariantSize::U8,
                valid_range_start: 0,
                valid_range_end: 1
            },
        ]);
    } else {
        assert!(cfg!(sanitized));

        assert_eq!(&invs, &[
            Invariant {
                offset: 0,
                size: InvariantSize::U16,
                valid_range_start: 1,
                valid_range_end: 65535
            },
            Invariant {
                offset: 2,
                size: InvariantSize::U8,
                valid_range_start: 0,
                valid_range_end: 255
            },
            Invariant {
                offset: 3,
                size: InvariantSize::U8,
                valid_range_start: 0,
                valid_range_end: 1
            },
        ]);
    }


    unsafe {
        let v = MyStruct { a: NonZeroU16::new(1).unwrap(), b: 2, c: true };
        assert!(assert_validity_of(&v as *const _));
    }

    if cfg!(sanitized) {
        assert_eq!(OPTION_INVARIANTS, &[
            Invariant {
                offset: 0,
                size: InvariantSize::Pointer,
                valid_range_start: 1,
                valid_range_end: 0,
            },
        ]);
    } else {
        assert_eq!(OPTION_INVARIANTS, &[]);
    }


    unsafe {
        let p = MaybeUninit::<Option<fn()>>::zeroed().as_ptr();
        assert!(assert_validity_of(p));
    }

    // There's two code paths for generating the data, be sure to test that compile time and
    // runtime matches.
    assert_eq!(validity_invariants_of::<Option<fn()>>(), OPTION_INVARIANTS);
    assert_eq!(validity_invariants_of::<MyStruct>(), MY_STRUCT_INVARIANTS);
}
