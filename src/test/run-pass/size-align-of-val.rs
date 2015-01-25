// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unstable)]

extern crate core;

use core::intrinsics::{size_of_val, min_align_of_val, pref_align_of_val};

struct SmallStruct(u8);
struct BigStruct([u64; 100]);
trait Trait { }
impl Trait for SmallStruct { }
impl Trait for BigStruct { }

trait PackedSize {
    fn packed_size(helper: Option<Self>) -> usize;
}

struct UnsizedStruct(i32, i32, str);
struct BigUnsizedStruct(i32, i32, [u64]);
struct SizedPartUS(i32, i32);
impl PackedSize for SizedPartUS {
    fn packed_size(_: Option<Self>) -> usize {
        core::mem::size_of::<i32>() * 2
    }
}
struct RecursiveUnsizedStruct(u64, u8, UnsizedStruct);
struct BigRecursiveUnsizedStruct(u64, u8, BigUnsizedStruct);
struct SizedPartRUS(u64, u8, SizedPartUS);
impl PackedSize for SizedPartRUS {
    fn packed_size(_: Option<Self>) -> usize {
        //u64 + u8 + [u8; 3] for padding + packed_size of SizedPartUS
        core::mem::size_of::<u64>() + core::mem::size_of::<u8>() * 4 +
        PackedSize::packed_size(None::<SizedPartUS>)
    }
}
struct UnsizedStructTrait(i8, i8, Trait + 'static);
struct SizedPartUST(i8, i8);
impl PackedSize for SizedPartUST {
    fn packed_size(_: Option<Self>) -> usize {
        core::mem::size_of::<i8>() * 2
    }
}

static SMALL_STRUCT: SmallStruct = SmallStruct(0);
static BIG_STRUCT: BigStruct = BigStruct([0; 100]);
static STRING: &'static str = "Test String Strictly For Testing Purposes";
static U8_SLICE: &'static [u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
static U64_SLICE: &'static [u64] = &[1, 2, 3, 4, 5, 6];

fn expected_unsized_struct_size<SizedPart: PackedSize, U: ?Sized>(us_part: &U) -> usize {
    let sized_size = PackedSize::packed_size(None::<SizedPart>);
    let sized_align = core::mem::min_align_of::<SizedPart>();

    let us_size = unsafe { size_of_val(us_part) };
    let us_align = unsafe { min_align_of_val(us_part) };

    let us_offset = match sized_size % us_align {
        0 => sized_size,
        rem => sized_size + (us_align - rem)
    };
    let align = core::cmp::max(us_align, sized_align);

    let size = us_offset + us_size;

    match size % align {
        0 => size,
        rem => size + (align - rem)
    }
}

fn main() {
    //A few helper variables that can't be constructed statically easily.
    let unsized_struct = unsafe {
        core::mem::transmute::<_, &UnsizedStruct>(STRING)
    };
    let recursive_unsized_struct = unsafe {
        core::mem::transmute::<_, &RecursiveUnsizedStruct>(unsized_struct)
    };
    let big_unsized_struct = unsafe {
        core::mem::transmute::<_, &BigUnsizedStruct>(U64_SLICE)
    };
    let big_recursive_unsized_struct = unsafe {
        core::mem::transmute::<_, &BigRecursiveUnsizedStruct>(big_unsized_struct)
    };
    let trait_unsized_struct_b = unsafe {
        core::mem::transmute::<_, &UnsizedStructTrait>(&BIG_STRUCT as &Trait)
    };
    let trait_unsized_struct_s = unsafe {
        core::mem::transmute::<_, &UnsizedStructTrait>(&SMALL_STRUCT as &Trait)
    };

    //Sizes
    assert_eq!(STRING.len(),
               unsafe { size_of_val(STRING) });
    assert_eq!(U8_SLICE.len() * core::mem::size_of::<u8>(),
               unsafe { size_of_val(U8_SLICE) });
    assert_eq!(U64_SLICE.len() * core::mem::size_of::<u64>(),
               unsafe { size_of_val(U64_SLICE) });
    assert_eq!(core::mem::size_of::<BigStruct>(),
               unsafe { size_of_val(&BIG_STRUCT as &Trait) });
    assert_eq!(core::mem::size_of::<SmallStruct>(),
               unsafe { size_of_val(&SMALL_STRUCT as &Trait) });
    assert_eq!(expected_unsized_struct_size::<SizedPartUS, _>(STRING),
               unsafe { size_of_val(unsized_struct) });
    assert_eq!(expected_unsized_struct_size::<SizedPartRUS, _>(STRING),
               unsafe { size_of_val(recursive_unsized_struct) });
    assert_eq!(expected_unsized_struct_size::<SizedPartUS, _>(U64_SLICE),
               unsafe { size_of_val(big_unsized_struct) });
    assert_eq!(expected_unsized_struct_size::<SizedPartRUS, _>(U64_SLICE),
               unsafe { size_of_val(big_recursive_unsized_struct) });
    assert_eq!(expected_unsized_struct_size::<SizedPartUST, _>(&BIG_STRUCT as &Trait),
               unsafe { size_of_val(trait_unsized_struct_b) });
    assert_eq!(expected_unsized_struct_size::<SizedPartUST, _>(&SMALL_STRUCT as &Trait),
               unsafe { size_of_val(trait_unsized_struct_s) });

    //Min alignments
    assert_eq!(core::mem::min_align_of::<u8>(),
               unsafe { min_align_of_val(STRING) });
    assert_eq!(core::mem::min_align_of::<u8>(),
               unsafe { min_align_of_val(U8_SLICE) });
    assert_eq!(core::mem::min_align_of::<u64>(),
               unsafe { min_align_of_val(U64_SLICE) });
    assert_eq!(core::mem::min_align_of::<BigStruct>(),
               unsafe { min_align_of_val(&BIG_STRUCT as &Trait) });
    assert_eq!(core::mem::min_align_of::<SmallStruct>(),
               unsafe { min_align_of_val(&SMALL_STRUCT as &Trait) });
    assert_eq!(core::cmp::max(core::mem::min_align_of::<SizedPartUS>(),
                              unsafe { min_align_of_val(STRING) }),
               unsafe { min_align_of_val(unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::min_align_of::<SizedPartRUS>(),
                              unsafe { min_align_of_val(STRING) }),
               unsafe { min_align_of_val(recursive_unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::min_align_of::<SizedPartUS>(),
                              unsafe { min_align_of_val(U64_SLICE) }),
               unsafe { min_align_of_val(big_unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::min_align_of::<SizedPartRUS>(),
                              unsafe { min_align_of_val(U64_SLICE) }),
               unsafe { min_align_of_val(big_recursive_unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::min_align_of::<SizedPartUST>(),
                              unsafe { min_align_of_val(&BIG_STRUCT as &Trait) }),
               unsafe { min_align_of_val(trait_unsized_struct_b) });
    assert_eq!(core::cmp::max(core::mem::min_align_of::<SizedPartUST>(),
                              unsafe { min_align_of_val(&SMALL_STRUCT as &Trait) }),
               unsafe { min_align_of_val(trait_unsized_struct_s) });

    //Pref alignments
    assert_eq!(core::mem::align_of::<u8>(),
               unsafe { pref_align_of_val(STRING) });
    assert_eq!(core::mem::align_of::<u8>(),
               unsafe { pref_align_of_val(U8_SLICE) });
    assert_eq!(core::mem::align_of::<u64>(),
               unsafe { pref_align_of_val(U64_SLICE) });
    assert_eq!(core::cmp::max(core::mem::align_of::<SizedPartUS>(),
                              unsafe { pref_align_of_val(STRING) }),
               unsafe { pref_align_of_val(unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::align_of::<SizedPartRUS>(),
                              unsafe { pref_align_of_val(STRING) }),
               unsafe { pref_align_of_val(recursive_unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::align_of::<SizedPartUS>(),
                              unsafe { pref_align_of_val(U64_SLICE) }),
               unsafe { pref_align_of_val(big_unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::align_of::<SizedPartRUS>(),
                              unsafe { pref_align_of_val(U64_SLICE) }),
               unsafe { pref_align_of_val(big_recursive_unsized_struct) });
    assert_eq!(core::cmp::max(core::mem::align_of::<SizedPartUST>(),
                              unsafe { pref_align_of_val(&BIG_STRUCT as &Trait) }),
               unsafe { pref_align_of_val(trait_unsized_struct_b) });
    assert_eq!(core::cmp::max(core::mem::align_of::<SizedPartUST>(),
                              unsafe { pref_align_of_val(&SMALL_STRUCT as &Trait) }),
               unsafe { pref_align_of_val(trait_unsized_struct_s) });
}

