// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The virtual memory representation of the MIR interpreter

use super::{
    UndefMask,
    Relocations,
};

use ty::layout::{Size, Align};
use syntax::ast::Mutability;

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct Allocation<Tag=(),Extra=()> {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer
    pub bytes: Vec<u8>,
    /// Maps from byte addresses to extra data for each pointer.
    /// Only the first byte of a pointer is inserted into the map; i.e.,
    /// every entry in this map applies to `pointer_size` consecutive bytes starting
    /// at the given offset.
    pub relocations: Relocations<Tag>,
    /// Denotes undefined memory. Reading from undefined memory is forbidden in miri
    pub undef_mask: UndefMask,
    /// The alignment of the allocation to detect unaligned reads.
    pub align: Align,
    /// Whether the allocation is mutable.
    /// Also used by codegen to determine if a static should be put into mutable memory,
    /// which happens for `static mut` and `static` with interior mutability.
    pub mutability: Mutability,
    /// Extra state for the machine.
    pub extra: Extra,
}

impl<Tag, Extra: Default> Allocation<Tag, Extra> {
    /// Creates a read-only allocation initialized by the given bytes
    pub fn from_bytes(slice: &[u8], align: Align) -> Self {
        let mut undef_mask = UndefMask::new(Size::ZERO);
        undef_mask.grow(Size::from_bytes(slice.len() as u64), true);
        Self {
            bytes: slice.to_owned(),
            relocations: Relocations::new(),
            undef_mask,
            align,
            mutability: Mutability::Immutable,
            extra: Extra::default(),
        }
    }

    pub fn from_byte_aligned_bytes(slice: &[u8]) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1, 1).unwrap())
    }

    pub fn undef(size: Size, align: Align) -> Self {
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        Allocation {
            bytes: vec![0; size.bytes() as usize],
            relocations: Relocations::new(),
            undef_mask: UndefMask::new(size),
            align,
            mutability: Mutability::Mutable,
            extra: Extra::default(),
        }
    }
}

impl<'tcx> ::serialize::UseSpecializedDecodable for &'tcx Allocation {}
