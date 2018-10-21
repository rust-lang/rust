// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_lints)]

use ty::layout::{HasDataLayout, Align, Size, TyLayout};
use ty::subst::Substs;
use ty;
use hir::def_id::DefId;

use super::{EvalResult, Pointer, PointerArithmetic, Allocation, AllocId, sign_extend, truncate};

/// Represents a constant value in Rust. Scalar and ScalarPair are optimizations which
/// matches the LocalValue optimizations for easy conversions between Value and ConstValue.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash)]
pub enum ConstValue<'tcx> {
    /// Never returned from the `const_eval` query, but the HIR contains these frequently in order
    /// to allow HIR creation to happen for everything before needing to be able to run constant
    /// evaluation
    Unevaluated(DefId, &'tcx Substs<'tcx>),

    /// An allocation + offset into the allocation.
    /// Invariant: The AllocId matches the allocation.
    ByRef(AllocId, &'tcx Allocation, Size),
}

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn try_as_by_ref(&self) -> Option<(AllocId, &'tcx Allocation, Size)> {
        match self {
            ConstValue::Unevaluated(..) => None,
            ConstValue::ByRef(a, b, c) => Some((*a, *b, *c)),
        }
    }

    #[inline]
    /// if this is ByRef, return the same thing but with the offset increased by `n`
    pub fn try_offset(&self, n: Size) -> Option<Self> {
        let (id, alloc, offset) = self.try_as_by_ref()?;
        Some(ConstValue::ByRef(id, alloc, offset + n))
    }

    #[inline]
    pub fn try_get_bytes(&self, hdl: impl HasDataLayout, n: Size, align: Align) -> Option<&[u8]> {
        let (_, alloc, offset) = self.try_as_by_ref()?;
        alloc.get_bytes(hdl, offset, n, align).ok()
    }

    #[inline]
    pub fn try_to_bits(&self, hdl: impl HasDataLayout, layout: TyLayout<'tcx>) -> Option<u128> {
        let bytes = self.try_get_bytes(hdl, layout.size, layout.align)?;
        let endian = hdl.data_layout().endian;
        super::read_target_uint(endian, &bytes).ok()
    }

    #[inline]
    pub fn try_to_usize(&self, hdl: impl HasDataLayout) -> Option<u128> {
        let size = hdl.data_layout().pointer_size;
        let align = hdl.data_layout().pointer_align;
        let bytes = self.try_get_bytes(hdl, size, align)?;
        let endian = hdl.data_layout().endian;
        super::read_target_uint(endian, &bytes).ok()
    }

    #[inline]
    pub fn try_to_ptr(
        &self,
        hdl: impl HasDataLayout,
    ) -> Option<Pointer> {
        let (_, alloc, offset) = self.try_as_by_ref()?;
        let size = hdl.data_layout().pointer_size;
        let required_align = hdl.data_layout().pointer_align;
        alloc.read_scalar(hdl, offset, size, required_align).ok()?.to_ptr().ok()
    }

    /// e.g. for vtables, fat pointers or single pointers
    #[inline]
    pub fn new_pointer_list(
        list: &[Scalar],
        tcx: ty::TyCtxt<'_, '_, 'tcx>,
    ) -> Self {
        let ps = tcx.data_layout().pointer_size;
        let mut alloc = Allocation::undef(
            ps * list.len() as u64,
            tcx.data_layout().pointer_align,
        );
        alloc.undef_mask.set_range_inbounds(Size::ZERO, ps * list.len() as u64, true);
        for (i, s) in list.iter().enumerate() {
            let (int, ptr) = match s {
                Scalar::Bits { bits, size } => {
                    assert!(*size as u64 == ps.bytes());
                    (*bits as u64, None)
                }
                Scalar::Ptr(ptr) => (ptr.offset.bytes(), Some(ptr)),
            };
            let i = i * ps.bytes() as usize;
            let j = i + ps.bytes() as usize;
            super::write_target_uint(
                tcx.data_layout().endian,
                &mut alloc.bytes[i..j],
                int.into(),
            ).unwrap();
            if let Some(ptr) = ptr {
                alloc.relocations.insert(
                    ps * i as u64,
                    (ptr.tag, ptr.alloc_id),
                );
            }
        }
        Self::from_allocation(tcx, alloc)
    }

    #[inline]
    pub fn from_allocation(
        tcx: ty::TyCtxt<'_, '_, 'tcx>,
        alloc: Allocation,
    ) -> Self {
        let alloc = tcx.intern_const_alloc(alloc);
        let alloc_id = tcx.alloc_map.lock().allocate(alloc);
        ConstValue::ByRef(alloc_id, alloc, Size::ZERO)
    }

    #[inline]
    pub fn new_slice(
        val: Scalar,
        len: u64,
        tcx: ty::TyCtxt<'_, '_, 'tcx>,
    ) -> Self {
        Self::new_pointer_list(
            &[
                val,
                Scalar::Bits {
                    bits: len as u128,
                    size: tcx.data_layout.pointer_size.bytes() as u8,
                },
            ],
            tcx,
        )
    }

    #[inline]
    pub fn new_dyn_trait(
        val: Scalar,
        vtable: Pointer,
        tcx: ty::TyCtxt<'_, '_, 'tcx>,
    ) -> Self {
        Self::new_pointer_list(&[val, vtable.into()], tcx)
    }
}

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of a `Allocation`, up to 8 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum Scalar<Tag=(), Id=AllocId> {
    /// The raw bytes of a simple value.
    Bits {
        /// The first `size` bytes are the value.
        /// Do not try to read less or more bytes that that. The remaining bytes must be 0.
        size: u8,
        bits: u128,
    },

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer<Tag, Id>),
}

impl<'tcx> Scalar<()> {
    #[inline]
    pub fn with_default_tag<Tag>(self) -> Scalar<Tag>
        where Tag: Default
    {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.with_default_tag()),
            Scalar::Bits { bits, size } => Scalar::Bits { bits, size },
        }
    }
}

impl<'tcx, Tag> Scalar<Tag> {
    #[inline]
    pub fn erase_tag(self) -> Scalar {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.erase_tag()),
            Scalar::Bits { bits, size } => Scalar::Bits { bits, size },
        }
    }

    #[inline]
    pub fn ptr_null(cx: impl HasDataLayout) -> Self {
        Scalar::Bits {
            bits: 0,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }
    }

    #[inline]
    pub fn zst() -> Self {
        Scalar::Bits { bits: 0, size: 0 }
    }

    #[inline]
    pub fn ptr_signed_offset(self, i: i64, cx: impl HasDataLayout) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, layout.pointer_size.bytes());
                Ok(Scalar::Bits {
                    bits: layout.signed_offset(bits as u64, i)? as u128,
                    size,
                })
            }
            Scalar::Ptr(ptr) => ptr.signed_offset(i, layout).map(Scalar::Ptr),
        }
    }

    #[inline]
    pub fn ptr_offset(self, i: Size, cx: impl HasDataLayout) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, layout.pointer_size.bytes());
                Ok(Scalar::Bits {
                    bits: layout.offset(bits as u64, i.bytes())? as u128,
                    size,
                })
            }
            Scalar::Ptr(ptr) => ptr.offset(i, layout).map(Scalar::Ptr),
        }
    }

    #[inline]
    pub fn ptr_wrapping_signed_offset(self, i: i64, cx: impl HasDataLayout) -> Self {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, layout.pointer_size.bytes());
                Scalar::Bits {
                    bits: layout.wrapping_signed_offset(bits as u64, i) as u128,
                    size,
                }
            }
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.wrapping_signed_offset(i, layout)),
        }
    }

    #[inline]
    pub fn is_null_ptr(self, cx: impl HasDataLayout) -> bool {
        match self {
            Scalar::Bits { bits, size } =>  {
                assert_eq!(size as u64, cx.data_layout().pointer_size.bytes());
                bits == 0
            },
            Scalar::Ptr(_) => false,
        }
    }

    #[inline]
    pub fn is_null(self) -> bool {
        match self {
            Scalar::Bits { bits, .. } => bits == 0,
            Scalar::Ptr(_) => false
        }
    }

    #[inline]
    pub fn from_bool(b: bool) -> Self {
        Scalar::Bits { bits: b as u128, size: 1 }
    }

    #[inline]
    pub fn from_char(c: char) -> Self {
        Scalar::Bits { bits: c as u128, size: 4 }
    }

    #[inline]
    pub fn from_uint(i: impl Into<u128>, size: Size) -> Self {
        let i = i.into();
        debug_assert_eq!(truncate(i, size), i,
                         "Unsigned value {} does not fit in {} bits", i, size.bits());
        Scalar::Bits { bits: i, size: size.bytes() as u8 }
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, size: Size) -> Self {
        let i = i.into();
        // `into` performed sign extension, we have to truncate
        let truncated = truncate(i as u128, size);
        debug_assert_eq!(sign_extend(truncated, size) as i128, i,
                         "Signed value {} does not fit in {} bits", i, size.bits());
        Scalar::Bits { bits: truncated, size: size.bytes() as u8 }
    }

    #[inline]
    pub fn from_f32(f: f32) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, size: 4 }
    }

    #[inline]
    pub fn from_f64(f: f64) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, size: 8 }
    }

    #[inline]
    pub fn to_bits(self, target_size: Size) -> EvalResult<'tcx, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(target_size.bytes(), size as u64);
                assert_ne!(size, 0, "to_bits cannot be used with zsts");
                Ok(bits)
            }
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }

    #[inline]
    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer<Tag>> {
        match self {
            Scalar::Bits { bits: 0, .. } => err!(InvalidNullPointerUsage),
            Scalar::Bits { .. } => err!(ReadBytesAsPointer),
            Scalar::Ptr(p) => Ok(p),
        }
    }

    #[inline]
    pub fn is_bits(self) -> bool {
        match self {
            Scalar::Bits { .. } => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_ptr(self) -> bool {
        match self {
            Scalar::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        match self {
            Scalar::Bits { bits: 0, size: 1 } => Ok(false),
            Scalar::Bits { bits: 1, size: 1 } => Ok(true),
            _ => err!(InvalidBool),
        }
    }

    pub fn to_char(self) -> EvalResult<'tcx, char> {
        let val = self.to_u32()?;
        match ::std::char::from_u32(val) {
            Some(c) => Ok(c),
            None => err!(InvalidChar(val as u128)),
        }
    }

    pub fn to_u8(self) -> EvalResult<'static, u8> {
        let sz = Size::from_bits(8);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u8 as u128, b);
        Ok(b as u8)
    }

    pub fn to_u32(self) -> EvalResult<'static, u32> {
        let sz = Size::from_bits(32);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u32 as u128, b);
        Ok(b as u32)
    }

    pub fn to_u64(self) -> EvalResult<'static, u64> {
        let sz = Size::from_bits(64);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u64 as u128, b);
        Ok(b as u64)
    }

    pub fn to_usize(self, cx: impl HasDataLayout) -> EvalResult<'static, u64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        assert_eq!(b as u64 as u128, b);
        Ok(b as u64)
    }

    pub fn to_i8(self) -> EvalResult<'static, i8> {
        let sz = Size::from_bits(8);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i8 as i128, b);
        Ok(b as i8)
    }

    pub fn to_i32(self) -> EvalResult<'static, i32> {
        let sz = Size::from_bits(32);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i32 as i128, b);
        Ok(b as i32)
    }

    pub fn to_i64(self) -> EvalResult<'static, i64> {
        let sz = Size::from_bits(64);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i64 as i128, b);
        Ok(b as i64)
    }

    pub fn to_isize(self, cx: impl HasDataLayout) -> EvalResult<'static, i64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        let b = sign_extend(b, cx.data_layout().pointer_size) as i128;
        assert_eq!(b as i64 as i128, b);
        Ok(b as i64)
    }

    #[inline]
    pub fn to_f32(self) -> EvalResult<'static, f32> {
        Ok(f32::from_bits(self.to_u32()?))
    }

    #[inline]
    pub fn to_f64(self) -> EvalResult<'static, f64> {
        Ok(f64::from_bits(self.to_u64()?))
    }
}

impl<Tag> From<Pointer<Tag>> for Scalar<Tag> {
    #[inline(always)]
    fn from(ptr: Pointer<Tag>) -> Self {
        Scalar::Ptr(ptr)
    }
}
