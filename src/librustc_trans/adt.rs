// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Representation of Algebraic Data Types
//!
//! This module determines how to represent enums, structs, and tuples
//! based on their monomorphized types; it is responsible both for
//! choosing a representation and translating basic operations on
//! values of those types.  (Note: exporting the representations for
//! debuggers is handled in debuginfo.rs, not here.)
//!
//! Note that the interface treats everything as a general case of an
//! enum, so structs/tuples/etc. have one pseudo-variant with
//! discriminant 0; i.e., as if they were a univariant enum.
//!
//! Having everything in one place will enable improvements to data
//! structure representation; possibilities include:
//!
//! - User-specified alignment (e.g., cacheline-aligning parts of
//!   concurrently accessed data structures); LLVM can't represent this
//!   directly, so we'd have to insert padding fields in any structure
//!   that might contain one and adjust GEP indices accordingly.  See
//!   issue #4578.
//!
//! - Store nested enums' discriminants in the same word.  Rather, if
//!   some variants start with enums, and those enums representations
//!   have unused alignment padding between discriminant and body, the
//!   outer enum's discriminant can be stored there and those variants
//!   can start at offset 0.  Kind of fancy, and might need work to
//!   make copies of the inner enum type cooperate, but it could help
//!   with `Option` or `Result` wrapped around another enum.
//!
//! - Tagged pointers would be neat, but given that any type can be
//!   used unboxed and any field can have pointers (including mutable)
//!   taken to it, implementing them for Rust seems difficult.

use super::Disr;

use std;

use llvm::{ValueRef, True, IntEQ, IntNE};
use rustc::ty::layout;
use rustc::ty::{self, Ty, AdtKind};
use syntax::attr;
use build::*;
use common::*;
use debuginfo::DebugLoc;
use glue;
use base;
use machine;
use monomorphize;
use type_::Type;
use type_of;
use value::Value;

#[derive(Copy, Clone, PartialEq)]
pub enum BranchKind {
    Switch,
    Single
}

type Hint = attr::ReprAttr;

#[derive(Copy, Clone)]
pub struct MaybeSizedValue {
    pub value: ValueRef,
    pub meta: ValueRef,
}

impl MaybeSizedValue {
    pub fn sized(value: ValueRef) -> MaybeSizedValue {
        MaybeSizedValue {
            value: value,
            meta: std::ptr::null_mut()
        }
    }

    pub fn unsized_(value: ValueRef, meta: ValueRef) -> MaybeSizedValue {
        MaybeSizedValue {
            value: value,
            meta: meta
        }
    }

    pub fn has_meta(&self) -> bool {
        !self.meta.is_null()
    }
}

/// Given an enum, struct, closure, or tuple, extracts fields.
/// Treats closures as a struct with one variant.
/// `empty_if_no_variants` is a switch to deal with empty enums.
/// If true, `variant_index` is disregarded and an empty Vec returned in this case.
fn compute_fields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>,
                            variant_index: usize,
                            empty_if_no_variants: bool) -> Vec<Ty<'tcx>> {
    match t.sty {
        ty::TyAdt(ref def, _) if def.variants.len() == 0 && empty_if_no_variants => {
            Vec::default()
        },
        ty::TyAdt(ref def, ref substs) => {
            def.variants[variant_index].fields.iter().map(|f| {
                monomorphize::field_ty(cx.tcx(), substs, f)
            }).collect::<Vec<_>>()
        },
        ty::TyTuple(fields) => fields.to_vec(),
        ty::TyClosure(_, substs) => {
            if variant_index > 0 { bug!("{} is a closure, which only has one variant", t);}
            substs.upvar_tys.to_vec()
        },
        _ => bug!("{} is not a type that can have fields.", t)
    }
}

/// This represents the (GEP) indices to follow to get to the discriminant field
pub type DiscrField = Vec<usize>;

/// LLVM-level types are a little complicated.
///
/// C-like enums need to be actual ints, not wrapped in a struct,
/// because that changes the ABI on some platforms (see issue #10308).
///
/// For nominal types, in some cases, we need to use LLVM named structs
/// and fill in the actual contents in a second pass to prevent
/// unbounded recursion; see also the comments in `trans::type_of`.
pub fn type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    generic_type_of(cx, t, None, false, false)
}


// Pass dst=true if the type you are passing is a DST. Yes, we could figure
// this out, but if you call this on an unsized type without realising it, you
// are going to get the wrong type (it will not include the unsized parts of it).
pub fn sizing_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                t: Ty<'tcx>, dst: bool) -> Type {
    generic_type_of(cx, t, None, true, dst)
}

pub fn incomplete_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    t: Ty<'tcx>, name: &str) -> Type {
    generic_type_of(cx, t, Some(name), false, false)
}

pub fn finish_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                t: Ty<'tcx>, llty: &mut Type) {
    let l = cx.layout_of(t);
    debug!("finish_type_of: {} with layout {:#?}", t, l);
    match *l {
        layout::CEnum { .. } | layout::General { .. }
        | layout::UntaggedUnion { .. } | layout::RawNullablePointer { .. } => { }
        layout::Univariant { ..}
        | layout::StructWrappedNullablePointer { .. } => {
            let (nonnull_variant, packed) = match *l {
                layout::Univariant { ref variant, .. } => (0, variant.packed),
                layout::StructWrappedNullablePointer { nndiscr, ref nonnull, .. } =>
                    (nndiscr, nonnull.packed),
                _ => unreachable!()
            };
            let fields = compute_fields(cx, t, nonnull_variant as usize, true);
            llty.set_struct_body(&struct_llfields(cx, &fields, false, false),
                                 packed)
        },
        _ => bug!("This function cannot handle {} with layout {:#?}", t, l)
    }
}

fn generic_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                             t: Ty<'tcx>,
                             name: Option<&str>,
                             sizing: bool,
                             dst: bool) -> Type {
    let l = cx.layout_of(t);
    debug!("adt::generic_type_of t: {:?} name: {:?} sizing: {} dst: {}",
           t, name, sizing, dst);
    match *l {
        layout::CEnum { discr, .. } => Type::from_integer(cx, discr),
        layout::RawNullablePointer { nndiscr, .. } => {
            let (def, substs) = match t.sty {
                ty::TyAdt(d, s) => (d, s),
                _ => bug!("{} is not an ADT", t)
            };
            let nnty = monomorphize::field_ty(cx.tcx(), substs,
                &def.variants[nndiscr as usize].fields[0]);
            type_of::sizing_type_of(cx, nnty)
        }
        layout::StructWrappedNullablePointer { nndiscr, ref nonnull, .. } => {
            let fields = compute_fields(cx, t, nndiscr as usize, false);
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, &fields, sizing, dst),
                                  nonnull.packed)
                }
                Some(name) => {
                    assert_eq!(sizing, false);
                    Type::named_struct(cx, name)
                }
            }
        }
        layout::Univariant { ref variant, .. } => {
            // Note that this case also handles empty enums.
            // Thus the true as the final parameter here.
            let fields = compute_fields(cx, t, 0, true);
            match name {
                None => {
                    let fields = struct_llfields(cx, &fields, sizing, dst);
                    Type::struct_(cx, &fields, variant.packed)
                }
                Some(name) => {
                    // Hypothesis: named_struct's can never need a
                    // drop flag. (... needs validation.)
                    assert_eq!(sizing, false);
                    Type::named_struct(cx, name)
                }
            }
        }
        layout::Vector { element, count } => {
            let elem_ty = Type::from_primitive(cx, element);
            Type::vector(&elem_ty, count)
        }
        layout::UntaggedUnion { ref variants, .. }=> {
            // Use alignment-sized ints to fill all the union storage.
            let size = variants.stride().bytes();
            let align = variants.align.abi();
            let fill = union_fill(cx, size, align);
            match name {
                None => {
                    Type::struct_(cx, &[fill], variants.packed)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&[fill], variants.packed);
                    llty
                }
            }
        }
        layout::General { discr, size, align, .. } => {
            // We need a representation that has:
            // * The alignment of the most-aligned field
            // * The size of the largest variant (rounded up to that alignment)
            // * No alignment padding anywhere any variant has actual data
            //   (currently matters only for enums small enough to be immediate)
            // * The discriminant in an obvious place.
            //
            // So we start with the discriminant, pad it up to the alignment with
            // more of its own type, then use alignment-sized ints to get the rest
            // of the size.
            //
            // FIXME #10604: this breaks when vector types are present.
            let size = size.bytes();
            let align = align.abi();
            let discr_ty = Type::from_integer(cx, discr);
            let discr_size = discr.size().bytes();
            let padded_discr_size = roundup(discr_size, align as u32);
            let variant_part_size = size-padded_discr_size;
            let variant_fill = union_fill(cx, variant_part_size, align);

            assert_eq!(machine::llalign_of_min(cx, variant_fill), align as u32);
            assert_eq!(padded_discr_size % discr_size, 0); // Ensure discr_ty can fill pad evenly
            let fields: Vec<Type> =
                [discr_ty,
                 Type::array(&discr_ty, (padded_discr_size - discr_size)/discr_size),
                 variant_fill].iter().cloned().collect();
            match name {
                None => {
                    Type::struct_(cx, &fields[..], false)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&fields[..], false);
                    llty
                }
            }
        }
        _ => bug!("Unsupported type {} represented as {:#?}", t, l)
    }
}

fn union_fill(cx: &CrateContext, size: u64, align: u64) -> Type {
    assert_eq!(size%align, 0);
    assert_eq!(align.count_ones(), 1, "Alignment must be a power fof 2. Got {}", align);
    let align_units = size/align;
    let dl = &cx.tcx().data_layout;
    let layout_align = layout::Align::from_bytes(align, align).unwrap();
    if let Some(ity) = layout::Integer::for_abi_align(dl, layout_align) {
        Type::array(&Type::from_integer(cx, ity), align_units)
    } else {
        Type::array(&Type::vector(&Type::i32(cx), align/4),
                    align_units)
    }
}


fn struct_llfields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, fields: &Vec<Ty<'tcx>>,
                             sizing: bool, dst: bool) -> Vec<Type> {
    if sizing {
        fields.iter().filter(|&ty| !dst || type_is_sized(cx.tcx(), *ty))
            .map(|&ty| type_of::sizing_type_of(cx, ty)).collect()
    } else {
        fields.iter().map(|&ty| type_of::in_memory_type_of(cx, ty)).collect()
    }
}

/// Obtain a representation of the discriminant sufficient to translate
/// destructuring; this may or may not involve the actual discriminant.
pub fn trans_switch<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                t: Ty<'tcx>,
                                scrutinee: ValueRef,
                                range_assert: bool)
                                -> (BranchKind, Option<ValueRef>) {
    let l = bcx.ccx().layout_of(t);
    match *l {
        layout::CEnum { .. } | layout::General { .. } |
        layout::RawNullablePointer { .. } | layout::StructWrappedNullablePointer { .. } => {
            (BranchKind::Switch, Some(trans_get_discr(bcx, t, scrutinee, None, range_assert)))
        }
        layout::Univariant { .. } | layout::UntaggedUnion { .. } => {
            // N.B.: Univariant means <= 1 enum variants (*not* == 1 variants).
            (BranchKind::Single, None)
        },
        _ => bug!("{} is not an enum.", t)
    }
}

pub fn is_discr_signed<'tcx>(l: &layout::Layout) -> bool {
    match *l {
        layout::CEnum { signed, .. }=> signed,
        _ => false,
    }
}

/// Obtain the actual discriminant of a value.
pub fn trans_get_discr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>,
                                   scrutinee: ValueRef, cast_to: Option<Type>,
                                   range_assert: bool)
    -> ValueRef {
    let (def, substs) = match t.sty {
        ty::TyAdt(ref def, substs) if def.adt_kind() == AdtKind::Enum => (def, substs),
        _ => bug!("{} is not an enum", t)
    };

    debug!("trans_get_discr t: {:?}", t);
    let l = bcx.ccx().layout_of(t);

    let val = match *l {
        layout::CEnum { discr, min, max, .. } => {
            load_discr(bcx, discr, scrutinee, min, max, range_assert)
        }
        layout::General { discr, .. } => {
            let ptr = StructGEP(bcx, scrutinee, 0);
            load_discr(bcx, discr, ptr, 0, def.variants.len() as u64 - 1,
                       range_assert)
        }
        layout::Univariant { .. } | layout::UntaggedUnion { .. } => C_u8(bcx.ccx(), 0),
        layout::RawNullablePointer { nndiscr, .. } => {
            let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
            let llptrty = type_of::sizing_type_of(bcx.ccx(),
                monomorphize::field_ty(bcx.ccx().tcx(), substs,
                &def.variants[nndiscr as usize].fields[0]));
            ICmp(bcx, cmp, Load(bcx, scrutinee), C_null(llptrty), DebugLoc::None)
        }
        layout::StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
            struct_wrapped_nullable_bitdiscr(bcx, nndiscr, discrfield, scrutinee)
        },
        _ => bug!("{} is not an enum", t)
    };
    match cast_to {
        None => val,
        Some(llty) => if is_discr_signed(&l) { SExt(bcx, val, llty) } else { ZExt(bcx, val, llty) }
    }
}

fn struct_wrapped_nullable_bitdiscr(bcx: Block, nndiscr: u64, discrfield: &layout::FieldPath,
                                    scrutinee: ValueRef) -> ValueRef {
    let llptrptr = GEPi(bcx, scrutinee,
        &discrfield.iter().map(|f| *f as usize).collect::<Vec<_>>()[..]);
    let llptr = Load(bcx, llptrptr);
    let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
    ICmp(bcx, cmp, llptr, C_null(val_ty(llptr)), DebugLoc::None)
}

/// Helper for cases where the discriminant is simply loaded.
fn load_discr(bcx: Block, ity: layout::Integer, ptr: ValueRef, min: u64, max: u64,
              range_assert: bool)
    -> ValueRef {
    let llty = Type::from_integer(bcx.ccx(), ity);
    assert_eq!(val_ty(ptr), llty.ptr_to());
    let bits = ity.size().bits();
    assert!(bits <= 64);
    let bits = bits as usize;
    let mask = !0u64 >> (64 - bits);
    // For a (max) discr of -1, max will be `-1 as usize`, which overflows.
    // However, that is fine here (it would still represent the full range),
    if max.wrapping_add(1) & mask == min & mask || !range_assert {
        // i.e., if the range is everything.  The lo==hi case would be
        // rejected by the LLVM verifier (it would mean either an
        // empty set, which is impossible, or the entire range of the
        // type, which is pointless).
        Load(bcx, ptr)
    } else {
        // llvm::ConstantRange can deal with ranges that wrap around,
        // so an overflow on (max + 1) is fine.
        LoadRangeAssert(bcx, ptr, min, max.wrapping_add(1), /* signed: */ True)
    }
}

/// Yield information about how to dispatch a case of the
/// discriminant-like value returned by `trans_switch`.
///
/// This should ideally be less tightly tied to `_match`.
pub fn trans_case<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>, value: Disr)
                              -> ValueRef {
    let l = bcx.ccx().layout_of(t);
    match *l {
        layout::CEnum { discr, .. }
        | layout::General { discr, .. }=> {
            C_integral(Type::from_integer(bcx.ccx(), discr), value.0, true)
        }
        layout::RawNullablePointer { .. } |
        layout::StructWrappedNullablePointer { .. } => {
            assert!(value == Disr(0) || value == Disr(1));
            C_bool(bcx.ccx(), value != Disr(0))
        }
        _ => {
            bug!("{} does not have a discriminant. Represented as {:#?}", t, l);
        }
    }
}

/// Set the discriminant for a new value of the given case of the given
/// representation.
pub fn trans_set_discr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>,
                                   val: ValueRef, to: Disr) {
    let l = bcx.ccx().layout_of(t);
    match *l {
        layout::CEnum{ discr, min, max, .. } => {
            assert_discr_in_range(Disr(min), Disr(max), to);
            Store(bcx, C_integral(Type::from_integer(bcx.ccx(), discr), to.0, true),
                  val);
        }
        layout::General{ discr, .. } => {
            Store(bcx, C_integral(Type::from_integer(bcx.ccx(), discr), to.0, true),
                  StructGEP(bcx, val, 0));
        }
        layout::Univariant { .. }
        | layout::UntaggedUnion { .. }
        | layout::Vector { .. } => {
            assert_eq!(to, Disr(0));
        }
        layout::RawNullablePointer { nndiscr, .. } => {
            let nnty = compute_fields(bcx.ccx(), t, nndiscr as usize, false)[0];
            if to.0 != nndiscr {
                let llptrty = type_of::sizing_type_of(bcx.ccx(), nnty);
                Store(bcx, C_null(llptrty), val);
            }
        }
        layout::StructWrappedNullablePointer { nndiscr, ref discrfield, ref nonnull, .. } => {
            if to.0 != nndiscr {
                if target_sets_discr_via_memset(bcx) {
                    // Issue #34427: As workaround for LLVM bug on
                    // ARM, use memset of 0 on whole struct rather
                    // than storing null to single target field.
                    let b = B(bcx);
                    let llptr = b.pointercast(val, Type::i8(b.ccx).ptr_to());
                    let fill_byte = C_u8(b.ccx, 0);
                    let size = C_uint(b.ccx, nonnull.stride().bytes());
                    let align = C_i32(b.ccx, nonnull.align.abi() as i32);
                    base::call_memset(&b, llptr, fill_byte, size, align, false);
                } else {
                    let path = discrfield.iter().map(|&i| i as usize).collect::<Vec<_>>();
                    let llptrptr = GEPi(bcx, val, &path[..]);
                    let llptrty = val_ty(llptrptr).element_type();
                    Store(bcx, C_null(llptrty), llptrptr);
                }
            }
        }
        _ => bug!("Cannot handle {} represented as {:#?}", t, l)
    }
}

fn target_sets_discr_via_memset<'blk, 'tcx>(bcx: Block<'blk, 'tcx>) -> bool {
    bcx.sess().target.target.arch == "arm" || bcx.sess().target.target.arch == "aarch64"
}

fn assert_discr_in_range(min: Disr, max: Disr, discr: Disr) {
    if min <= max {
        assert!(min <= discr && discr <= max)
    } else {
        assert!(min <= discr || discr <= max)
    }
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>,
                                   val: MaybeSizedValue, discr: Disr, ix: usize) -> ValueRef {
    trans_field_ptr_builder(&bcx.build(), t, val, discr, ix)
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr_builder<'blk, 'tcx>(bcx: &BlockAndBuilder<'blk, 'tcx>,
                                           t: Ty<'tcx>,
                                           val: MaybeSizedValue,
                                           discr: Disr, ix: usize)
                                           -> ValueRef {
    let l = bcx.ccx().layout_of(t);
    debug!("trans_field_ptr_builder on {} represented as {:#?}", t, l);
    // Note: if this ever needs to generate conditionals (e.g., if we
    // decide to do some kind of cdr-coding-like non-unique repr
    // someday), it will need to return a possibly-new bcx as well.
    match *l {
        layout::Univariant { ref variant, .. } => {
            assert_eq!(discr, Disr(0));
            struct_field_ptr(bcx, &variant,
             &compute_fields(bcx.ccx(), t, 0, false),
             val, ix, false)
        }
        layout::Vector { count, .. } => {
            assert_eq!(discr.0, 0);
            assert!((ix as u64) < count);
            bcx.struct_gep(val.value, ix)
        }
        layout::General { discr: d, ref variants, .. } => {
            let mut fields = compute_fields(bcx.ccx(), t, discr.0 as usize, false);
            fields.insert(0, d.to_ty(&bcx.ccx().tcx(), false));
            struct_field_ptr(bcx, &variants[discr.0 as usize],
             &fields,
             val, ix + 1, true)
        }
        layout::UntaggedUnion { .. } => {
            let fields = compute_fields(bcx.ccx(), t, 0, false);
            let ty = type_of::in_memory_type_of(bcx.ccx(), fields[ix]);
            if bcx.is_unreachable() { return C_undef(ty.ptr_to()); }
            bcx.pointercast(val.value, ty.ptr_to())
        }
        layout::RawNullablePointer { nndiscr, .. } |
        layout::StructWrappedNullablePointer { nndiscr,  .. } if discr.0 != nndiscr => {
            let nullfields = compute_fields(bcx.ccx(), t, (1-nndiscr) as usize, false);
            // The unit-like case might have a nonzero number of unit-like fields.
            // (e.d., Result of Either with (), as one side.)
            let ty = type_of::type_of(bcx.ccx(), nullfields[ix]);
            assert_eq!(machine::llsize_of_alloc(bcx.ccx(), ty), 0);
            // The contents of memory at this pointer can't matter, but use
            // the value that's "reasonable" in case of pointer comparison.
            if bcx.is_unreachable() { return C_undef(ty.ptr_to()); }
            bcx.pointercast(val.value, ty.ptr_to())
        }
        layout::RawNullablePointer { nndiscr, .. } => {
            let nnty = compute_fields(bcx.ccx(), t, nndiscr as usize, false)[0];
            assert_eq!(ix, 0);
            assert_eq!(discr.0, nndiscr);
            let ty = type_of::type_of(bcx.ccx(), nnty);
            if bcx.is_unreachable() { return C_undef(ty.ptr_to()); }
            bcx.pointercast(val.value, ty.ptr_to())
        }
        layout::StructWrappedNullablePointer { ref nonnull, nndiscr, .. } => {
            assert_eq!(discr.0, nndiscr);
            struct_field_ptr(bcx, &nonnull,
             &compute_fields(bcx.ccx(), t, discr.0 as usize, false),
             val, ix, false)
        }
        _ => bug!("element access in type without elements: {} represented as {:#?}", t, l)
    }
}

fn struct_field_ptr<'blk, 'tcx>(bcx: &BlockAndBuilder<'blk, 'tcx>,
                                st: &layout::Struct, fields: &Vec<Ty<'tcx>>, val: MaybeSizedValue,
                                ix: usize, needs_cast: bool) -> ValueRef {
    let ccx = bcx.ccx();
    let fty = fields[ix];
    let ll_fty = type_of::in_memory_type_of(bcx.ccx(), fty);
    if bcx.is_unreachable() {
        return C_undef(ll_fty.ptr_to());
    }

    let ptr_val = if needs_cast {
        let fields = fields.iter().map(|&ty| {
            type_of::in_memory_type_of(ccx, ty)
        }).collect::<Vec<_>>();
        let real_ty = Type::struct_(ccx, &fields[..], st.packed);
        bcx.pointercast(val.value, real_ty.ptr_to())
    } else {
        val.value
    };

    // Simple case - we can just GEP the field
    //   * First field - Always aligned properly
    //   * Packed struct - There is no alignment padding
    //   * Field is sized - pointer is properly aligned already
    if ix == 0 || st.packed || type_is_sized(bcx.tcx(), fty) {
        return bcx.struct_gep(ptr_val, ix);
    }

    // If the type of the last field is [T] or str, then we don't need to do
    // any adjusments
    match fty.sty {
        ty::TySlice(..) | ty::TyStr => {
            return bcx.struct_gep(ptr_val, ix);
        }
        _ => ()
    }

    // There's no metadata available, log the case and just do the GEP.
    if !val.has_meta() {
        debug!("Unsized field `{}`, of `{:?}` has no metadata for adjustment",
               ix, Value(ptr_val));
        return bcx.struct_gep(ptr_val, ix);
    }

    let dbloc = DebugLoc::None;

    // We need to get the pointer manually now.
    // We do this by casting to a *i8, then offsetting it by the appropriate amount.
    // We do this instead of, say, simply adjusting the pointer from the result of a GEP
    // because the field may have an arbitrary alignment in the LLVM representation
    // anyway.
    //
    // To demonstrate:
    //   struct Foo<T: ?Sized> {
    //      x: u16,
    //      y: T
    //   }
    //
    // The type Foo<Foo<Trait>> is represented in LLVM as { u16, { u16, u8 }}, meaning that
    // the `y` field has 16-bit alignment.

    let meta = val.meta;


    let offset = st.offsets[ix].bytes();
    let unaligned_offset = C_uint(bcx.ccx(), offset);

    // Get the alignment of the field
    let (_, align) = glue::size_and_align_of_dst(bcx, fty, meta);

    // Bump the unaligned offset up to the appropriate alignment using the
    // following expression:
    //
    //   (unaligned offset + (align - 1)) & -align

    // Calculate offset
    dbloc.apply(bcx.fcx());
    let align_sub_1 = bcx.sub(align, C_uint(bcx.ccx(), 1u64));
    let offset = bcx.and(bcx.add(unaligned_offset, align_sub_1),
                         bcx.neg(align));

    debug!("struct_field_ptr: DST field offset: {:?}", Value(offset));

    // Cast and adjust pointer
    let byte_ptr = bcx.pointercast(ptr_val, Type::i8p(bcx.ccx()));
    let byte_ptr = bcx.gep(byte_ptr, &[offset]);

    // Finally, cast back to the type expected
    let ll_fty = type_of::in_memory_type_of(bcx.ccx(), fty);
    debug!("struct_field_ptr: Field type is {:?}", ll_fty);
    bcx.pointercast(byte_ptr, ll_fty.ptr_to())
}

/// Construct a constant value, suitable for initializing a
/// GlobalVariable, given a case and constant values for its fields.
/// Note that this may have a different LLVM type (and different
/// alignment!) from the representation's `type_of`, so it needs a
/// pointer cast before use.
///
/// The LLVM type system does not directly support unions, and only
/// pointers can be bitcast, so a constant (and, by extension, the
/// GlobalVariable initialized by it) will have a type that can vary
/// depending on which case of an enum it is.
///
/// To understand the alignment situation, consider `enum E { V64(u64),
/// V32(u32, u32) }` on Windows.  The type has 8-byte alignment to
/// accommodate the u64, but `V32(x, y)` would have LLVM type `{i32,
/// i32, i32}`, which is 4-byte aligned.
///
/// Currently the returned value has the same size as the type, but
/// this could be changed in the future to avoid allocating unnecessary
/// space after values of shorter-than-maximum cases.
pub fn trans_const<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>, discr: Disr,
                             vals: &[ValueRef]) -> ValueRef {
    let l = ccx.layout_of(t);
    let dl = &ccx.tcx().data_layout;
    match *l {
        layout::CEnum { discr: d, min, max, .. } => {
            assert_eq!(vals.len(), 0);
            assert_discr_in_range(Disr(min), Disr(max), discr);
            C_integral(Type::from_integer(ccx, d), discr.0, true)
        }
        layout::General { discr: d, ref variants, .. } => {
            let variant = &variants[discr.0 as usize];
            let lldiscr = C_integral(Type::from_integer(ccx, d), discr.0 as u64, true);
            let mut vals_with_discr = vec![lldiscr];
            vals_with_discr.extend_from_slice(vals);
            let mut contents = build_const_struct(ccx, &variant,
                &vals_with_discr[..]);
            let needed_padding = l.size(dl).bytes() - variant.min_size.bytes();
            if needed_padding > 0 {
                contents.push(padding(ccx, needed_padding));
            }
            C_struct(ccx, &contents[..], false)
        }
        layout::UntaggedUnion { ref variants, .. }=> {
            assert_eq!(discr, Disr(0));
            let contents = build_const_union(ccx, variants, vals[0]);
            C_struct(ccx, &contents, variants.packed)
        }
        layout::Univariant { ref variant, .. } => {
            assert_eq!(discr, Disr(0));
            let contents = build_const_struct(ccx,
                &variant, vals);
            C_struct(ccx, &contents[..], variant.packed)
        }
        layout::Vector { .. } => {
            C_vector(vals)
        }
        layout::RawNullablePointer { nndiscr, .. } => {
            let nnty = compute_fields(ccx, t, nndiscr as usize, false)[0];
            if discr.0 == nndiscr {
                assert_eq!(vals.len(), 1);
                vals[0]
            } else {
                C_null(type_of::sizing_type_of(ccx, nnty))
            }
        }
        layout::StructWrappedNullablePointer { ref nonnull, nndiscr, .. } => {
            if discr.0 == nndiscr {
                C_struct(ccx, &build_const_struct(ccx, &nonnull, vals),
                         false)
            } else {
                let fields = compute_fields(ccx, t, nndiscr as usize, false);
                let vals = fields.iter().map(|&ty| {
                    // Always use null even if it's not the `discrfield`th
                    // field; see #8506.
                    C_null(type_of::sizing_type_of(ccx, ty))
                }).collect::<Vec<ValueRef>>();
                C_struct(ccx, &build_const_struct(ccx, &nonnull, &vals[..]),
                         false)
            }
        }
        _ => bug!("trans_const: cannot handle type {} repreented as {:#?}", t, l)
    }
}

/// Building structs is a little complicated, because we might need to
/// insert padding if a field's value is less aligned than its type.
///
/// Continuing the example from `trans_const`, a value of type `(u32,
/// E)` should have the `E` at offset 8, but if that field's
/// initializer is 4-byte aligned then simply translating the tuple as
/// a two-element struct will locate it at offset 4, and accesses to it
/// will read the wrong memory.
fn build_const_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                st: &layout::Struct,
                                vals: &[ValueRef])
                                -> Vec<ValueRef> {
    assert_eq!(vals.len(), st.offsets.len());

    if vals.len() == 0 {
        return Vec::new();
    }

    // offset of current value
    let mut offset = 0;
    let mut cfields = Vec::new();
    let offsets = st.offsets.iter().map(|i| i.bytes());
    for (&val, target_offset) in vals.iter().zip(offsets) {
        if offset < target_offset {
            cfields.push(padding(ccx, target_offset - offset));
            offset = target_offset;
        }
        assert!(!is_undef(val));
        cfields.push(val);
        offset += machine::llsize_of_alloc(ccx, val_ty(val));
    }

    if offset < st.stride().bytes() {
        cfields.push(padding(ccx, st.stride().bytes() - offset));
    }

    cfields
}

fn build_const_union<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               un: &layout::Union,
                               field_val: ValueRef)
                               -> Vec<ValueRef> {
    let mut cfields = vec![field_val];

    let offset = machine::llsize_of_alloc(ccx, val_ty(field_val));
    let size = un.stride().bytes();
    if offset != size {
        cfields.push(padding(ccx, size - offset));
    }

    cfields
}

fn padding(ccx: &CrateContext, size: u64) -> ValueRef {
    C_undef(Type::array(&Type::i8(ccx), size))
}

// FIXME this utility routine should be somewhere more general
#[inline]
fn roundup(x: u64, a: u32) -> u64 { let a = a as u64; ((x + (a - 1)) / a) * a }

/// Extract a field of a constant value, as appropriate for its
/// representation.
///
/// (Not to be confused with `common::const_get_elt`, which operates on
/// raw LLVM-level structs and arrays.)
pub fn const_get_field<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>,
                       val: ValueRef, _discr: Disr,
                       ix: usize) -> ValueRef {
    let l = ccx.layout_of(t);
    match *l {
        layout::CEnum { .. } => bug!("element access in C-like enum const"),
        layout::Univariant { .. } | layout::Vector { .. } => const_struct_field(val, ix),
        layout::UntaggedUnion { .. } => const_struct_field(val, 0),
        layout::General { .. } => const_struct_field(val, ix + 1),
        layout::RawNullablePointer { .. } => {
            assert_eq!(ix, 0);
            val
        },
        layout::StructWrappedNullablePointer{ .. } => const_struct_field(val, ix),
        _ => bug!("{} does not have fields.", t)
    }
}

/// Extract field of struct-like const, skipping our alignment padding.
fn const_struct_field(val: ValueRef, ix: usize) -> ValueRef {
    // Get the ix-th non-undef element of the struct.
    let mut real_ix = 0; // actual position in the struct
    let mut ix = ix; // logical index relative to real_ix
    let mut field;
    loop {
        loop {
            field = const_get_elt(val, &[real_ix]);
            if !is_undef(field) {
                break;
            }
            real_ix = real_ix + 1;
        }
        if ix == 0 {
            return field;
        }
        ix = ix - 1;
        real_ix = real_ix + 1;
    }
}
