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
use common::*;
use builder::Builder;
use base;
use machine;
use monomorphize;
use type_::Type;
use type_of;

/// Given an enum, struct, closure, or tuple, extracts fields.
/// Treats closures as a struct with one variant.
/// `empty_if_no_variants` is a switch to deal with empty enums.
/// If true, `variant_index` is disregarded and an empty Vec returned in this case.
pub fn compute_fields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>,
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
        ty::TyClosure(def_id, substs) => {
            if variant_index > 0 { bug!("{} is a closure, which only has one variant", t);}
            substs.upvar_tys(def_id, cx.tcx()).collect()
        },
        _ => bug!("{} is not a type that can have fields.", t)
    }
}

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
            let (nonnull_variant_index, nonnull_variant, packed) = match *l {
                layout::Univariant { ref variant, .. } => (0, variant, variant.packed),
                layout::StructWrappedNullablePointer { nndiscr, ref nonnull, .. } =>
                    (nndiscr, nonnull, nonnull.packed),
                _ => unreachable!()
            };
            let fields = compute_fields(cx, t, nonnull_variant_index as usize, true);
            llty.set_struct_body(&struct_llfields(cx, &fields, nonnull_variant, false, false),
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
                    Type::struct_(cx, &struct_llfields(cx, &fields, nonnull, sizing, dst),
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
                    let fields = struct_llfields(cx, &fields, &variant, sizing, dst);
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
            let size = size.bytes();
            let align = align.abi();
            assert!(align <= std::u32::MAX as u64);
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
                             variant: &layout::Struct,
                             sizing: bool, dst: bool) -> Vec<Type> {
    let fields = variant.field_index_by_increasing_offset().map(|i| fields[i as usize]);
    if sizing {
        fields.filter(|ty| !dst || cx.shared().type_is_sized(*ty))
            .map(|ty| type_of::sizing_type_of(cx, ty)).collect()
    } else {
        fields.map(|ty| type_of::in_memory_type_of(cx, ty)).collect()
    }
}

pub fn is_discr_signed<'tcx>(l: &layout::Layout) -> bool {
    match *l {
        layout::CEnum { signed, .. }=> signed,
        _ => false,
    }
}

/// Obtain the actual discriminant of a value.
pub fn trans_get_discr<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    t: Ty<'tcx>,
    scrutinee: ValueRef,
    cast_to: Option<Type>,
    range_assert: bool
) -> ValueRef {
    let (def, substs) = match t.sty {
        ty::TyAdt(ref def, substs) if def.adt_kind() == AdtKind::Enum => (def, substs),
        _ => bug!("{} is not an enum", t)
    };

    debug!("trans_get_discr t: {:?}", t);
    let l = bcx.ccx.layout_of(t);

    let val = match *l {
        layout::CEnum { discr, min, max, .. } => {
            load_discr(bcx, discr, scrutinee, min, max, range_assert)
        }
        layout::General { discr, .. } => {
            let ptr = bcx.struct_gep(scrutinee, 0);
            load_discr(bcx, discr, ptr, 0, def.variants.len() as u64 - 1,
                       range_assert)
        }
        layout::Univariant { .. } | layout::UntaggedUnion { .. } => C_u8(bcx.ccx, 0),
        layout::RawNullablePointer { nndiscr, .. } => {
            let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
            let llptrty = type_of::sizing_type_of(bcx.ccx,
                monomorphize::field_ty(bcx.tcx(), substs,
                &def.variants[nndiscr as usize].fields[0]));
            bcx.icmp(cmp, bcx.load(scrutinee), C_null(llptrty))
        }
        layout::StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
            struct_wrapped_nullable_bitdiscr(bcx, nndiscr, discrfield, scrutinee)
        },
        _ => bug!("{} is not an enum", t)
    };
    match cast_to {
        None => val,
        Some(llty) => if is_discr_signed(&l) { bcx.sext(val, llty) } else { bcx.zext(val, llty) }
    }
}

fn struct_wrapped_nullable_bitdiscr(
    bcx: &Builder,
    nndiscr: u64,
    discrfield: &layout::FieldPath,
    scrutinee: ValueRef
) -> ValueRef {
    let llptrptr = bcx.gepi(scrutinee,
        &discrfield.iter().map(|f| *f as usize).collect::<Vec<_>>()[..]);
    let llptr = bcx.load(llptrptr);
    let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
    bcx.icmp(cmp, llptr, C_null(val_ty(llptr)))
}

/// Helper for cases where the discriminant is simply loaded.
fn load_discr(bcx: &Builder, ity: layout::Integer, ptr: ValueRef, min: u64, max: u64,
              range_assert: bool)
    -> ValueRef {
    let llty = Type::from_integer(bcx.ccx, ity);
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
        bcx.load(ptr)
    } else {
        // llvm::ConstantRange can deal with ranges that wrap around,
        // so an overflow on (max + 1) is fine.
        bcx.load_range_assert(ptr, min, max.wrapping_add(1), /* signed: */ True)
    }
}

/// Yield information about how to dispatch a case of the
/// discriminant-like value returned by `trans_switch`.
///
/// This should ideally be less tightly tied to `_match`.
pub fn trans_case<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, t: Ty<'tcx>, value: Disr) -> ValueRef {
    let l = bcx.ccx.layout_of(t);
    match *l {
        layout::CEnum { discr, .. }
        | layout::General { discr, .. }=> {
            C_integral(Type::from_integer(bcx.ccx, discr), value.0, true)
        }
        layout::RawNullablePointer { .. } |
        layout::StructWrappedNullablePointer { .. } => {
            assert!(value == Disr(0) || value == Disr(1));
            C_bool(bcx.ccx, value != Disr(0))
        }
        _ => {
            bug!("{} does not have a discriminant. Represented as {:#?}", t, l);
        }
    }
}

/// Set the discriminant for a new value of the given case of the given
/// representation.
pub fn trans_set_discr<'a, 'tcx>(bcx: &Builder<'a, 'tcx>, t: Ty<'tcx>, val: ValueRef, to: Disr) {
    let l = bcx.ccx.layout_of(t);
    match *l {
        layout::CEnum{ discr, min, max, .. } => {
            assert_discr_in_range(Disr(min), Disr(max), to);
            bcx.store(C_integral(Type::from_integer(bcx.ccx, discr), to.0, true),
                  val, None);
        }
        layout::General{ discr, .. } => {
            bcx.store(C_integral(Type::from_integer(bcx.ccx, discr), to.0, true),
                  bcx.struct_gep(val, 0), None);
        }
        layout::Univariant { .. }
        | layout::UntaggedUnion { .. }
        | layout::Vector { .. } => {
            assert_eq!(to, Disr(0));
        }
        layout::RawNullablePointer { nndiscr, .. } => {
            let nnty = compute_fields(bcx.ccx, t, nndiscr as usize, false)[0];
            if to.0 != nndiscr {
                let llptrty = type_of::sizing_type_of(bcx.ccx, nnty);
                bcx.store(C_null(llptrty), val, None);
            }
        }
        layout::StructWrappedNullablePointer { nndiscr, ref discrfield, ref nonnull, .. } => {
            if to.0 != nndiscr {
                if target_sets_discr_via_memset(bcx) {
                    // Issue #34427: As workaround for LLVM bug on
                    // ARM, use memset of 0 on whole struct rather
                    // than storing null to single target field.
                    let llptr = bcx.pointercast(val, Type::i8(bcx.ccx).ptr_to());
                    let fill_byte = C_u8(bcx.ccx, 0);
                    let size = C_uint(bcx.ccx, nonnull.stride().bytes());
                    let align = C_i32(bcx.ccx, nonnull.align.abi() as i32);
                    base::call_memset(bcx, llptr, fill_byte, size, align, false);
                } else {
                    let path = discrfield.iter().map(|&i| i as usize).collect::<Vec<_>>();
                    let llptrptr = bcx.gepi(val, &path[..]);
                    let llptrty = val_ty(llptrptr).element_type();
                    bcx.store(C_null(llptrty), llptrptr, None);
                }
            }
        }
        _ => bug!("Cannot handle {} represented as {:#?}", t, l)
    }
}

fn target_sets_discr_via_memset<'a, 'tcx>(bcx: &Builder<'a, 'tcx>) -> bool {
    bcx.sess().target.target.arch == "arm" || bcx.sess().target.target.arch == "aarch64"
}

pub fn assert_discr_in_range(min: Disr, max: Disr, discr: Disr) {
    if min <= max {
        assert!(min <= discr && discr <= max)
    } else {
        assert!(min <= discr || discr <= max)
    }
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
        layout::Univariant { ref variant, .. } => {
            const_struct_field(val, variant.memory_index[ix] as usize)
        }
        layout::Vector { .. } => const_struct_field(val, ix),
        layout::UntaggedUnion { .. } => const_struct_field(val, 0),
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
