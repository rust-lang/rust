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

use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, HasDataLayout, LayoutTyper, Size};

use context::CrateContext;
use monomorphize;
use type_::Type;
use type_of;

/// LLVM-level types are a little complicated.
///
/// C-like enums need to be actual ints, not wrapped in a struct,
/// because that changes the ABI on some platforms (see issue #10308).
///
/// For nominal types, in some cases, we need to use LLVM named structs
/// and fill in the actual contents in a second pass to prevent
/// unbounded recursion; see also the comments in `trans::type_of`.
pub fn type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    generic_type_of(cx, t, None)
}

pub fn incomplete_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    t: Ty<'tcx>, name: &str) -> Type {
    generic_type_of(cx, t, Some(name))
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
            llty.set_struct_body(&struct_llfields(cx, t, nonnull_variant_index as usize,
                                                  nonnull_variant, None),
                                 packed)
        },
        _ => bug!("This function cannot handle {} with layout {:#?}", t, l)
    }
}

fn generic_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                             t: Ty<'tcx>,
                             name: Option<&str>) -> Type {
    let l = cx.layout_of(t);
    debug!("adt::generic_type_of t: {:?} name: {:?}", t, name);
    match *l {
        layout::CEnum { discr, .. } => Type::from_integer(cx, discr),
        layout::RawNullablePointer { nndiscr, .. } => {
            let (def, substs) = match t.sty {
                ty::TyAdt(d, s) => (d, s),
                _ => bug!("{} is not an ADT", t)
            };
            let nnty = monomorphize::field_ty(cx.tcx(), substs,
                &def.variants[nndiscr as usize].fields[0]);
            if let layout::Scalar { value: layout::Pointer, .. } = *cx.layout_of(nnty) {
                Type::i8p(cx)
            } else {
                type_of::type_of(cx, nnty)
            }
        }
        layout::StructWrappedNullablePointer { nndiscr, ref nonnull, .. } => {
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, t, nndiscr as usize, nonnull, None),
                                  nonnull.packed)
                }
                Some(name) => {
                    Type::named_struct(cx, name)
                }
            }
        }
        layout::Univariant { ref variant, .. } => {
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, t, 0, &variant, None),
                                  variant.packed)
                }
                Some(name) => {
                    Type::named_struct(cx, name)
                }
            }
        }
        layout::UntaggedUnion { ref variants, .. }=> {
            // Use alignment-sized ints to fill all the union storage.
            let fill = union_fill(cx, variants.stride(), variants.align);
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
        layout::General { discr, size, align, primitive_align, .. } => {
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
            let discr_ty = Type::from_integer(cx, discr);
            let discr_size = discr.size().bytes();
            let padded_discr_size = discr.size().abi_align(align);
            let variant_part_size = size - padded_discr_size;

            // Ensure discr_ty can fill pad evenly
            assert_eq!(padded_discr_size.bytes() % discr_size, 0);
            let fields = [
                discr_ty,
                Type::array(&discr_ty, padded_discr_size.bytes() / discr_size - 1),
                union_fill(cx, variant_part_size, primitive_align)
            ];
            match name {
                None => {
                    Type::struct_(cx, &fields, false)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&fields, false);
                    llty
                }
            }
        }
        _ => bug!("Unsupported type {} represented as {:#?}", t, l)
    }
}

fn union_fill(cx: &CrateContext, size: Size, align: Align) -> Type {
    let abi_align = align.abi();
    let elem_ty = if let Some(ity) = layout::Integer::for_abi_align(cx, align) {
        Type::from_integer(cx, ity)
    } else {
        let vec_align = cx.data_layout().vector_align(Size::from_bytes(abi_align));
        assert_eq!(vec_align.abi(), abi_align);
        Type::vector(&Type::i32(cx), abi_align / 4)
    };

    let size = size.bytes();
    assert_eq!(size % abi_align, 0);
    Type::array(&elem_ty, size / abi_align)
}

// Lookup `Struct::memory_index` and double it to account for padding
pub fn struct_llfields_index(variant: &layout::Struct, index: usize) -> usize {
    (variant.memory_index[index] as usize) << 1
}

pub fn struct_llfields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                 t: Ty<'tcx>,
                                 variant_index: usize,
                                 variant: &layout::Struct,
                                 discr: Option<Ty<'tcx>>) -> Vec<Type> {
    let field_count = match t.sty {
        ty::TyAdt(ref def, _) if def.variants.len() == 0 => return vec![],
        ty::TyAdt(ref def, _) => {
            discr.is_some() as usize + def.variants[variant_index].fields.len()
        },
        ty::TyTuple(fields, _) => fields.len(),
        ty::TyClosure(def_id, substs) => {
            if variant_index > 0 { bug!("{} is a closure, which only has one variant", t);}
            substs.upvar_tys(def_id, cx.tcx()).count()
        },
        ty::TyGenerator(def_id, substs, _) => {
            if variant_index > 0 { bug!("{} is a generator, which only has one variant", t);}
            substs.field_tys(def_id, cx.tcx()).count()
        },
        _ => bug!("{} is not a type that can have fields.", t)
    };
    debug!("struct_llfields: variant: {:?}", variant);
    let mut first_field = true;
    let mut offset = Size::from_bytes(0);
    let mut result: Vec<Type> = Vec::with_capacity(field_count * 2);
    let field_iter = variant.field_index_by_increasing_offset().map(|i| {
        (i, match t.sty {
            ty::TyAdt(..) if i == 0 && discr.is_some() => discr.unwrap(),
            ty::TyAdt(ref def, ref substs) => {
                monomorphize::field_ty(cx.tcx(), substs,
                    &def.variants[variant_index].fields[i as usize - discr.is_some() as usize])
            },
            ty::TyTuple(fields, _) => fields[i as usize],
            ty::TyClosure(def_id, substs) => {
                substs.upvar_tys(def_id, cx.tcx()).nth(i).unwrap()
            },
            ty::TyGenerator(def_id, substs, _) => {
                let ty = substs.field_tys(def_id, cx.tcx()).nth(i).unwrap();
                cx.tcx().normalize_associated_type(&ty)
            },
            _ => bug!()
        }, variant.offsets[i as usize])
    });
    for (index, ty, target_offset) in field_iter {
        debug!("struct_llfields: {} ty: {} offset: {:?} target_offset: {:?}",
            index, ty, offset, target_offset);
        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        if first_field {
            assert_eq!(padding.bytes(), 0);
            first_field = false;
        } else {
            result.push(Type::array(&Type::i8(cx), padding.bytes()));
            debug!("    padding before: {:?}", padding);
        }
        let llty = type_of::in_memory_type_of(cx, ty);
        result.push(llty);
        let layout = cx.layout_of(ty);
        if variant.packed {
            assert_eq!(padding.bytes(), 0);
        } else {
            let field_align = layout.align(cx);
            assert!(field_align.abi() <= variant.align.abi(),
                    "non-packed type has field with larger align ({}): {:#?}",
                    field_align.abi(), variant);
        }
        let target_size = layout.size(&cx.tcx().data_layout);
        offset = target_offset + target_size;
    }
    if variant.sized && field_count > 0 {
        if offset > variant.stride() {
            bug!("variant: {:?} stride: {:?} offset: {:?}",
                variant, variant.stride(), offset);
        }
        let padding = variant.stride() - offset;
        debug!("struct_llfields: pad_bytes: {:?} offset: {:?} min_size: {:?} stride: {:?}",
            padding, offset, variant.min_size, variant.stride());
        result.push(Type::array(&Type::i8(cx), padding.bytes()));
        assert!(result.len() == (field_count * 2));
    } else {
        debug!("struct_llfields: offset: {:?} min_size: {:?} stride: {:?}",
               offset, variant.min_size, variant.stride());
    }

    result
}

pub fn is_discr_signed<'tcx>(l: &layout::Layout) -> bool {
    match *l {
        layout::CEnum { signed, .. }=> signed,
        _ => false,
    }
}

pub fn assert_discr_in_range<D: PartialOrd>(min: D, max: D, discr: D) {
    if min <= max {
        assert!(min <= discr && discr <= max)
    } else {
        assert!(min <= discr || discr <= max)
    }
}
