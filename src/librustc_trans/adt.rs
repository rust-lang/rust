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
use rustc::ty::layout::{self, Align, HasDataLayout, LayoutOf, Size, FullLayout};

use context::CrateContext;
use type_::Type;

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
        layout::CEnum { .. } | layout::General { .. } | layout::UntaggedUnion { .. } => { }
        layout::Univariant { ..} | layout::NullablePointer { .. } => {
            if let layout::Abi::Scalar(_) = l.abi {
                return;
            }
            let (variant_layout, variant) = match *l {
                layout::Univariant(ref variant) => {
                    let is_enum = if let ty::TyAdt(def, _) = t.sty {
                        def.is_enum()
                    } else {
                        false
                    };
                    if is_enum {
                        (l.for_variant(0), variant)
                    } else {
                        (l, variant)
                    }
                }
                layout::NullablePointer { nndiscr, ref nonnull, .. } =>
                    (l.for_variant(nndiscr as usize), nonnull),
                _ => unreachable!()
            };
            llty.set_struct_body(&struct_llfields(cx, variant_layout, variant), variant.packed)
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
        layout::NullablePointer { nndiscr, ref nonnull, .. } => {
            if let layout::Abi::Scalar(_) = l.abi {
                return cx.llvm_type_of(l.field(cx, 0).ty);
            }
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, l.for_variant(nndiscr as usize),
                                                       nonnull),
                                  nonnull.packed)
                }
                Some(name) => {
                    Type::named_struct(cx, name)
                }
            }
        }
        layout::Univariant(ref variant) => {
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, l, &variant),
                                  variant.packed)
                }
                Some(name) => {
                    Type::named_struct(cx, name)
                }
            }
        }
        layout::UntaggedUnion(ref un) => {
            // Use alignment-sized ints to fill all the union storage.
            let fill = union_fill(cx, un.stride(), un.align);
            match name {
                None => {
                    Type::struct_(cx, &[fill], un.packed)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&[fill], un.packed);
                    llty
                }
            }
        }
        layout::General { size, align, .. } => {
            let fill = union_fill(cx, size, align);
            match name {
                None => {
                    Type::struct_(cx, &[fill], false)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&[fill], false);
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

/// Double an index and add 1 to account for padding.
pub fn memory_index_to_gep(index: u64) -> u64 {
    1 + index * 2
}

pub fn struct_llfields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                 layout: FullLayout<'tcx>,
                                 variant: &layout::Struct) -> Vec<Type> {
    let field_count = layout.fields.count();
    debug!("struct_llfields: variant: {:?}", variant);
    let mut offset = Size::from_bytes(0);
    let mut result: Vec<Type> = Vec::with_capacity(1 + field_count * 2);
    for i in variant.field_index_by_increasing_offset() {
        let field = layout.field(cx, i);
        let target_offset = variant.offsets[i as usize];
        debug!("struct_llfields: {}: {:?} offset: {:?} target_offset: {:?}",
            i, field, offset, target_offset);
        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        result.push(Type::array(&Type::i8(cx), padding.bytes()));
        debug!("    padding before: {:?}", padding);

        let llty = cx.llvm_type_of(field.ty);
        result.push(llty);

        if variant.packed {
            assert_eq!(padding.bytes(), 0);
        } else {
            let field_align = field.align(cx);
            assert!(field_align.abi() <= variant.align.abi(),
                    "non-packed type has field with larger align ({}): {:#?}",
                    field_align.abi(), variant);
        }

        offset = target_offset + field.size(cx);
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
        assert!(result.len() == 1 + field_count * 2);
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
