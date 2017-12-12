// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::FnType;
use common::*;
use rustc::hir;
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::layout::{self, Align, LayoutOf, Size, TyLayout};
use rustc_back::PanicStrategy;
use trans_item::DefPathBasedNames;
use type_::Type;

use std::fmt::Write;

fn uncached_llvm_type<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                layout: TyLayout<'tcx>,
                                defer: &mut Option<(Type, TyLayout<'tcx>)>)
                                -> Type {
    match layout.abi {
        layout::Abi::Scalar(_) => bug!("handled elsewhere"),
        layout::Abi::Vector { ref element, count } => {
            // LLVM has a separate type for 64-bit SIMD vectors on X86 called
            // `x86_mmx` which is needed for some SIMD operations. As a bit of a
            // hack (all SIMD definitions are super unstable anyway) we
            // recognize any one-element SIMD vector as "this should be an
            // x86_mmx" type. In general there shouldn't be a need for other
            // one-element SIMD vectors, so it's assumed this won't clash with
            // much else.
            let use_x86_mmx = count == 1 && layout.size.bits() == 64 &&
                (ccx.sess().target.target.arch == "x86" ||
                 ccx.sess().target.target.arch == "x86_64");
            if use_x86_mmx {
                return Type::x86_mmx(ccx)
            } else {
                let element = layout.scalar_llvm_type_at(ccx, element, Size::from_bytes(0));
                return Type::vector(&element, count);
            }
        }
        layout::Abi::ScalarPair(..) => {
            return Type::struct_(ccx, &[
                layout.scalar_pair_element_llvm_type(ccx, 0),
                layout.scalar_pair_element_llvm_type(ccx, 1),
            ], false);
        }
        layout::Abi::Uninhabited |
        layout::Abi::Aggregate { .. } => {}
    }

    let name = match layout.ty.sty {
        ty::TyClosure(..) |
        ty::TyGenerator(..) |
        ty::TyAdt(..) |
        ty::TyDynamic(..) |
        ty::TyForeign(..) |
        ty::TyStr => {
            let mut name = String::with_capacity(32);
            let printer = DefPathBasedNames::new(ccx.tcx(), true, true);
            printer.push_type_name(layout.ty, &mut name);
            match (&layout.ty.sty, &layout.variants) {
                (&ty::TyAdt(def, _), &layout::Variants::Single { index }) => {
                    if def.is_enum() && !def.variants.is_empty() {
                        write!(&mut name, "::{}", def.variants[index].name).unwrap();
                    }
                }
                _ => {}
            }
            Some(name)
        }
        _ => None
    };

    match layout.fields {
        layout::FieldPlacement::Union(_) => {
            let size = layout.size.bytes();
            let fill = Type::array(&Type::i8(ccx), size);
            match name {
                None => {
                    Type::struct_(ccx, &[fill], layout.is_packed())
                }
                Some(ref name) => {
                    let mut llty = Type::named_struct(ccx, name);
                    llty.set_struct_body(&[fill], layout.is_packed());
                    llty
                }
            }
        }
        layout::FieldPlacement::Array { count, .. } => {
            Type::array(&layout.field(ccx, 0).llvm_type(ccx), count)
        }
        layout::FieldPlacement::Arbitrary { .. } => {
            match name {
                None => {
                    Type::struct_(ccx, &struct_llfields(ccx, layout), layout.is_packed())
                }
                Some(ref name) => {
                    let llty = Type::named_struct(ccx, name);
                    *defer = Some((llty, layout));
                    llty
                }
            }
        }
    }
}

fn struct_llfields<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                             layout: TyLayout<'tcx>) -> Vec<Type> {
    debug!("struct_llfields: {:#?}", layout);
    let field_count = layout.fields.count();

    let mut offset = Size::from_bytes(0);
    let mut result: Vec<Type> = Vec::with_capacity(1 + field_count * 2);
    for i in layout.fields.index_by_increasing_offset() {
        let field = layout.field(ccx, i);
        let target_offset = layout.fields.offset(i as usize);
        debug!("struct_llfields: {}: {:?} offset: {:?} target_offset: {:?}",
            i, field, offset, target_offset);
        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        result.push(Type::array(&Type::i8(ccx), padding.bytes()));
        debug!("    padding before: {:?}", padding);

        result.push(field.llvm_type(ccx));

        if layout.is_packed() {
            assert_eq!(padding.bytes(), 0);
        } else {
            assert!(field.align.abi() <= layout.align.abi(),
                    "non-packed type has field with larger align ({}): {:#?}",
                    field.align.abi(), layout);
        }

        offset = target_offset + field.size;
    }
    if !layout.is_unsized() && field_count > 0 {
        if offset > layout.size {
            bug!("layout: {:#?} stride: {:?} offset: {:?}",
                 layout, layout.size, offset);
        }
        let padding = layout.size - offset;
        debug!("struct_llfields: pad_bytes: {:?} offset: {:?} stride: {:?}",
               padding, offset, layout.size);
        result.push(Type::array(&Type::i8(ccx), padding.bytes()));
        assert!(result.len() == 1 + field_count * 2);
    } else {
        debug!("struct_llfields: offset: {:?} stride: {:?}",
               offset, layout.size);
    }

    result
}

impl<'a, 'tcx> CrateContext<'a, 'tcx> {
    pub fn align_of(&self, ty: Ty<'tcx>) -> Align {
        self.layout_of(ty).align
    }

    pub fn size_of(&self, ty: Ty<'tcx>) -> Size {
        self.layout_of(ty).size
    }

    pub fn size_and_align_of(&self, ty: Ty<'tcx>) -> (Size, Align) {
        self.layout_of(ty).size_and_align()
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PointerKind {
    /// Most general case, we know no restrictions to tell LLVM.
    Shared,

    /// `&T` where `T` contains no `UnsafeCell`, is `noalias` and `readonly`.
    Frozen,

    /// `&mut T`, when we know `noalias` is safe for LLVM.
    UniqueBorrowed,

    /// `Box<T>`, unlike `UniqueBorrowed`, it also has `noalias` on returns.
    UniqueOwned
}

#[derive(Copy, Clone)]
pub struct PointeeInfo {
    pub size: Size,
    pub align: Align,
    pub safe: Option<PointerKind>,
}

pub trait LayoutLlvmExt<'tcx> {
    fn is_llvm_immediate(&self) -> bool;
    fn is_llvm_scalar_pair<'a>(&self) -> bool;
    fn llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type;
    fn immediate_llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type;
    fn scalar_llvm_type_at<'a>(&self, ccx: &CrateContext<'a, 'tcx>,
                               scalar: &layout::Scalar, offset: Size) -> Type;
    fn scalar_pair_element_llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>,
                                         index: usize) -> Type;
    fn llvm_field_index(&self, index: usize) -> u64;
    fn pointee_info_at<'a>(&self, ccx: &CrateContext<'a, 'tcx>, offset: Size)
                           -> Option<PointeeInfo>;
}

impl<'tcx> LayoutLlvmExt<'tcx> for TyLayout<'tcx> {
    fn is_llvm_immediate(&self) -> bool {
        match self.abi {
            layout::Abi::Uninhabited |
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } => true,
            layout::Abi::ScalarPair(..) => false,
            layout::Abi::Aggregate { .. } => self.is_zst()
        }
    }

    fn is_llvm_scalar_pair<'a>(&self) -> bool {
        match self.abi {
            layout::Abi::ScalarPair(..) => true,
            layout::Abi::Uninhabited |
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } |
            layout::Abi::Aggregate { .. } => false
        }
    }

    /// Get the LLVM type corresponding to a Rust type, i.e. `rustc::ty::Ty`.
    /// The pointee type of the pointer in `PlaceRef` is always this type.
    /// For sized types, it is also the right LLVM type for an `alloca`
    /// containing a value of that type, and most immediates (except `bool`).
    /// Unsized types, however, are represented by a "minimal unit", e.g.
    /// `[T]` becomes `T`, while `str` and `Trait` turn into `i8` - this
    /// is useful for indexing slices, as `&[T]`'s data pointer is `T*`.
    /// If the type is an unsized struct, the regular layout is generated,
    /// with the inner-most trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    fn llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type {
        if let layout::Abi::Scalar(ref scalar) = self.abi {
            // Use a different cache for scalars because pointers to DSTs
            // can be either fat or thin (data pointers of fat pointers).
            if let Some(&llty) = ccx.scalar_lltypes().borrow().get(&self.ty) {
                return llty;
            }
            let llty = match self.ty.sty {
                ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
                ty::TyRawPtr(ty::TypeAndMut { ty, .. }) => {
                    ccx.layout_of(ty).llvm_type(ccx).ptr_to()
                }
                ty::TyAdt(def, _) if def.is_box() => {
                    ccx.layout_of(self.ty.boxed_ty()).llvm_type(ccx).ptr_to()
                }
                ty::TyFnPtr(sig) => {
                    let sig = ccx.tcx().erase_late_bound_regions_and_normalize(&sig);
                    FnType::new(ccx, sig, &[]).llvm_type(ccx).ptr_to()
                }
                _ => self.scalar_llvm_type_at(ccx, scalar, Size::from_bytes(0))
            };
            ccx.scalar_lltypes().borrow_mut().insert(self.ty, llty);
            return llty;
        }


        // Check the cache.
        let variant_index = match self.variants {
            layout::Variants::Single { index } => Some(index),
            _ => None
        };
        if let Some(&llty) = ccx.lltypes().borrow().get(&(self.ty, variant_index)) {
            return llty;
        }

        debug!("llvm_type({:#?})", self);

        assert!(!self.ty.has_escaping_regions(), "{:?} has escaping regions", self.ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = ccx.tcx().erase_regions(&self.ty);

        let mut defer = None;
        let llty = if self.ty != normal_ty {
            let mut layout = ccx.layout_of(normal_ty);
            if let Some(v) = variant_index {
                layout = layout.for_variant(ccx, v);
            }
            layout.llvm_type(ccx)
        } else {
            uncached_llvm_type(ccx, *self, &mut defer)
        };
        debug!("--> mapped {:#?} to llty={:?}", self, llty);

        ccx.lltypes().borrow_mut().insert((self.ty, variant_index), llty);

        if let Some((mut llty, layout)) = defer {
            llty.set_struct_body(&struct_llfields(ccx, layout), layout.is_packed())
        }

        llty
    }

    fn immediate_llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type {
        if let layout::Abi::Scalar(ref scalar) = self.abi {
            if scalar.is_bool() {
                return Type::i1(ccx);
            }
        }
        self.llvm_type(ccx)
    }

    fn scalar_llvm_type_at<'a>(&self, ccx: &CrateContext<'a, 'tcx>,
                               scalar: &layout::Scalar, offset: Size) -> Type {
        match scalar.value {
            layout::Int(i, _) => Type::from_integer(ccx, i),
            layout::F32 => Type::f32(ccx),
            layout::F64 => Type::f64(ccx),
            layout::Pointer => {
                // If we know the alignment, pick something better than i8.
                let pointee = if let Some(pointee) = self.pointee_info_at(ccx, offset) {
                    Type::pointee_for_abi_align(ccx, pointee.align)
                } else {
                    Type::i8(ccx)
                };
                pointee.ptr_to()
            }
        }
    }

    fn scalar_pair_element_llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>,
                                         index: usize) -> Type {
        // HACK(eddyb) special-case fat pointers until LLVM removes
        // pointee types, to avoid bitcasting every `OperandRef::deref`.
        match self.ty.sty {
            ty::TyRef(..) |
            ty::TyRawPtr(_) => {
                return self.field(ccx, index).llvm_type(ccx);
            }
            ty::TyAdt(def, _) if def.is_box() => {
                let ptr_ty = ccx.tcx().mk_mut_ptr(self.ty.boxed_ty());
                return ccx.layout_of(ptr_ty).scalar_pair_element_llvm_type(ccx, index);
            }
            _ => {}
        }

        let (a, b) = match self.abi {
            layout::Abi::ScalarPair(ref a, ref b) => (a, b),
            _ => bug!("TyLayout::scalar_pair_element_llty({:?}): not applicable", self)
        };
        let scalar = [a, b][index];

        // Make sure to return the same type `immediate_llvm_type` would,
        // to avoid dealing with two types and the associated conversions.
        // This means that `(bool, bool)` is represented as `{i1, i1}`,
        // both in memory and as an immediate, while `bool` is typically
        // `i8` in memory and only `i1` when immediate. While we need to
        // load/store `bool` as `i8` to avoid crippling LLVM optimizations,
        // `i1` in a LLVM aggregate is valid and mostly equivalent to `i8`.
        if scalar.is_bool() {
            return Type::i1(ccx);
        }

        let offset = if index == 0 {
            Size::from_bytes(0)
        } else {
            a.value.size(ccx).abi_align(b.value.align(ccx))
        };
        self.scalar_llvm_type_at(ccx, scalar, offset)
    }

    fn llvm_field_index(&self, index: usize) -> u64 {
        match self.abi {
            layout::Abi::Scalar(_) |
            layout::Abi::ScalarPair(..) => {
                bug!("TyLayout::llvm_field_index({:?}): not applicable", self)
            }
            _ => {}
        }
        match self.fields {
            layout::FieldPlacement::Union(_) => {
                bug!("TyLayout::llvm_field_index({:?}): not applicable", self)
            }

            layout::FieldPlacement::Array { .. } => {
                index as u64
            }

            layout::FieldPlacement::Arbitrary { .. } => {
                1 + (self.fields.memory_index(index) as u64) * 2
            }
        }
    }

    fn pointee_info_at<'a>(&self, ccx: &CrateContext<'a, 'tcx>, offset: Size)
                           -> Option<PointeeInfo> {
        if let Some(&pointee) = ccx.pointee_infos().borrow().get(&(self.ty, offset)) {
            return pointee;
        }

        let mut result = None;
        match self.ty.sty {
            ty::TyRawPtr(mt) if offset.bytes() == 0 => {
                let (size, align) = ccx.size_and_align_of(mt.ty);
                result = Some(PointeeInfo {
                    size,
                    align,
                    safe: None
                });
            }

            ty::TyRef(_, mt) if offset.bytes() == 0 => {
                let (size, align) = ccx.size_and_align_of(mt.ty);

                let kind = match mt.mutbl {
                    hir::MutImmutable => if ccx.shared().type_is_freeze(mt.ty) {
                        PointerKind::Frozen
                    } else {
                        PointerKind::Shared
                    },
                    hir::MutMutable => {
                        if ccx.shared().tcx().sess.opts.debugging_opts.mutable_noalias ||
                           ccx.shared().tcx().sess.panic_strategy() == PanicStrategy::Abort {
                            PointerKind::UniqueBorrowed
                        } else {
                            PointerKind::Shared
                        }
                    }
                };

                result = Some(PointeeInfo {
                    size,
                    align,
                    safe: Some(kind)
                });
            }

            _ => {
                let mut data_variant = match self.variants {
                    layout::Variants::NicheFilling { dataful_variant, .. } => {
                        // Only the niche itself is always initialized,
                        // so only check for a pointer at its offset.
                        //
                        // If the niche is a pointer, it's either valid
                        // (according to its type), or null (which the
                        // niche field's scalar validity range encodes).
                        // This allows using `dereferenceable_or_null`
                        // for e.g. `Option<&T>`, and this will continue
                        // to work as long as we don't start using more
                        // niches than just null (e.g. the first page
                        // of the address space, or unaligned pointers).
                        if self.fields.offset(0) == offset {
                            Some(self.for_variant(ccx, dataful_variant))
                        } else {
                            None
                        }
                    }
                    _ => Some(*self)
                };

                if let Some(variant) = data_variant {
                    // We're not interested in any unions.
                    if let layout::FieldPlacement::Union(_) = variant.fields {
                        data_variant = None;
                    }
                }

                if let Some(variant) = data_variant {
                    let ptr_end = offset + layout::Pointer.size(ccx);
                    for i in 0..variant.fields.count() {
                        let field_start = variant.fields.offset(i);
                        if field_start <= offset {
                            let field = variant.field(ccx, i);
                            if ptr_end <= field_start + field.size {
                                // We found the right field, look inside it.
                                result = field.pointee_info_at(ccx, offset - field_start);
                                break;
                            }
                        }
                    }
                }

                // FIXME(eddyb) This should be for `ptr::Unique<T>`, not `Box<T>`.
                if let Some(ref mut pointee) = result {
                    if let ty::TyAdt(def, _) = self.ty.sty {
                        if def.is_box() && offset.bytes() == 0 {
                            pointee.safe = Some(PointerKind::UniqueOwned);
                        }
                    }
                }
            }
        }

        ccx.pointee_infos().borrow_mut().insert((self.ty, offset), result);
        result
    }
}
