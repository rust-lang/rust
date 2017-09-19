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
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::layout::{self, HasDataLayout, Align, LayoutOf, Size, FullLayout};
use trans_item::DefPathBasedNames;
use type_::Type;

use syntax::ast;

pub fn fat_ptr_base_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> Type {
    match ty.sty {
        ty::TyRef(_, ty::TypeAndMut { ty: t, .. }) |
        ty::TyRawPtr(ty::TypeAndMut { ty: t, .. }) if ccx.shared().type_has_metadata(t) => {
            ccx.llvm_type_of(t).ptr_to()
        }
        ty::TyAdt(def, _) if def.is_box() => {
            ccx.llvm_type_of(ty.boxed_ty()).ptr_to()
        }
        _ => bug!("expected fat ptr ty but got {:?}", ty)
    }
}

pub fn unsized_info_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> Type {
    let unsized_part = ccx.tcx().struct_tail(ty);
    match unsized_part.sty {
        ty::TyStr | ty::TyArray(..) | ty::TySlice(_) => {
            Type::uint_from_ty(ccx, ast::UintTy::Us)
        }
        ty::TyDynamic(..) => Type::vtable_ptr(ccx),
        _ => bug!("Unexpected tail in unsized_info_ty: {:?} for ty={:?}",
                          unsized_part, ty)
    }
}

fn uncached_llvm_type<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                ty: Ty<'tcx>,
                                defer: &mut Option<(Type, FullLayout<'tcx>)>)
                                -> Type {
    let ptr_ty = |ty: Ty<'tcx>| {
        if cx.shared().type_has_metadata(ty) {
            if let ty::TyStr = ty.sty {
                // This means we get a nicer name in the output (str is always
                // unsized).
                cx.str_slice_type()
            } else {
                let ptr_ty = cx.llvm_type_of(ty).ptr_to();
                let info_ty = unsized_info_ty(cx, ty);
                Type::struct_(cx, &[
                    Type::array(&Type::i8(cx), 0),
                    ptr_ty,
                    Type::array(&Type::i8(cx), 0),
                    info_ty,
                    Type::array(&Type::i8(cx), 0)
                ], false)
            }
        } else {
            cx.llvm_type_of(ty).ptr_to()
        }
    };
    match ty.sty {
        ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) => {
            return ptr_ty(ty);
        }
        ty::TyAdt(def, _) if def.is_box() => {
            return ptr_ty(ty.boxed_ty());
        }
        ty::TyFnPtr(sig) => {
            let sig = cx.tcx().erase_late_bound_regions_and_normalize(&sig);
            return FnType::new(cx, sig, &[]).llvm_type(cx).ptr_to();
        }
        _ => {}
    }

    let layout = cx.layout_of(ty);
    if let layout::Abi::Scalar(value) = layout.abi {
        let llty = match value {
            layout::Int(layout::I1, _) => Type::i8(cx),
            layout::Int(i, _) => Type::from_integer(cx, i),
            layout::F32 => Type::f32(cx),
            layout::F64 => Type::f64(cx),
            layout::Pointer => cx.llvm_type_of(layout::Pointer.to_ty(cx.tcx()))
        };
        return llty;
    }

    if let layout::Abi::Vector { .. } = layout.abi {
        return Type::vector(&cx.llvm_type_of(layout.field(cx, 0).ty),
                            layout.fields.count() as u64);
    }

    let name = match ty.sty {
        ty::TyClosure(..) | ty::TyGenerator(..) | ty::TyAdt(..) => {
            let mut name = String::with_capacity(32);
            let printer = DefPathBasedNames::new(cx.tcx(), true, true);
            printer.push_type_name(ty, &mut name);
            Some(name)
        }
        _ => None
    };

    match *layout.fields {
        layout::FieldPlacement::Union(_) => {
            let size = layout.size(cx).bytes();
            let fill = Type::array(&Type::i8(cx), size);
            match name {
                None => {
                    Type::struct_(cx, &[fill], layout.is_packed())
                }
                Some(ref name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&[fill], layout.is_packed());
                    llty
                }
            }
        }
        layout::FieldPlacement::Array { count, .. } => {
            Type::array(&cx.llvm_type_of(layout.field(cx, 0).ty), count)
        }
        layout::FieldPlacement::Arbitrary { .. } => {
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, layout), layout.is_packed())
                }
                Some(ref name) => {
                    let llty = Type::named_struct(cx, name);
                    *defer = Some((llty, layout));
                    llty
                }
            }
        }
    }
}

pub fn struct_llfields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                 layout: FullLayout<'tcx>) -> Vec<Type> {
    debug!("struct_llfields: {:#?}", layout);
    let align = layout.align(cx);
    let size = layout.size(cx);
    let field_count = layout.fields.count();

    let mut offset = Size::from_bytes(0);
    let mut result: Vec<Type> = Vec::with_capacity(1 + field_count * 2);
    for i in layout.fields.index_by_increasing_offset() {
        let field = layout.field(cx, i);
        let target_offset = layout.fields.offset(i as usize);
        debug!("struct_llfields: {}: {:?} offset: {:?} target_offset: {:?}",
            i, field, offset, target_offset);
        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        result.push(Type::array(&Type::i8(cx), padding.bytes()));
        debug!("    padding before: {:?}", padding);

        let llty = cx.llvm_type_of(field.ty);
        result.push(llty);

        if layout.is_packed() {
            assert_eq!(padding.bytes(), 0);
        } else {
            let field_align = field.align(cx);
            assert!(field_align.abi() <= align.abi(),
                    "non-packed type has field with larger align ({}): {:#?}",
                    field_align.abi(), layout);
        }

        offset = target_offset + field.size(cx);
    }
    if !layout.is_unsized() && field_count > 0 {
        if offset > size {
            bug!("layout: {:#?} stride: {:?} offset: {:?}",
                 layout, size, offset);
        }
        let padding = size - offset;
        debug!("struct_llfields: pad_bytes: {:?} offset: {:?} stride: {:?}",
               padding, offset, size);
        result.push(Type::array(&Type::i8(cx), padding.bytes()));
        assert!(result.len() == 1 + field_count * 2);
    } else {
        debug!("struct_llfields: offset: {:?} stride: {:?}",
               offset, size);
    }

    result
}

impl<'a, 'tcx> CrateContext<'a, 'tcx> {
    pub fn align_of(&self, ty: Ty<'tcx>) -> Align {
        self.layout_of(ty).align(self)
    }

    pub fn size_of(&self, ty: Ty<'tcx>) -> Size {
        self.layout_of(ty).size(self)
    }

    pub fn size_and_align_of(&self, ty: Ty<'tcx>) -> (Size, Align) {
        let layout = self.layout_of(ty);
        (layout.size(self), layout.align(self))
    }

    /// Returns alignment if it is different than the primitive alignment.
    pub fn over_align_of(&self, ty: Ty<'tcx>) -> Option<Align> {
        let layout = self.layout_of(ty);
        let align = layout.align(self);
        let primitive_align = layout.primitive_align(self);
        if align != primitive_align {
            Some(align)
        } else {
            None
        }
    }

    /// Get the LLVM type corresponding to a Rust type, i.e. `rustc::ty::Ty`.
    /// The pointee type of the pointer in `LvalueRef` is always this type.
    /// For sized types, it is also the right LLVM type for an `alloca`
    /// containing a value of that type, and most immediates (except `bool`).
    /// Unsized types, however, are represented by a "minimal unit", e.g.
    /// `[T]` becomes `T`, while `str` and `Trait` turn into `i8` - this
    /// is useful for indexing slices, as `&[T]`'s data pointer is `T*`.
    /// If the type is an unsized struct, the regular layout is generated,
    /// with the inner-most trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    pub fn llvm_type_of(&self, ty: Ty<'tcx>) -> Type {
        // Check the cache.
        if let Some(&llty) = self.lltypes().borrow().get(&ty) {
            return llty;
        }

        debug!("type_of {:?}", ty);

        assert!(!ty.has_escaping_regions(), "{:?} has escaping regions", ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = self.tcx().erase_regions(&ty);

        if ty != normal_ty {
            let llty = self.llvm_type_of(normal_ty);
            debug!("--> normalized {:?} to {:?} llty={:?}", ty, normal_ty, llty);
            self.lltypes().borrow_mut().insert(ty, llty);
            return llty;
        }

        let mut defer = None;
        let llty = uncached_llvm_type(self, ty, &mut defer);

        debug!("--> mapped ty={:?} to llty={:?}", ty, llty);

        self.lltypes().borrow_mut().insert(ty, llty);

        if let Some((mut llty, layout)) = defer {
            llty.set_struct_body(&struct_llfields(self, layout), layout.is_packed())
        }

        llty
    }

    pub fn immediate_llvm_type_of(&self, ty: Ty<'tcx>) -> Type {
        if ty.is_bool() {
            Type::i1(self)
        } else {
            self.llvm_type_of(ty)
        }
    }
}

pub trait LayoutLlvmExt {
    fn llvm_field_index(&self, index: usize) -> u64;
}

impl<'tcx> LayoutLlvmExt for FullLayout<'tcx> {
    fn llvm_field_index(&self, index: usize) -> u64 {
        if let layout::Abi::Scalar(_) = self.abi {
            bug!("FullLayout::llvm_field_index({:?}): not applicable", self);
        }
        match *self.fields {
            layout::FieldPlacement::Union(_) => {
                bug!("FullLayout::llvm_field_index({:?}): not applicable", self)
            }

            layout::FieldPlacement::Array { .. } => {
                index as u64
            }

            layout::FieldPlacement::Arbitrary { .. } => {
                1 + (self.fields.memory_index(index) as u64) * 2
            }
        }
    }
}
