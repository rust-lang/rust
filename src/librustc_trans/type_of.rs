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
use rustc::ty::layout::{self, HasDataLayout, Align, LayoutOf, Size, TyLayout};
use trans_item::DefPathBasedNames;
use type_::Type;

use std::fmt::Write;

fn uncached_llvm_type<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                layout: TyLayout<'tcx>,
                                defer: &mut Option<(Type, TyLayout<'tcx>)>)
                                -> Type {
    match layout.abi {
        layout::Abi::Scalar(_) => bug!("handled elsewhere"),
        layout::Abi::Vector => {
            return Type::vector(&layout.field(ccx, 0).llvm_type(ccx),
                                layout.fields.count() as u64);
        }
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

pub trait LayoutLlvmExt<'tcx> {
    fn is_llvm_immediate(&self) -> bool;
    fn llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type;
    fn immediate_llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type;
    fn over_align(&self) -> Option<Align>;
    fn llvm_field_index(&self, index: usize) -> u64;
}

impl<'tcx> LayoutLlvmExt<'tcx> for TyLayout<'tcx> {
    fn is_llvm_immediate(&self) -> bool {
        match self.abi {
            layout::Abi::Scalar(_) | layout::Abi::Vector => true,

            layout::Abi::Aggregate { .. } => self.is_zst()
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
    fn llvm_type<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Type {
        if let layout::Abi::Scalar(value) = self.abi {
            // Use a different cache for scalars because pointers to DSTs
            // can be either fat or thin (data pointers of fat pointers).
            if let Some(&llty) = ccx.scalar_lltypes().borrow().get(&self.ty) {
                return llty;
            }
            let llty = match value {
                layout::Int(layout::I1, _) => Type::i8(ccx),
                layout::Int(i, _) => Type::from_integer(ccx, i),
                layout::F32 => Type::f32(ccx),
                layout::F64 => Type::f64(ccx),
                layout::Pointer => {
                    let pointee = match self.ty.sty {
                        ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
                        ty::TyRawPtr(ty::TypeAndMut { ty, .. }) => {
                            ccx.layout_of(ty).llvm_type(ccx)
                        }
                        ty::TyAdt(def, _) if def.is_box() => {
                            ccx.layout_of(self.ty.boxed_ty()).llvm_type(ccx)
                        }
                        ty::TyFnPtr(sig) => {
                            let sig = ccx.tcx().erase_late_bound_regions_and_normalize(&sig);
                            FnType::new(ccx, sig, &[]).llvm_type(ccx)
                        }
                        _ => Type::i8(ccx)
                    };
                    pointee.ptr_to()
                }
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
                layout = layout.for_variant(v);
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
        if let layout::Abi::Scalar(layout::Int(layout::I1, _)) = self.abi {
            Type::i1(ccx)
        } else {
            self.llvm_type(ccx)
        }
    }

    fn over_align(&self) -> Option<Align> {
        if self.align != self.primitive_align {
            Some(self.align)
        } else {
            None
        }
    }

    fn llvm_field_index(&self, index: usize) -> u64 {
        if let layout::Abi::Scalar(_) = self.abi {
            bug!("TyLayout::llvm_field_index({:?}): not applicable", self);
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
}
