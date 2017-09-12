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
use adt;
use common::*;
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::layout::{Align, Layout, LayoutOf, Size, FullLayout};
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

fn compute_llvm_type<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    // Check the cache.
    if let Some(&llty) = cx.lltypes().borrow().get(&t) {
        return llty;
    }

    debug!("type_of {:?}", t);

    assert!(!t.has_escaping_regions(), "{:?} has escaping regions", t);

    // Replace any typedef'd types with their equivalent non-typedef
    // type. This ensures that all LLVM nominal types that contain
    // Rust types are defined as the same LLVM types.  If we don't do
    // this then, e.g. `Option<{myfield: bool}>` would be a different
    // type than `Option<myrec>`.
    let t_norm = cx.tcx().erase_regions(&t);

    if t != t_norm {
        let llty = cx.llvm_type_of(t_norm);
        debug!("--> normalized {:?} to {:?} llty={:?}", t, t_norm, llty);
        cx.lltypes().borrow_mut().insert(t, llty);
        return llty;
    }

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

    let mut llty = match t.sty {
      ty::TyBool => Type::bool(cx),
      ty::TyChar => Type::char(cx),
      ty::TyInt(t) => Type::int_from_ty(cx, t),
      ty::TyUint(t) => Type::uint_from_ty(cx, t),
      ty::TyFloat(t) => Type::float_from_ty(cx, t),
      ty::TyNever => Type::nil(cx),
      ty::TyClosure(..) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache.
          adt::incomplete_type_of(cx, t, "closure")
      }
      ty::TyGenerator(..) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache.
          adt::incomplete_type_of(cx, t, "generator")
      }

      ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
      ty::TyRawPtr(ty::TypeAndMut{ty, ..}) => {
          ptr_ty(ty)
      }
      ty::TyAdt(def, _) if def.is_box() => {
          ptr_ty(t.boxed_ty())
      }

      ty::TyArray(ty, size) => {
          let llty = cx.llvm_type_of(ty);
          let size = size.val.to_const_int().unwrap().to_u64().unwrap();
          Type::array(&llty, size)
      }

      ty::TySlice(ty) => {
          Type::array(&cx.llvm_type_of(ty), 0)
      }
      ty::TyStr => {
          Type::array(&Type::i8(cx), 0)
      }
      ty::TyDynamic(..) |
      ty::TyForeign(..) => adt::type_of(cx, t),

      ty::TyFnDef(..) => Type::nil(cx),
      ty::TyFnPtr(sig) => {
        let sig = cx.tcx().erase_late_bound_regions_and_normalize(&sig);
        FnType::new(cx, sig, &[]).llvm_type(cx).ptr_to()
      }
      ty::TyTuple(ref tys, _) if tys.is_empty() => Type::nil(cx),
      ty::TyTuple(..) => {
          adt::type_of(cx, t)
      }
      ty::TyAdt(..) if t.is_simd() => {
          let e = t.simd_type(cx.tcx());
          if !e.is_machine() {
              cx.sess().fatal(&format!("monomorphising SIMD type `{}` with \
                                        a non-machine element type `{}`",
                                       t, e))
          }
          let llet = cx.llvm_type_of(e);
          let n = t.simd_size(cx.tcx()) as u64;
          Type::vector(&llet, n)
      }
      ty::TyAdt(..) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache. This
          // avoids creating more than one copy of the enum when one
          // of the enum's variants refers to the enum itself.
          let name = llvm_type_name(cx, t);
          adt::incomplete_type_of(cx, t, &name[..])
      }

      ty::TyInfer(..) |
      ty::TyProjection(..) |
      ty::TyParam(..) |
      ty::TyAnon(..) |
      ty::TyError => bug!("type_of with {:?}", t),
    };

    debug!("--> mapped t={:?} to llty={:?}", t, llty);

    cx.lltypes().borrow_mut().insert(t, llty);

    // If this was an enum or struct, fill in the type now.
    match t.sty {
        ty::TyAdt(..) | ty::TyClosure(..) | ty::TyGenerator(..) if !t.is_simd() && !t.is_box() => {
            adt::finish_type_of(cx, t, &mut llty);
        }
        _ => ()
    }

    llty
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
        compute_llvm_type(self, ty)
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
        match **self {
            Layout::Scalar { .. } |
            Layout::CEnum { .. } |
            Layout::UntaggedUnion { .. } |
            Layout::RawNullablePointer { .. } => {
                bug!("FullLayout::llvm_field_index({:?}): not applicable", self)
            }

            Layout::Vector { .. } |
            Layout::Array { .. } => {
                index as u64
            }

            Layout::FatPointer { .. } => {
                adt::memory_index_to_gep(index as u64)
            }

            Layout::Univariant { ref variant, .. } => {
                adt::memory_index_to_gep(variant.memory_index[index] as u64)
            }

            Layout::General { ref variants, .. } => {
                if let Some(v) = self.variant_index {
                    adt::memory_index_to_gep(variants[v].memory_index[index] as u64)
                } else {
                    assert_eq!(index, 0);
                    index as u64
                }
            }

            Layout::StructWrappedNullablePointer { nndiscr, ref nonnull, .. } => {
                if self.variant_index == Some(nndiscr as usize) {
                    adt::memory_index_to_gep(nonnull.memory_index[index] as u64)
                } else {
                    bug!("FullLayout::llvm_field_index({:?}): not applicable", self)
                }
            }
        }
    }
}

fn llvm_type_name<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> String {
    let mut name = String::with_capacity(32);
    let printer = DefPathBasedNames::new(cx.tcx(), true, true);
    printer.push_type_name(ty, &mut name);
    name
}
