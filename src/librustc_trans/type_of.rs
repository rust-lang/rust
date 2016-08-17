// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use rustc::hir::def_id::DefId;
use abi::FnType;
use adt;
use common::*;
use machine;
use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TypeFoldable};

use type_::Type;

use syntax::ast;

// LLVM doesn't like objects that are too big. Issue #17913
fn ensure_array_fits_in_address_space<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                llet: Type,
                                                size: machine::llsize,
                                                scapegoat: Ty<'tcx>) {
    let esz = machine::llsize_of_alloc(ccx, llet);
    match esz.checked_mul(size) {
        Some(n) if n < ccx.obj_size_bound() => {}
        _ => { ccx.report_overbig_object(scapegoat) }
    }
}

// A "sizing type" is an LLVM type, the size and alignment of which are
// guaranteed to be equivalent to what you would get out of `type_of()`. It's
// useful because:
//
// (1) It may be cheaper to compute the sizing type than the full type if all
//     you're interested in is the size and/or alignment;
//
// (2) It won't make any recursive calls to determine the structure of the
//     type behind pointers. This can help prevent infinite loops for
//     recursive types. For example, enum types rely on this behavior.

pub fn sizing_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    if let Some(t) = cx.llsizingtypes().borrow().get(&t).cloned() {
        return t;
    }

    debug!("sizing_type_of {:?}", t);
    let _recursion_lock = cx.enter_type_of(t);

    let llsizingty = match t.sty {
        _ if !type_is_sized(cx.tcx(), t) => {
            Type::struct_(cx, &[Type::i8p(cx), unsized_info_ty(cx, t)], false)
        }

        ty::TyBool => Type::bool(cx),
        ty::TyChar => Type::char(cx),
        ty::TyInt(t) => Type::int_from_ty(cx, t),
        ty::TyUint(t) => Type::uint_from_ty(cx, t),
        ty::TyFloat(t) => Type::float_from_ty(cx, t),
        ty::TyNever => Type::nil(cx),

        ty::TyBox(ty) |
        ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) => {
            if type_is_sized(cx.tcx(), ty) {
                Type::i8p(cx)
            } else {
                Type::struct_(cx, &[Type::i8p(cx), unsized_info_ty(cx, ty)], false)
            }
        }

        ty::TyFnDef(..) => Type::nil(cx),
        ty::TyFnPtr(_) => Type::i8p(cx),

        ty::TyArray(ty, size) => {
            let llty = sizing_type_of(cx, ty);
            let size = size as u64;
            ensure_array_fits_in_address_space(cx, llty, size, t);
            Type::array(&llty, size)
        }

        ty::TyTuple(ref tys) if tys.is_empty() => {
            Type::nil(cx)
        }

        ty::TyTuple(..) | ty::TyEnum(..) | ty::TyClosure(..) => {
            let repr = adt::represent_type(cx, t);
            adt::sizing_type_of(cx, &repr, false)
        }

        ty::TyStruct(..) => {
            if t.is_simd() {
                let e = t.simd_type(cx.tcx());
                if !e.is_machine() {
                    cx.sess().fatal(&format!("monomorphising SIMD type `{}` with \
                                              a non-machine element type `{}`",
                                             t, e))
                }
                let llet = type_of(cx, e);
                let n = t.simd_size(cx.tcx()) as u64;
                ensure_array_fits_in_address_space(cx, llet, n, t);
                Type::vector(&llet, n)
            } else {
                let repr = adt::represent_type(cx, t);
                adt::sizing_type_of(cx, &repr, false)
            }
        }

        ty::TyProjection(..) | ty::TyInfer(..) | ty::TyParam(..) |
        ty::TyAnon(..) | ty::TyError => {
            bug!("fictitious type {:?} in sizing_type_of()", t)
        }
        ty::TySlice(_) | ty::TyTrait(..) | ty::TyStr => bug!()
    };

    debug!("--> mapped t={:?} to llsizingty={:?}", t, llsizingty);

    cx.llsizingtypes().borrow_mut().insert(t, llsizingty);

    // FIXME(eddyb) Temporary sanity check for ty::layout.
    let layout = cx.tcx().normalizing_infer_ctxt(Reveal::All).enter(|infcx| {
        t.layout(&infcx)
    });
    match layout {
        Ok(layout) => {
            if !type_is_sized(cx.tcx(), t) {
                if !layout.is_unsized() {
                    bug!("layout should be unsized for type `{}` / {:#?}",
                         t, layout);
                }

                // Unsized types get turned into a fat pointer for LLVM.
                return llsizingty;
            }
            let r = layout.size(&cx.tcx().data_layout).bytes();
            let l = machine::llsize_of_alloc(cx, llsizingty);
            if r != l {
                bug!("size differs (rustc: {}, llvm: {}) for type `{}` / {:#?}",
                     r, l, t, layout);
            }
            let r = layout.align(&cx.tcx().data_layout).abi();
            let l = machine::llalign_of_min(cx, llsizingty) as u64;
            if r != l {
                bug!("align differs (rustc: {}, llvm: {}) for type `{}` / {:#?}",
                     r, l, t, layout);
            }
        }
        Err(e) => {
            bug!("failed to get layout for `{}`: {}", t, e);
        }
    }
    llsizingty
}

pub fn fat_ptr_base_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> Type {
    match ty.sty {
        ty::TyBox(t) |
        ty::TyRef(_, ty::TypeAndMut { ty: t, .. }) |
        ty::TyRawPtr(ty::TypeAndMut { ty: t, .. }) if !type_is_sized(ccx.tcx(), t) => {
            in_memory_type_of(ccx, t).ptr_to()
        }
        _ => bug!("expected fat ptr ty but got {:?}", ty)
    }
}

fn unsized_info_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> Type {
    let unsized_part = ccx.tcx().struct_tail(ty);
    match unsized_part.sty {
        ty::TyStr | ty::TyArray(..) | ty::TySlice(_) => {
            Type::uint_from_ty(ccx, ast::UintTy::Us)
        }
        ty::TyTrait(_) => Type::vtable_ptr(ccx),
        _ => bug!("Unexpected tail in unsized_info_ty: {:?} for ty={:?}",
                          unsized_part, ty)
    }
}

pub fn immediate_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    if t.is_bool() {
        Type::i1(cx)
    } else {
        type_of(cx, t)
    }
}

/// Get the LLVM type corresponding to a Rust type, i.e. `rustc::ty::Ty`.
/// This is the right LLVM type for an alloca containing a value of that type,
/// and the pointee of an Lvalue Datum (which is always a LLVM pointer).
/// For unsized types, the returned type is a fat pointer, thus the resulting
/// LLVM type for a `Trait` Lvalue is `{ i8*, void(i8*)** }*`, which is a double
/// indirection to the actual data, unlike a `i8` Lvalue, which is just `i8*`.
/// This is needed due to the treatment of immediate values, as a fat pointer
/// is too large for it to be placed in SSA value (by our rules).
/// For the raw type without far pointer indirection, see `in_memory_type_of`.
pub fn type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> Type {
    let ty = if !type_is_sized(cx.tcx(), ty) {
        cx.tcx().mk_imm_ptr(ty)
    } else {
        ty
    };
    in_memory_type_of(cx, ty)
}

/// Get the LLVM type corresponding to a Rust type, i.e. `rustc::ty::Ty`.
/// This is the right LLVM type for a field/array element of that type,
/// and is the same as `type_of` for all Sized types.
/// Unsized types, however, are represented by a "minimal unit", e.g.
/// `[T]` becomes `T`, while `str` and `Trait` turn into `i8` - this
/// is useful for indexing slices, as `&[T]`'s data pointer is `T*`.
/// If the type is an unsized struct, the regular layout is generated,
/// with the inner-most trailing unsized field using the "minimal unit"
/// of that field's type - this is useful for taking the address of
/// that field and ensuring the struct has the right alignment.
/// For the LLVM type of a value as a whole, see `type_of`.
/// NB: If you update this, be sure to update `sizing_type_of()` as well.
pub fn in_memory_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
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
        let llty = in_memory_type_of(cx, t_norm);
        debug!("--> normalized {:?} to {:?} llty={:?}", t, t_norm, llty);
        cx.lltypes().borrow_mut().insert(t, llty);
        return llty;
    }

    let mut llty = match t.sty {
      ty::TyBool => Type::bool(cx),
      ty::TyChar => Type::char(cx),
      ty::TyInt(t) => Type::int_from_ty(cx, t),
      ty::TyUint(t) => Type::uint_from_ty(cx, t),
      ty::TyFloat(t) => Type::float_from_ty(cx, t),
      ty::TyNever => Type::nil(cx),
      ty::TyEnum(def, ref substs) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache. This
          // avoids creating more than one copy of the enum when one
          // of the enum's variants refers to the enum itself.
          let repr = adt::represent_type(cx, t);
          let name = llvm_type_name(cx, def.did, &substs.types);
          adt::incomplete_type_of(cx, &repr, &name[..])
      }
      ty::TyClosure(..) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache.
          let repr = adt::represent_type(cx, t);
          adt::incomplete_type_of(cx, &repr, "closure")
      }

      ty::TyBox(ty) |
      ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
      ty::TyRawPtr(ty::TypeAndMut{ty, ..}) => {
          if !type_is_sized(cx.tcx(), ty) {
              if let ty::TyStr = ty.sty {
                  // This means we get a nicer name in the output (str is always
                  // unsized).
                  cx.tn().find_type("str_slice").unwrap()
              } else {
                  let ptr_ty = in_memory_type_of(cx, ty).ptr_to();
                  let info_ty = unsized_info_ty(cx, ty);
                  Type::struct_(cx, &[ptr_ty, info_ty], false)
              }
          } else {
              in_memory_type_of(cx, ty).ptr_to()
          }
      }

      ty::TyArray(ty, size) => {
          let size = size as u64;
          // we must use `sizing_type_of` here as the type may
          // not be fully initialized.
          let szty = sizing_type_of(cx, ty);
          ensure_array_fits_in_address_space(cx, szty, size, t);

          let llty = in_memory_type_of(cx, ty);
          Type::array(&llty, size)
      }

      // Unsized slice types (and str) have the type of their element, and
      // traits have the type of u8. This is so that the data pointer inside
      // fat pointers is of the right type (e.g. for array accesses), even
      // when taking the address of an unsized field in a struct.
      ty::TySlice(ty) => in_memory_type_of(cx, ty),
      ty::TyStr | ty::TyTrait(..) => Type::i8(cx),

      ty::TyFnDef(..) => Type::nil(cx),
      ty::TyFnPtr(f) => {
        let sig = cx.tcx().erase_late_bound_regions(&f.sig);
        let sig = cx.tcx().normalize_associated_type(&sig);
        FnType::new(cx, f.abi, &sig, &[]).llvm_type(cx).ptr_to()
      }
      ty::TyTuple(ref tys) if tys.is_empty() => Type::nil(cx),
      ty::TyTuple(..) => {
          let repr = adt::represent_type(cx, t);
          adt::type_of(cx, &repr)
      }
      ty::TyStruct(def, ref substs) => {
          if t.is_simd() {
              let e = t.simd_type(cx.tcx());
              if !e.is_machine() {
                  cx.sess().fatal(&format!("monomorphising SIMD type `{}` with \
                                            a non-machine element type `{}`",
                                           t, e))
              }
              let llet = in_memory_type_of(cx, e);
              let n = t.simd_size(cx.tcx()) as u64;
              ensure_array_fits_in_address_space(cx, llet, n, t);
              Type::vector(&llet, n)
          } else {
              // Only create the named struct, but don't fill it in. We fill it
              // in *after* placing it into the type cache. This prevents
              // infinite recursion with recursive struct types.
              let repr = adt::represent_type(cx, t);
              let name = llvm_type_name(cx, def.did, &substs.types);
              adt::incomplete_type_of(cx, &repr, &name[..])
          }
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
        ty::TyEnum(..) | ty::TyStruct(..) | ty::TyClosure(..)
                if !t.is_simd() => {
            let repr = adt::represent_type(cx, t);
            adt::finish_type_of(cx, &repr, &mut llty);
        }
        _ => ()
    }

    llty
}

pub fn align_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>)
                          -> machine::llalign {
    let llty = sizing_type_of(cx, t);
    machine::llalign_of_min(cx, llty)
}

fn llvm_type_name<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                            did: DefId,
                            tps: &[Ty<'tcx>])
                            -> String {
    let base = cx.tcx().item_path_str(did);
    let strings: Vec<String> = tps.iter().map(|t| t.to_string()).collect();
    let tstr = if strings.is_empty() {
        base
    } else {
        format!("{}<{}>", base, strings.join(", "))
    };

    if did.krate == 0 {
        tstr
    } else {
        format!("{}.{}", did.krate, tstr)
    }
}
