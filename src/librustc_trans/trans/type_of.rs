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

use middle::def_id::DefId;
use middle::infer;
use middle::subst;
use trans::adt;
use trans::common::*;
use trans::foreign;
use trans::machine;
use middle::ty::{self, RegionEscape, Ty};

use trans::type_::Type;

use syntax::abi;
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

pub fn arg_is_indirect<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                 arg_ty: Ty<'tcx>) -> bool {
    !type_is_immediate(ccx, arg_ty) && !type_is_fat_ptr(ccx.tcx(), arg_ty)
}

pub fn return_uses_outptr<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    ty: Ty<'tcx>) -> bool {
    arg_is_indirect(ccx, ty)
}

pub fn type_of_explicit_arg<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                      arg_ty: Ty<'tcx>) -> Type {
    let llty = arg_type_of(ccx, arg_ty);
    if arg_is_indirect(ccx, arg_ty) {
        llty.ptr_to()
    } else {
        llty
    }
}

/// Yields the types of the "real" arguments for a function using the `RustCall`
/// ABI by untupling the arguments of the function.
pub fn untuple_arguments<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                   inputs: &[Ty<'tcx>])
                                   -> Vec<Ty<'tcx>> {
    if inputs.is_empty() {
        return Vec::new()
    }

    let mut result = Vec::new();
    for (i, &arg_prior_to_tuple) in inputs.iter().enumerate() {
        if i < inputs.len() - 1 {
            result.push(arg_prior_to_tuple);
        }
    }

    match inputs[inputs.len() - 1].sty {
        ty::TyTuple(ref tupled_arguments) => {
            debug!("untuple_arguments(): untupling arguments");
            for &tupled_argument in tupled_arguments {
                result.push(tupled_argument);
            }
        }
        _ => {
            ccx.tcx().sess.bug("argument to function with \"rust-call\" ABI \
                                is neither a tuple nor unit")
        }
    }

    result
}

pub fn type_of_rust_fn<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                 llenvironment_type: Option<Type>,
                                 sig: &ty::FnSig<'tcx>,
                                 abi: abi::Abi)
                                 -> Type
{
    debug!("type_of_rust_fn(sig={:?},abi={:?})",
           sig,
           abi);

    assert!(!sig.variadic); // rust fns are never variadic

    let mut atys: Vec<Type> = Vec::new();

    // First, munge the inputs, if this has the `rust-call` ABI.
    let inputs_temp;
    let inputs = if abi == abi::RustCall {
        inputs_temp = untuple_arguments(cx, &sig.inputs);
        &inputs_temp
    } else {
        &sig.inputs
    };

    // Arg 0: Output pointer.
    // (if the output type is non-immediate)
    let lloutputtype = match sig.output {
        ty::FnConverging(output) => {
            let use_out_pointer = return_uses_outptr(cx, output);
            let lloutputtype = arg_type_of(cx, output);
            // Use the output as the actual return value if it's immediate.
            if use_out_pointer {
                atys.push(lloutputtype.ptr_to());
                Type::void(cx)
            } else if return_type_is_void(cx, output) {
                Type::void(cx)
            } else {
                lloutputtype
            }
        }
        ty::FnDiverging => Type::void(cx)
    };

    // Arg 1: Environment
    match llenvironment_type {
        None => {}
        Some(llenvironment_type) => atys.push(llenvironment_type),
    }

    // ... then explicit args.
    for input in inputs {
        let arg_ty = type_of_explicit_arg(cx, input);

        if type_is_fat_ptr(cx.tcx(), input) {
            atys.extend(arg_ty.field_types());
        } else {
            atys.push(arg_ty);
        }
    }

    Type::func(&atys[..], &lloutputtype)
}

// Given a function type and a count of ty params, construct an llvm type
pub fn type_of_fn_from_ty<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, fty: Ty<'tcx>) -> Type {
    match fty.sty {
        ty::TyBareFn(_, ref f) => {
            // FIXME(#19925) once fn item types are
            // zero-sized, we'll need to do something here
            if f.abi == abi::Rust || f.abi == abi::RustCall {
                let sig = cx.tcx().erase_late_bound_regions(&f.sig);
                let sig = infer::normalize_associated_type(cx.tcx(), &sig);
                type_of_rust_fn(cx, None, &sig, f.abi)
            } else {
                foreign::lltype_for_foreign_fn(cx, fty)
            }
        }
        _ => {
            cx.sess().bug("type_of_fn_from_ty given non-closure, non-bare-fn")
        }
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
    match cx.llsizingtypes().borrow().get(&t).cloned() {
        Some(t) => return t,
        None => ()
    }

    debug!("sizing_type_of {:?}", t);
    let _recursion_lock = cx.enter_type_of(t);

    let llsizingty = match t.sty {
        _ if !type_is_sized(cx.tcx(), t) => {
            Type::struct_(cx, &[Type::i8p(cx), Type::i8p(cx)], false)
        }

        ty::TyBool => Type::bool(cx),
        ty::TyChar => Type::char(cx),
        ty::TyInt(t) => Type::int_from_ty(cx, t),
        ty::TyUint(t) => Type::uint_from_ty(cx, t),
        ty::TyFloat(t) => Type::float_from_ty(cx, t),

        ty::TyBox(ty) |
        ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) => {
            if type_is_sized(cx.tcx(), ty) {
                Type::i8p(cx)
            } else {
                Type::struct_(cx, &[Type::i8p(cx), Type::i8p(cx)], false)
            }
        }

        ty::TyBareFn(..) => Type::i8p(cx),

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
            adt::sizing_type_of(cx, &*repr, false)
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
                adt::sizing_type_of(cx, &*repr, false)
            }
        }

        ty::TyProjection(..) | ty::TyInfer(..) | ty::TyParam(..) | ty::TyError(..) => {
            cx.sess().bug(&format!("fictitious type {:?} in sizing_type_of()",
                                   t))
        }
        ty::TySlice(_) | ty::TyTrait(..) | ty::TyStr => unreachable!()
    };

    debug!("--> mapped t={:?} to llsizingty={}",
            t,
            cx.tn().type_to_string(llsizingty));

    cx.llsizingtypes().borrow_mut().insert(t, llsizingty);
    llsizingty
}

pub fn foreign_arg_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    if t.is_bool() {
        Type::i1(cx)
    } else {
        type_of(cx, t)
    }
}

pub fn arg_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, t: Ty<'tcx>) -> Type {
    if t.is_bool() {
        Type::i1(cx)
    } else if type_is_immediate(cx, t) && type_of(cx, t).is_aggregate() {
        // We want to pass small aggregates as immediate values, but using an aggregate LLVM type
        // for this leads to bad optimizations, so its arg type is an appropriately sized integer
        match machine::llsize_of_alloc(cx, sizing_type_of(cx, t)) {
            0 => type_of(cx, t),
            n => Type::ix(cx, n * 8),
        }
    } else {
        type_of(cx, t)
    }
}

/// Get the LLVM type corresponding to a Rust type, i.e. `middle::ty::Ty`.
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

/// Get the LLVM type corresponding to a Rust type, i.e. `middle::ty::Ty`.
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
    match cx.lltypes().borrow().get(&t) {
        Some(&llty) => return llty,
        None => ()
    }

    debug!("type_of {:?}", t);

    assert!(!t.has_escaping_regions());

    // Replace any typedef'd types with their equivalent non-typedef
    // type. This ensures that all LLVM nominal types that contain
    // Rust types are defined as the same LLVM types.  If we don't do
    // this then, e.g. `Option<{myfield: bool}>` would be a different
    // type than `Option<myrec>`.
    let t_norm = cx.tcx().erase_regions(&t);

    if t != t_norm {
        let llty = in_memory_type_of(cx, t_norm);
        debug!("--> normalized {:?} {:?} to {:?} {:?} llty={}",
                t,
                t,
                t_norm,
                t_norm,
                cx.tn().type_to_string(llty));
        cx.lltypes().borrow_mut().insert(t, llty);
        return llty;
    }

    let mut llty = match t.sty {
      ty::TyBool => Type::bool(cx),
      ty::TyChar => Type::char(cx),
      ty::TyInt(t) => Type::int_from_ty(cx, t),
      ty::TyUint(t) => Type::uint_from_ty(cx, t),
      ty::TyFloat(t) => Type::float_from_ty(cx, t),
      ty::TyEnum(def, ref substs) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache. This
          // avoids creating more than one copy of the enum when one
          // of the enum's variants refers to the enum itself.
          let repr = adt::represent_type(cx, t);
          let tps = substs.types.get_slice(subst::TypeSpace);
          let name = llvm_type_name(cx, def.did, tps);
          adt::incomplete_type_of(cx, &*repr, &name[..])
      }
      ty::TyClosure(..) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache.
          let repr = adt::represent_type(cx, t);
          // Unboxed closures can have substitutions in all spaces
          // inherited from their environment, so we use entire
          // contents of the VecPerParamSpace to construct the llvm
          // name
          adt::incomplete_type_of(cx, &*repr, "closure")
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
                  let unsized_part = cx.tcx().struct_tail(ty);
                  let info_ty = match unsized_part.sty {
                      ty::TyStr | ty::TyArray(..) | ty::TySlice(_) => {
                          Type::uint_from_ty(cx, ast::TyUs)
                      }
                      ty::TyTrait(_) => Type::vtable_ptr(cx),
                      _ => panic!("Unexpected type returned from \
                                   struct_tail: {:?} for ty={:?}",
                                  unsized_part, ty)
                  };
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

      ty::TyBareFn(..) => {
          type_of_fn_from_ty(cx, t).ptr_to()
      }
      ty::TyTuple(ref tys) if tys.is_empty() => Type::nil(cx),
      ty::TyTuple(..) => {
          let repr = adt::represent_type(cx, t);
          adt::type_of(cx, &*repr)
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
              let tps = substs.types.get_slice(subst::TypeSpace);
              let name = llvm_type_name(cx, def.did, tps);
              adt::incomplete_type_of(cx, &*repr, &name[..])
          }
      }

      ty::TyInfer(..) => cx.sess().bug("type_of with TyInfer"),
      ty::TyProjection(..) => cx.sess().bug("type_of with TyProjection"),
      ty::TyParam(..) => cx.sess().bug("type_of with ty_param"),
      ty::TyError(..) => cx.sess().bug("type_of with TyError"),
    };

    debug!("--> mapped t={:?} to llty={}",
            t,
            cx.tn().type_to_string(llty));

    cx.lltypes().borrow_mut().insert(t, llty);

    // If this was an enum or struct, fill in the type now.
    match t.sty {
        ty::TyEnum(..) | ty::TyStruct(..) | ty::TyClosure(..)
                if !t.is_simd() => {
            let repr = adt::represent_type(cx, t);
            adt::finish_type_of(cx, &*repr, &mut llty);
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
