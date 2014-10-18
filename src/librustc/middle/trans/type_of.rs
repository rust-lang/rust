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

use middle::subst;
use middle::trans::adt;
use middle::trans::common::*;
use middle::trans::foreign;
use middle::trans::machine;
use middle::ty;
use util::ppaux;
use util::ppaux::Repr;

use middle::trans::type_::Type;

use syntax::abi;
use syntax::ast;

use std::num::CheckedMul;

// LLVM doesn't like objects that are too big. Issue #17913
fn ensure_array_fits_in_address_space(ccx: &CrateContext,
                                      llet: Type,
                                      size: machine::llsize,
                                      scapegoat: ty::t) {
    let esz = machine::llsize_of_alloc(ccx, llet);
    match esz.checked_mul(&size) {
        Some(n) if n < ccx.max_obj_size() => {}
        _ => { ccx.report_overbig_object(scapegoat) }
    }
}

pub fn arg_is_indirect(ccx: &CrateContext, arg_ty: ty::t) -> bool {
    !type_is_immediate(ccx, arg_ty)
}

pub fn return_uses_outptr(ccx: &CrateContext, ty: ty::t) -> bool {
    !type_is_immediate(ccx, ty)
}

pub fn type_of_explicit_arg(ccx: &CrateContext, arg_ty: ty::t) -> Type {
    let llty = arg_type_of(ccx, arg_ty);
    if arg_is_indirect(ccx, arg_ty) {
        llty.ptr_to()
    } else {
        llty
    }
}

/// Yields the types of the "real" arguments for this function. For most
/// functions, these are simply the types of the arguments. For functions with
/// the `RustCall` ABI, however, this untuples the arguments of the function.
fn untuple_arguments_if_necessary(ccx: &CrateContext,
                                  inputs: &[ty::t],
                                  abi: abi::Abi)
                                  -> Vec<ty::t> {
    if abi != abi::RustCall {
        return inputs.iter().map(|x| (*x).clone()).collect()
    }

    if inputs.len() == 0 {
        return Vec::new()
    }

    let mut result = Vec::new();
    for (i, &arg_prior_to_tuple) in inputs.iter().enumerate() {
        if i < inputs.len() - 1 {
            result.push(arg_prior_to_tuple);
        }
    }

    match ty::get(inputs[inputs.len() - 1]).sty {
        ty::ty_tup(ref tupled_arguments) => {
            debug!("untuple_arguments_if_necessary(): untupling arguments");
            for &tupled_argument in tupled_arguments.iter() {
                result.push(tupled_argument);
            }
        }
        ty::ty_nil => {}
        _ => {
            ccx.tcx().sess.bug("argument to function with \"rust-call\" ABI \
                                is neither a tuple nor unit")
        }
    }

    result
}

pub fn type_of_rust_fn(cx: &CrateContext,
                       llenvironment_type: Option<Type>,
                       inputs: &[ty::t],
                       output: ty::t,
                       abi: abi::Abi)
                       -> Type {
    let mut atys: Vec<Type> = Vec::new();

    // First, munge the inputs, if this has the `rust-call` ABI.
    let inputs = untuple_arguments_if_necessary(cx, inputs, abi);

    // Arg 0: Output pointer.
    // (if the output type is non-immediate)
    let use_out_pointer = return_uses_outptr(cx, output);
    let lloutputtype = arg_type_of(cx, output);
    if use_out_pointer {
        atys.push(lloutputtype.ptr_to());
    }

    // Arg 1: Environment
    match llenvironment_type {
        None => {}
        Some(llenvironment_type) => atys.push(llenvironment_type),
    }

    // ... then explicit args.
    let input_tys = inputs.iter().map(|&arg_ty| type_of_explicit_arg(cx, arg_ty));
    atys.extend(input_tys);

    // Use the output as the actual return value if it's immediate.
    if use_out_pointer || return_type_is_void(cx, output) {
        Type::func(atys.as_slice(), &Type::void(cx))
    } else {
        Type::func(atys.as_slice(), &lloutputtype)
    }
}

// Given a function type and a count of ty params, construct an llvm type
pub fn type_of_fn_from_ty(cx: &CrateContext, fty: ty::t) -> Type {
    match ty::get(fty).sty {
        ty::ty_closure(ref f) => {
            type_of_rust_fn(cx,
                            Some(Type::i8p(cx)),
                            f.sig.inputs.as_slice(),
                            f.sig.output,
                            f.abi)
        }
        ty::ty_bare_fn(ref f) => {
            if f.abi == abi::Rust || f.abi == abi::RustCall {
                type_of_rust_fn(cx,
                                None,
                                f.sig.inputs.as_slice(),
                                f.sig.output,
                                f.abi)
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

pub fn sizing_type_of(cx: &CrateContext, t: ty::t) -> Type {
    match cx.llsizingtypes().borrow().find_copy(&t) {
        Some(t) => return t,
        None => ()
    }

    let llsizingty = match ty::get(t).sty {
        _ if !ty::lltype_is_sized(cx.tcx(), t) => {
            cx.sess().bug(format!("trying to take the sizing type of {}, an unsized type",
                                  ppaux::ty_to_string(cx.tcx(), t)).as_slice())
        }

        ty::ty_nil | ty::ty_bot => Type::nil(cx),
        ty::ty_bool => Type::bool(cx),
        ty::ty_char => Type::char(cx),
        ty::ty_int(t) => Type::int_from_ty(cx, t),
        ty::ty_uint(t) => Type::uint_from_ty(cx, t),
        ty::ty_float(t) => Type::float_from_ty(cx, t),

        ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) | ty::ty_ptr(ty::mt{ty, ..}) => {
            if ty::type_is_sized(cx.tcx(), ty) {
                Type::i8p(cx)
            } else {
                Type::struct_(cx, [Type::i8p(cx), Type::i8p(cx)], false)
            }
        }

        ty::ty_bare_fn(..) => Type::i8p(cx),
        ty::ty_closure(..) => Type::struct_(cx, [Type::i8p(cx), Type::i8p(cx)], false),

        ty::ty_vec(ty, Some(size)) => {
            let llty = sizing_type_of(cx, ty);
            let size = size as u64;
            ensure_array_fits_in_address_space(cx, llty, size, t);
            Type::array(&llty, size)
        }

        ty::ty_tup(..) | ty::ty_enum(..) | ty::ty_unboxed_closure(..) => {
            let repr = adt::represent_type(cx, t);
            adt::sizing_type_of(cx, &*repr, false)
        }

        ty::ty_struct(..) => {
            if ty::type_is_simd(cx.tcx(), t) {
                let llet = type_of(cx, ty::simd_type(cx.tcx(), t));
                let n = ty::simd_size(cx.tcx(), t) as u64;
                ensure_array_fits_in_address_space(cx, llet, n, t);
                Type::vector(&llet, n)
            } else {
                let repr = adt::represent_type(cx, t);
                adt::sizing_type_of(cx, &*repr, false)
            }
        }

        ty::ty_open(_) => {
            Type::struct_(cx, [Type::i8p(cx), Type::i8p(cx)], false)
        }

        ty::ty_infer(..) | ty::ty_param(..) | ty::ty_err(..) => {
            cx.sess().bug(format!("fictitious type {} in sizing_type_of()",
                                  ppaux::ty_to_string(cx.tcx(), t)).as_slice())
        }
        ty::ty_vec(_, None) | ty::ty_trait(..) | ty::ty_str => fail!("unreachable")
    };

    cx.llsizingtypes().borrow_mut().insert(t, llsizingty);
    llsizingty
}

pub fn arg_type_of(cx: &CrateContext, t: ty::t) -> Type {
    if ty::type_is_bool(t) {
        Type::i1(cx)
    } else {
        type_of(cx, t)
    }
}

// NB: If you update this, be sure to update `sizing_type_of()` as well.
pub fn type_of(cx: &CrateContext, t: ty::t) -> Type {
    fn type_of_unsize_info(cx: &CrateContext, t: ty::t) -> Type {
        // It is possible to end up here with a sized type. This happens with a
        // struct which might be unsized, but is monomorphised to a sized type.
        // In this case we'll fake a fat pointer with no unsize info (we use 0).
        // However, its still a fat pointer, so we need some type use.
        if ty::type_is_sized(cx.tcx(), t) {
            return Type::i8p(cx);
        }

        match ty::get(ty::unsized_part_of_type(cx.tcx(), t)).sty {
            ty::ty_str | ty::ty_vec(..) => Type::uint_from_ty(cx, ast::TyU),
            ty::ty_trait(_) => Type::vtable_ptr(cx),
            _ => fail!("Unexpected type returned from unsized_part_of_type : {}",
                       t.repr(cx.tcx()))
        }
    }

    // Check the cache.
    match cx.lltypes().borrow().find(&t) {
        Some(&llty) => return llty,
        None => ()
    }

    debug!("type_of {} {}", t.repr(cx.tcx()), ty::get(t).sty);

    // Replace any typedef'd types with their equivalent non-typedef
    // type. This ensures that all LLVM nominal types that contain
    // Rust types are defined as the same LLVM types.  If we don't do
    // this then, e.g. `Option<{myfield: bool}>` would be a different
    // type than `Option<myrec>`.
    let t_norm = ty::normalize_ty(cx.tcx(), t);

    if t != t_norm {
        let llty = type_of(cx, t_norm);
        debug!("--> normalized {} {} to {} {} llty={}",
                t.repr(cx.tcx()),
                t,
                t_norm.repr(cx.tcx()),
                t_norm,
                cx.tn().type_to_string(llty));
        cx.lltypes().borrow_mut().insert(t, llty);
        return llty;
    }

    let mut llty = match ty::get(t).sty {
      ty::ty_nil | ty::ty_bot => Type::nil(cx),
      ty::ty_bool => Type::bool(cx),
      ty::ty_char => Type::char(cx),
      ty::ty_int(t) => Type::int_from_ty(cx, t),
      ty::ty_uint(t) => Type::uint_from_ty(cx, t),
      ty::ty_float(t) => Type::float_from_ty(cx, t),
      ty::ty_enum(did, ref substs) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache. This
          // avoids creating more than one copy of the enum when one
          // of the enum's variants refers to the enum itself.
          let repr = adt::represent_type(cx, t);
          let tps = substs.types.get_slice(subst::TypeSpace);
          let name = llvm_type_name(cx, an_enum, did, tps);
          adt::incomplete_type_of(cx, &*repr, name.as_slice())
      }
      ty::ty_unboxed_closure(did, _) => {
          // Only create the named struct, but don't fill it in. We
          // fill it in *after* placing it into the type cache.
          let repr = adt::represent_type(cx, t);
          let name = llvm_type_name(cx, an_unboxed_closure, did, []);
          adt::incomplete_type_of(cx, &*repr, name.as_slice())
      }

      ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) | ty::ty_ptr(ty::mt{ty, ..}) => {
          match ty::get(ty).sty {
              ty::ty_str => {
                  // This means we get a nicer name in the output (str is always
                  // unsized).
                  cx.tn().find_type("str_slice").unwrap()
              }
              ty::ty_trait(..) => Type::opaque_trait(cx),
              _ if !ty::type_is_sized(cx.tcx(), ty) => {
                  let p_ty = type_of(cx, ty).ptr_to();
                  Type::struct_(cx, [p_ty, type_of_unsize_info(cx, ty)], false)
              }
              _ => type_of(cx, ty).ptr_to(),
          }
      }

      ty::ty_vec(ty, Some(size)) => {
          let size = size as u64;
          let llty = type_of(cx, ty);
          ensure_array_fits_in_address_space(cx, llty, size, t);
          Type::array(&llty, size)
      }
      ty::ty_vec(ty, None) => {
          type_of(cx, ty)
      }

      ty::ty_trait(..) => {
          Type::opaque_trait_data(cx)
      }

      ty::ty_str => Type::i8(cx),

      ty::ty_bare_fn(_) => {
          type_of_fn_from_ty(cx, t).ptr_to()
      }
      ty::ty_closure(_) => {
          let fn_ty = type_of_fn_from_ty(cx, t).ptr_to();
          Type::struct_(cx, [fn_ty, Type::i8p(cx)], false)
      }
      ty::ty_tup(..) => {
          let repr = adt::represent_type(cx, t);
          adt::type_of(cx, &*repr)
      }
      ty::ty_struct(did, ref substs) => {
          if ty::type_is_simd(cx.tcx(), t) {
              let llet = type_of(cx, ty::simd_type(cx.tcx(), t));
              let n = ty::simd_size(cx.tcx(), t) as u64;
              ensure_array_fits_in_address_space(cx, llet, n, t);
              Type::vector(&llet, n)
          } else {
              // Only create the named struct, but don't fill it in. We fill it
              // in *after* placing it into the type cache. This prevents
              // infinite recursion with recursive struct types.
              let repr = adt::represent_type(cx, t);
              let tps = substs.types.get_slice(subst::TypeSpace);
              let name = llvm_type_name(cx, a_struct, did, tps);
              adt::incomplete_type_of(cx, &*repr, name.as_slice())
          }
      }

      ty::ty_open(t) => match ty::get(t).sty {
          ty::ty_struct(..) => {
              let p_ty = type_of(cx, t).ptr_to();
              Type::struct_(cx, [p_ty, type_of_unsize_info(cx, t)], false)
          }
          ty::ty_vec(ty, None) => {
              let p_ty = type_of(cx, ty).ptr_to();
              Type::struct_(cx, [p_ty, type_of_unsize_info(cx, t)], false)
          }
          ty::ty_str => {
              let p_ty = Type::i8p(cx);
              Type::struct_(cx, [p_ty, type_of_unsize_info(cx, t)], false)
          }
          ty::ty_trait(..) => Type::opaque_trait(cx),
          _ => cx.sess().bug(format!("ty_open with sized type: {}",
                                     ppaux::ty_to_string(cx.tcx(), t)).as_slice())
      },

      ty::ty_infer(..) => cx.sess().bug("type_of with ty_infer"),
      ty::ty_param(..) => cx.sess().bug("type_of with ty_param"),
      ty::ty_err(..) => cx.sess().bug("type_of with ty_err"),
    };

    debug!("--> mapped t={} {} to llty={}",
            t.repr(cx.tcx()),
            t,
            cx.tn().type_to_string(llty));

    cx.lltypes().borrow_mut().insert(t, llty);

    // If this was an enum or struct, fill in the type now.
    match ty::get(t).sty {
        ty::ty_enum(..) | ty::ty_struct(..) | ty::ty_unboxed_closure(..)
                if !ty::type_is_simd(cx.tcx(), t) => {
            let repr = adt::represent_type(cx, t);
            adt::finish_type_of(cx, &*repr, &mut llty);
        }
        _ => ()
    }

    return llty;
}

pub fn align_of(cx: &CrateContext, t: ty::t) -> machine::llalign {
    let llty = sizing_type_of(cx, t);
    machine::llalign_of_min(cx, llty)
}

// Want refinements! (Or case classes, I guess
pub enum named_ty {
    a_struct,
    an_enum,
    an_unboxed_closure,
}

pub fn llvm_type_name(cx: &CrateContext,
                      what: named_ty,
                      did: ast::DefId,
                      tps: &[ty::t])
                      -> String
{
    let name = match what {
        a_struct => "struct",
        an_enum => "enum",
        an_unboxed_closure => return "closure".to_string(),
    };

    let base = ty::item_path_str(cx.tcx(), did);
    let strings: Vec<String> = tps.iter().map(|t| t.repr(cx.tcx())).collect();
    let tstr = format!("{}<{}>", base, strings);
    if did.krate == 0 {
        format!("{}.{}", name, tstr)
    } else {
        format!("{}.{}[{}{}]", name, tstr, "#", did.krate)
    }
}

pub fn type_of_dtor(ccx: &CrateContext, self_ty: ty::t) -> Type {
    let self_ty = type_of(ccx, self_ty).ptr_to();
    Type::func([self_ty], &Type::void(ccx))
}
