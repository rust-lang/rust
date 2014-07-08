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
use middle::ty;
use util::ppaux::Repr;

use middle::trans::type_::Type;

use syntax::abi;
use syntax::ast;

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

pub fn type_of_rust_fn(cx: &CrateContext, has_env: bool,
                       inputs: &[ty::t], output: ty::t) -> Type {
    let mut atys: Vec<Type> = Vec::new();

    // Arg 0: Output pointer.
    // (if the output type is non-immediate)
    let use_out_pointer = return_uses_outptr(cx, output);
    let lloutputtype = arg_type_of(cx, output);
    if use_out_pointer {
        atys.push(lloutputtype.ptr_to());
    }

    // Arg 1: Environment
    if has_env {
        atys.push(Type::i8p(cx));
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
            type_of_rust_fn(cx, true, f.sig.inputs.as_slice(), f.sig.output)
        }
        ty::ty_bare_fn(ref f) => {
            if f.abi == abi::Rust || f.abi == abi::RustIntrinsic {
                type_of_rust_fn(cx,
                                false,
                                f.sig.inputs.as_slice(),
                                f.sig.output)
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
    match cx.llsizingtypes.borrow().find_copy(&t) {
        Some(t) => return t,
        None => ()
    }

    let llsizingty = match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot => Type::nil(cx),
        ty::ty_bool => Type::bool(cx),
        ty::ty_char => Type::char(cx),
        ty::ty_int(t) => Type::int_from_ty(cx, t),
        ty::ty_uint(t) => Type::uint_from_ty(cx, t),
        ty::ty_float(t) => Type::float_from_ty(cx, t),

        ty::ty_box(..) |
        ty::ty_ptr(..) => Type::i8p(cx),
        ty::ty_uniq(ty) => {
            match ty::get(ty).sty {
                ty::ty_trait(..) => Type::opaque_trait(cx),
                _ => Type::i8p(cx),
            }
        }
        ty::ty_rptr(_, mt) => {
            match ty::get(mt.ty).sty {
                ty::ty_vec(_, None) | ty::ty_str => {
                    Type::struct_(cx, [Type::i8p(cx), Type::i8p(cx)], false)
                }
                ty::ty_trait(..) => Type::opaque_trait(cx),
                _ => Type::i8p(cx),
            }
        }

        ty::ty_bare_fn(..) => Type::i8p(cx),
        ty::ty_closure(..) => Type::struct_(cx, [Type::i8p(cx), Type::i8p(cx)], false),

        ty::ty_vec(mt, Some(size)) => {
            Type::array(&sizing_type_of(cx, mt.ty), size as u64)
        }

        ty::ty_tup(..) | ty::ty_enum(..) => {
            let repr = adt::represent_type(cx, t);
            adt::sizing_type_of(cx, &*repr)
        }

        ty::ty_struct(..) => {
            if ty::type_is_simd(cx.tcx(), t) {
                let et = ty::simd_type(cx.tcx(), t);
                let n = ty::simd_size(cx.tcx(), t);
                Type::vector(&type_of(cx, et), n as u64)
            } else {
                let repr = adt::represent_type(cx, t);
                adt::sizing_type_of(cx, &*repr)
            }
        }

        ty::ty_infer(..) | ty::ty_param(..) |
        ty::ty_err(..) | ty::ty_vec(_, None) | ty::ty_str | ty::ty_trait(..) => {
            cx.sess().bug(format!("fictitious type {:?} in sizing_type_of()",
                                  ty::get(t).sty).as_slice())
        }
    };

    cx.llsizingtypes.borrow_mut().insert(t, llsizingty);
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
    // Check the cache.
    match cx.lltypes.borrow().find(&t) {
        Some(&llty) => return llty,
        None => ()
    }

    debug!("type_of {} {:?}", t.repr(cx.tcx()), t);

    // Replace any typedef'd types with their equivalent non-typedef
    // type. This ensures that all LLVM nominal types that contain
    // Rust types are defined as the same LLVM types.  If we don't do
    // this then, e.g. `Option<{myfield: bool}>` would be a different
    // type than `Option<myrec>`.
    let t_norm = ty::normalize_ty(cx.tcx(), t);

    if t != t_norm {
        let llty = type_of(cx, t_norm);
        debug!("--> normalized {} {:?} to {} {:?} llty={}",
                t.repr(cx.tcx()),
                t,
                t_norm.repr(cx.tcx()),
                t_norm,
                cx.tn.type_to_string(llty));
        cx.lltypes.borrow_mut().insert(t, llty);
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
      ty::ty_box(typ) => {
          Type::at_box(cx, type_of(cx, typ)).ptr_to()
      }
      ty::ty_uniq(typ) => {
          match ty::get(typ).sty {
              ty::ty_vec(mt, None) => Type::vec(cx, &type_of(cx, mt.ty)).ptr_to(),
              ty::ty_str => Type::vec(cx, &Type::i8(cx)).ptr_to(),
              ty::ty_trait(..) => Type::opaque_trait(cx),
              _ => type_of(cx, typ).ptr_to(),
          }
      }
      ty::ty_ptr(ref mt) => type_of(cx, mt.ty).ptr_to(),
      ty::ty_rptr(_, ref mt) => {
          match ty::get(mt.ty).sty {
              ty::ty_vec(mt, None) => {
                  let p_ty = type_of(cx, mt.ty).ptr_to();
                  let u_ty = Type::uint_from_ty(cx, ast::TyU);
                  Type::struct_(cx, [p_ty, u_ty], false)
              }
              ty::ty_str => {
                  // This means we get a nicer name in the output
                  cx.tn.find_type("str_slice").unwrap()
              }
              ty::ty_trait(..) => Type::opaque_trait(cx),
              _ => type_of(cx, mt.ty).ptr_to(),
          }
      }

      ty::ty_vec(ref mt, Some(n)) => {
          Type::array(&type_of(cx, mt.ty), n as u64)
      }

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
              let et = ty::simd_type(cx.tcx(), t);
              let n = ty::simd_size(cx.tcx(), t);
              Type::vector(&type_of(cx, et), n as u64)
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

      ty::ty_vec(_, None) => cx.sess().bug("type_of with unsized ty_vec"),
      ty::ty_str => cx.sess().bug("type_of with unsized (bare) ty_str"),
      ty::ty_trait(..) => cx.sess().bug("type_of with unsized ty_trait"),
      ty::ty_infer(..) => cx.sess().bug("type_of with ty_infer"),
      ty::ty_param(..) => cx.sess().bug("type_of with ty_param"),
      ty::ty_err(..) => cx.sess().bug("type_of with ty_err")
    };

    debug!("--> mapped t={} {:?} to llty={}",
            t.repr(cx.tcx()),
            t,
            cx.tn.type_to_string(llty));

    cx.lltypes.borrow_mut().insert(t, llty);

    // If this was an enum or struct, fill in the type now.
    match ty::get(t).sty {
        ty::ty_enum(..) | ty::ty_struct(..) if !ty::type_is_simd(cx.tcx(), t) => {
            let repr = adt::represent_type(cx, t);
            adt::finish_type_of(cx, &*repr, &mut llty);
        }
        _ => ()
    }

    return llty;
}

// Want refinements! (Or case classes, I guess
pub enum named_ty { a_struct, an_enum }

pub fn llvm_type_name(cx: &CrateContext,
                      what: named_ty,
                      did: ast::DefId,
                      tps: &[ty::t])
                      -> String
{
    let name = match what {
        a_struct => { "struct" }
        an_enum => { "enum" }
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
