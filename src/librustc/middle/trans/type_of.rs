// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::trans::adt;
use middle::trans::common::*;
use middle::trans::foreign;
use middle::ty;
use util::ppaux;

use middle::trans::type_::Type;

use syntax::ast;
use syntax::opt_vec;

pub fn arg_is_indirect(ccx: &mut CrateContext, arg_ty: ty::t) -> bool {
    !type_is_immediate(ccx, arg_ty)
}

pub fn return_uses_outptr(ccx: &mut CrateContext, ty: ty::t) -> bool {
    !type_is_immediate(ccx, ty)
}

pub fn type_of_explicit_arg(ccx: &mut CrateContext, arg_ty: ty::t) -> Type {
    let llty = type_of(ccx, arg_ty);
    if arg_is_indirect(ccx, arg_ty) {
        llty.ptr_to()
    } else {
        llty
    }
}

pub fn type_of_explicit_args(ccx: &mut CrateContext,
                             inputs: &[ty::t]) -> ~[Type] {
    inputs.map(|&arg_ty| type_of_explicit_arg(ccx, arg_ty))
}

pub fn type_of_rust_fn(cx: &mut CrateContext,
                       inputs: &[ty::t],
                       output: ty::t) -> Type {
    let mut atys: ~[Type] = ~[];

    // Arg 0: Output pointer.
    // (if the output type is non-immediate)
    let use_out_pointer = return_uses_outptr(cx, output);
    let lloutputtype = type_of(cx, output);
    if use_out_pointer {
        atys.push(lloutputtype.ptr_to());
    }

    // Arg 1: Environment
    atys.push(Type::opaque_box(cx).ptr_to());

    // ... then explicit args.
    atys.push_all(type_of_explicit_args(cx, inputs));

    // Use the output as the actual return value if it's immediate.
    if !use_out_pointer && !ty::type_is_voidish(cx.tcx, output) {
        Type::func(atys, &lloutputtype)
    } else {
        Type::func(atys, &Type::void())
    }
}

// Given a function type and a count of ty params, construct an llvm type
pub fn type_of_fn_from_ty(cx: &mut CrateContext, fty: ty::t) -> Type {
    return match ty::get(fty).sty {
        ty::ty_closure(ref f) => {
            type_of_rust_fn(cx, f.sig.inputs, f.sig.output)
        }
        ty::ty_bare_fn(ref f) => {
            if f.abis.is_rust() || f.abis.is_intrinsic() {
                type_of_rust_fn(cx, f.sig.inputs, f.sig.output)
            } else {
                foreign::lltype_for_foreign_fn(cx, fty)
            }
        }
        _ => {
            cx.sess.bug("type_of_fn_from_ty given non-closure, non-bare-fn")
        }
    };
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

pub fn sizing_type_of(cx: &mut CrateContext, t: ty::t) -> Type {
    match cx.llsizingtypes.find_copy(&t) {
        Some(t) => return t,
        None => ()
    }

    let llsizingty = match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot => Type::nil(),
        ty::ty_bool => Type::bool(),
        ty::ty_char => Type::char(),
        ty::ty_int(t) => Type::int_from_ty(cx, t),
        ty::ty_uint(t) => Type::uint_from_ty(cx, t),
        ty::ty_float(t) => Type::float_from_ty(t),

        ty::ty_estr(ty::vstore_uniq) |
        ty::ty_estr(ty::vstore_box) |
        ty::ty_evec(_, ty::vstore_uniq) |
        ty::ty_evec(_, ty::vstore_box) |
        ty::ty_box(*) |
        ty::ty_opaque_box |
        ty::ty_uniq(*) |
        ty::ty_ptr(*) |
        ty::ty_rptr(*) |
        ty::ty_type |
        ty::ty_opaque_closure_ptr(*) => Type::i8p(),

        ty::ty_estr(ty::vstore_slice(*)) |
        ty::ty_evec(_, ty::vstore_slice(*)) => {
            Type::struct_([Type::i8p(), Type::i8p()], false)
        }

        ty::ty_bare_fn(*) => Type::i8p(),
        ty::ty_closure(*) => Type::struct_([Type::i8p(), Type::i8p()], false),
        ty::ty_trait(_, _, store, _, _) => Type::opaque_trait(cx, store),

        ty::ty_estr(ty::vstore_fixed(size)) => Type::array(&Type::i8(), size as u64),
        ty::ty_evec(mt, ty::vstore_fixed(size)) => {
            Type::array(&sizing_type_of(cx, mt.ty), size as u64)
        }

        ty::ty_unboxed_vec(mt) => {
            let sz_ty = sizing_type_of(cx, mt.ty);
            Type::vec(cx.sess.targ_cfg.arch, &sz_ty)
        }

        ty::ty_tup(*) | ty::ty_enum(*) => {
            let repr = adt::represent_type(cx, t);
            Type::struct_(adt::sizing_fields_of(cx, repr), false)
        }

        ty::ty_struct(did, _) => {
            if ty::type_is_simd(cx.tcx, t) {
                let et = ty::simd_type(cx.tcx, t);
                let n = ty::simd_size(cx.tcx, t);
                Type::vector(&type_of(cx, et), n as u64)
            } else {
                let repr = adt::represent_type(cx, t);
                let packed = ty::lookup_packed(cx.tcx, did);
                Type::struct_(adt::sizing_fields_of(cx, repr), packed)
            }
        }

        ty::ty_self(_) | ty::ty_infer(*) | ty::ty_param(*) | ty::ty_err(*) => {
            cx.tcx.sess.bug(format!("fictitious type {:?} in sizing_type_of()", ty::get(t).sty))
        }
    };

    cx.llsizingtypes.insert(t, llsizingty);
    llsizingty
}

// NB: If you update this, be sure to update `sizing_type_of()` as well.
pub fn type_of(cx: &mut CrateContext, t: ty::t) -> Type {
    debug2!("type_of {:?}: {:?}", t, ty::get(t));

    // Check the cache.
    match cx.lltypes.find(&t) {
        Some(&t) => return t,
        None => ()
    }

    // Replace any typedef'd types with their equivalent non-typedef
    // type. This ensures that all LLVM nominal types that contain
    // Rust types are defined as the same LLVM types.  If we don't do
    // this then, e.g. `Option<{myfield: bool}>` would be a different
    // type than `Option<myrec>`.
    let t_norm = ty::normalize_ty(cx.tcx, t);

    if t != t_norm {
        let llty = type_of(cx, t_norm);
        cx.lltypes.insert(t, llty);
        return llty;
    }

    let mut llty = match ty::get(t).sty {
      ty::ty_nil | ty::ty_bot => Type::nil(),
      ty::ty_bool => Type::bool(),
      ty::ty_char => Type::char(),
      ty::ty_int(t) => Type::int_from_ty(cx, t),
      ty::ty_uint(t) => Type::uint_from_ty(cx, t),
      ty::ty_float(t) => Type::float_from_ty(t),
      ty::ty_estr(ty::vstore_uniq) => {
        Type::vec(cx.sess.targ_cfg.arch, &Type::i8()).ptr_to()
      }
      ty::ty_enum(did, ref substs) => {
        // Only create the named struct, but don't fill it in. We
        // fill it in *after* placing it into the type cache. This
        // avoids creating more than one copy of the enum when one
        // of the enum's variants refers to the enum itself.

        Type::named_struct(llvm_type_name(cx, an_enum, did, substs.tps))
      }
      ty::ty_estr(ty::vstore_box) => {
        Type::box(cx, &Type::vec(cx.sess.targ_cfg.arch, &Type::i8())).ptr_to()
      }
      ty::ty_evec(ref mt, ty::vstore_box) => {
          let e_ty = type_of(cx, mt.ty);
          let v_ty = Type::vec(cx.sess.targ_cfg.arch, &e_ty);
          Type::box(cx, &v_ty).ptr_to()
      }
      ty::ty_box(ref mt) => {
          let ty = type_of(cx, mt.ty);
          Type::box(cx, &ty).ptr_to()
      }
      ty::ty_opaque_box => Type::opaque_box(cx).ptr_to(),
      ty::ty_uniq(ref mt) => {
          let ty = type_of(cx, mt.ty);
          if ty::type_contents(cx.tcx, mt.ty).contains_managed() {
              Type::unique(cx, &ty).ptr_to()
          } else {
              ty.ptr_to()
          }
      }
      ty::ty_evec(ref mt, ty::vstore_uniq) => {
          let ty = type_of(cx, mt.ty);
          let ty = Type::vec(cx.sess.targ_cfg.arch, &ty);
          if ty::type_contents(cx.tcx, mt.ty).contains_managed() {
              Type::unique(cx, &ty).ptr_to()
          } else {
              ty.ptr_to()
          }
      }
      ty::ty_unboxed_vec(ref mt) => {
          let ty = type_of(cx, mt.ty);
          Type::vec(cx.sess.targ_cfg.arch, &ty)
      }
      ty::ty_ptr(ref mt) => type_of(cx, mt.ty).ptr_to(),
      ty::ty_rptr(_, ref mt) => type_of(cx, mt.ty).ptr_to(),

      ty::ty_evec(ref mt, ty::vstore_slice(_)) => {
          let p_ty = type_of(cx, mt.ty).ptr_to();
          let u_ty = Type::uint_from_ty(cx, ast::ty_u);
          Type::struct_([p_ty, u_ty], false)
      }

      ty::ty_estr(ty::vstore_slice(_)) => {
          // This means we get a nicer name in the output
          cx.tn.find_type("str_slice").unwrap()
      }

      ty::ty_estr(ty::vstore_fixed(n)) => {
          Type::array(&Type::i8(), (n + 1u) as u64)
      }

      ty::ty_evec(ref mt, ty::vstore_fixed(n)) => {
          Type::array(&type_of(cx, mt.ty), n as u64)
      }

      ty::ty_bare_fn(_) => {
          type_of_fn_from_ty(cx, t).ptr_to()
      }
      ty::ty_closure(_) => {
          let ty = type_of_fn_from_ty(cx, t);
          Type::func_pair(cx, &ty)
      }
      ty::ty_trait(_, _, store, _, _) => Type::opaque_trait(cx, store),
      ty::ty_type => cx.tydesc_type.ptr_to(),
      ty::ty_tup(*) => {
          let repr = adt::represent_type(cx, t);
          Type::struct_(adt::fields_of(cx, repr), false)
      }
      ty::ty_opaque_closure_ptr(_) => Type::opaque_box(cx).ptr_to(),
      ty::ty_struct(did, ref substs) => {
          if ty::type_is_simd(cx.tcx, t) {
              let et = ty::simd_type(cx.tcx, t);
              let n = ty::simd_size(cx.tcx, t);
              Type::vector(&type_of(cx, et), n as u64)
          } else {
              // Only create the named struct, but don't fill it in. We fill it
              // in *after* placing it into the type cache. This prevents
              // infinite recursion with recursive struct types.
              Type::named_struct(llvm_type_name(cx, a_struct, did, substs.tps))
          }
      }
      ty::ty_self(*) => cx.tcx.sess.unimpl("type_of: ty_self"),
      ty::ty_infer(*) => cx.tcx.sess.bug("type_of with ty_infer"),
      ty::ty_param(*) => cx.tcx.sess.bug("type_of with ty_param"),
      ty::ty_err(*) => cx.tcx.sess.bug("type_of with ty_err")
    };

    cx.lltypes.insert(t, llty);

    // If this was an enum or struct, fill in the type now.
    match ty::get(t).sty {
      ty::ty_enum(*) => {
          let repr = adt::represent_type(cx, t);
          llty.set_struct_body(adt::fields_of(cx, repr), false);
      }

      ty::ty_struct(did, _) => {
        if !ty::type_is_simd(cx.tcx, t) {
          let repr = adt::represent_type(cx, t);
          let packed = ty::lookup_packed(cx.tcx, did);
          llty.set_struct_body(adt::fields_of(cx, repr), packed);
        }
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
                      tps: &[ty::t]) -> ~str {
    let name = match what {
        a_struct => { "struct" }
        an_enum => { "enum" }
    };
    let tstr = ppaux::parameterized(cx.tcx, ty::item_path_str(cx.tcx, did),
                                    &ty::NonerasedRegions(opt_vec::Empty), tps);
    if did.crate == 0 {
        format!("{}.{}", name, tstr)
    } else {
        format!("{}.{}[\\#{}]", name, tstr, did.crate)
    }
}

pub fn type_of_dtor(ccx: &mut CrateContext, self_ty: ty::t) -> Type {
    let self_ty = type_of(ccx, self_ty).ptr_to();
    Type::func([self_ty], &Type::void())
}
