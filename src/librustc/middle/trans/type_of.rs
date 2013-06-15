// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use lib::llvm::llvm;
use lib::llvm::{TypeRef};
use middle::trans::adt;
use middle::trans::base;
use middle::trans::common::*;
use middle::trans::common;
use middle::ty;
use util::ppaux;

use syntax::ast;

pub fn arg_is_indirect(_: @CrateContext, arg_ty: &ty::t) -> bool {
    !ty::type_is_immediate(*arg_ty)
}

pub fn type_of_explicit_arg(ccx: @CrateContext, arg_ty: &ty::t) -> TypeRef {
    let llty = type_of(ccx, *arg_ty);
    if arg_is_indirect(ccx, arg_ty) {T_ptr(llty)} else {llty}
}

pub fn type_of_explicit_args(ccx: @CrateContext,
                             inputs: &[ty::t]) -> ~[TypeRef] {
    inputs.map(|arg_ty| type_of_explicit_arg(ccx, arg_ty))
}

pub fn type_of_fn(cx: @CrateContext, inputs: &[ty::t], output: ty::t)
               -> TypeRef {
    unsafe {
        let mut atys: ~[TypeRef] = ~[];

        // Arg 0: Output pointer.
        // (if the output type is non-immediate)
        let output_is_immediate = ty::type_is_immediate(output);
        let lloutputtype = type_of(cx, output);
        if !output_is_immediate {
            atys.push(T_ptr(lloutputtype));
        }

        // Arg 1: Environment
        atys.push(T_opaque_box_ptr(cx));

        // ... then explicit args.
        atys.push_all(type_of_explicit_args(cx, inputs));

        // Use the output as the actual return value if it's immediate.
        if output_is_immediate {
            T_fn(atys, lloutputtype)
        } else {
            T_fn(atys, llvm::LLVMVoidTypeInContext(cx.llcx))
        }
    }
}

// Given a function type and a count of ty params, construct an llvm type
pub fn type_of_fn_from_ty(cx: @CrateContext, fty: ty::t) -> TypeRef {
    match ty::get(fty).sty {
        ty::ty_closure(ref f) => type_of_fn(cx, f.sig.inputs, f.sig.output),
        ty::ty_bare_fn(ref f) => type_of_fn(cx, f.sig.inputs, f.sig.output),
        _ => {
            cx.sess.bug("type_of_fn_from_ty given non-closure, non-bare-fn")
        }
    }
}

pub fn type_of_non_gc_box(cx: @CrateContext, t: ty::t) -> TypeRef {
    assert!(!ty::type_needs_infer(t));

    let t_norm = ty::normalize_ty(cx.tcx, t);
    if t != t_norm {
        type_of_non_gc_box(cx, t_norm)
    } else {
        match ty::get(t).sty {
          ty::ty_box(mt) => {
            T_ptr(T_box(cx, type_of(cx, mt.ty)))
          }
          ty::ty_uniq(mt) => {
            T_ptr(T_unique(cx, type_of(cx, mt.ty)))
          }
          _ => {
            cx.sess.bug("non-box in type_of_non_gc_box");
          }
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
//     recursive types. For example, `static_size_of_enum()` relies on this
//     behavior.

pub fn sizing_type_of(cx: @CrateContext, t: ty::t) -> TypeRef {
    match cx.llsizingtypes.find(&t) {
        Some(t) => return *t,
        None => ()
    }

    let llsizingty = match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot => T_nil(),
        ty::ty_bool => T_bool(),
        ty::ty_int(t) => T_int_ty(cx, t),
        ty::ty_uint(t) => T_uint_ty(cx, t),
        ty::ty_float(t) => T_float_ty(cx, t),

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
        ty::ty_opaque_closure_ptr(*) => T_ptr(T_i8()),

        ty::ty_estr(ty::vstore_slice(*)) |
        ty::ty_evec(_, ty::vstore_slice(*)) => {
            T_struct([T_ptr(T_i8()), T_ptr(T_i8())], false)
        }

        ty::ty_bare_fn(*) => T_ptr(T_i8()),
        ty::ty_closure(*) => T_struct([T_ptr(T_i8()), T_ptr(T_i8())], false),
        ty::ty_trait(_, _, store, _) => T_opaque_trait(cx, store),

        ty::ty_estr(ty::vstore_fixed(size)) => T_array(T_i8(), size),
        ty::ty_evec(mt, ty::vstore_fixed(size)) => {
            T_array(sizing_type_of(cx, mt.ty), size)
        }

        ty::ty_unboxed_vec(mt) => T_vec(cx, sizing_type_of(cx, mt.ty)),

        ty::ty_tup(*) | ty::ty_enum(*) => {
            let repr = adt::represent_type(cx, t);
            T_struct(adt::sizing_fields_of(cx, repr), false)
        }

        ty::ty_struct(did, _) => {
            if ty::type_is_simd(cx.tcx, t) {
                let et = ty::simd_type(cx.tcx, t);
                let n = ty::simd_size(cx.tcx, t);
                T_vector(type_of(cx, et), n)
            } else {
                let repr = adt::represent_type(cx, t);
                let packed = ty::lookup_packed(cx.tcx, did);
                T_struct(adt::sizing_fields_of(cx, repr), packed)
            }
        }

        ty::ty_self(_) | ty::ty_infer(*) | ty::ty_param(*) | ty::ty_err(*) => {
            cx.tcx.sess.bug(
                fmt!("fictitious type %? in sizing_type_of()",
                     ty::get(t).sty))
        }
    };

    cx.llsizingtypes.insert(t, llsizingty);
    llsizingty
}

// NB: If you update this, be sure to update `sizing_type_of()` as well.
pub fn type_of(cx: @CrateContext, t: ty::t) -> TypeRef {
    debug!("type_of %?: %?", t, ty::get(t));

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

    let llty = match ty::get(t).sty {
      ty::ty_nil | ty::ty_bot => T_nil(),
      ty::ty_bool => T_bool(),
      ty::ty_int(t) => T_int_ty(cx, t),
      ty::ty_uint(t) => T_uint_ty(cx, t),
      ty::ty_float(t) => T_float_ty(cx, t),
      ty::ty_estr(ty::vstore_uniq) => {
        T_unique_ptr(T_unique(cx, T_vec(cx, T_i8())))
      }
      ty::ty_enum(did, ref substs) => {
        // Only create the named struct, but don't fill it in. We
        // fill it in *after* placing it into the type cache. This
        // avoids creating more than one copy of the enum when one
        // of the enum's variants refers to the enum itself.

        common::T_named_struct(llvm_type_name(cx,
                                              an_enum,
                                              did,
                                              substs.tps))
      }
      ty::ty_estr(ty::vstore_box) => {
        T_box_ptr(T_box(cx, T_vec(cx, T_i8())))
      }
      ty::ty_evec(ref mt, ty::vstore_box) => {
        T_box_ptr(T_box(cx, T_vec(cx, type_of(cx, mt.ty))))
      }
      ty::ty_box(ref mt) => T_box_ptr(T_box(cx, type_of(cx, mt.ty))),
      ty::ty_opaque_box => T_box_ptr(T_box(cx, T_i8())),
      ty::ty_uniq(ref mt) => T_unique_ptr(T_unique(cx, type_of(cx, mt.ty))),
      ty::ty_evec(ref mt, ty::vstore_uniq) => {
        T_unique_ptr(T_unique(cx, T_vec(cx, type_of(cx, mt.ty))))
      }
      ty::ty_unboxed_vec(ref mt) => {
        T_vec(cx, type_of(cx, mt.ty))
      }
      ty::ty_ptr(ref mt) => T_ptr(type_of(cx, mt.ty)),
      ty::ty_rptr(_, ref mt) => T_ptr(type_of(cx, mt.ty)),

      ty::ty_evec(ref mt, ty::vstore_slice(_)) => {
        T_struct([T_ptr(type_of(cx, mt.ty)), T_uint_ty(cx, ast::ty_u)], false)
      }

      ty::ty_estr(ty::vstore_slice(_)) => {
        T_struct([T_ptr(T_i8()), T_uint_ty(cx, ast::ty_u)], false)
      }

      ty::ty_estr(ty::vstore_fixed(n)) => {
        T_array(T_i8(), n + 1u /* +1 for trailing null */)
      }

      ty::ty_evec(ref mt, ty::vstore_fixed(n)) => {
        T_array(type_of(cx, mt.ty), n)
      }

      ty::ty_bare_fn(_) => T_ptr(type_of_fn_from_ty(cx, t)),
      ty::ty_closure(_) => T_fn_pair(cx, type_of_fn_from_ty(cx, t)),
      ty::ty_trait(_, _, store, _) => T_opaque_trait(cx, store),
      ty::ty_type => T_ptr(cx.tydesc_type),
      ty::ty_tup(*) => {
          let repr = adt::represent_type(cx, t);
          T_struct(adt::fields_of(cx, repr), false)
      }
      ty::ty_opaque_closure_ptr(_) => T_opaque_box_ptr(cx),
      ty::ty_struct(did, ref substs) => {
        if ty::type_is_simd(cx.tcx, t) {
          let et = ty::simd_type(cx.tcx, t);
          let n = ty::simd_size(cx.tcx, t);
          T_vector(type_of(cx, et), n)
        } else {
          // Only create the named struct, but don't fill it in. We fill it
          // in *after* placing it into the type cache. This prevents
          // infinite recursion with recursive struct types.
          T_named_struct(llvm_type_name(cx,
                                        a_struct,
                                        did,
                                        substs.tps))
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
          common::set_struct_body(llty, adt::fields_of(cx, repr),
                                  false);
      }

      ty::ty_struct(did, _) => {
        if !ty::type_is_simd(cx.tcx, t) {
          let repr = adt::represent_type(cx, t);
          let packed = ty::lookup_packed(cx.tcx, did);
          common::set_struct_body(llty, adt::fields_of(cx, repr),
                                  packed);
        }
      }
      _ => ()
    }

    return llty;
}

// Want refinements! (Or case classes, I guess
pub enum named_ty { a_struct, an_enum }

pub fn llvm_type_name(cx: @CrateContext,
                      what: named_ty,
                      did: ast::def_id,
                      tps: &[ty::t]) -> ~str {
    let name = match what {
        a_struct => { "~struct" }
        an_enum => { "~enum" }
    };
    return fmt!(
        "%s %s[#%d]",
          name,
        ppaux::parameterized(
            cx.tcx,
            ty::item_path_str(cx.tcx, did),
            None,
            tps),
        did.crate
    );
}

pub fn type_of_dtor(ccx: @CrateContext, self_ty: ty::t) -> TypeRef {
    T_fn([T_ptr(type_of(ccx, self_ty))] /* self */, T_nil())
}

pub fn type_of_rooted(ccx: @CrateContext, t: ty::t) -> TypeRef {
    let addrspace = base::get_tydesc(ccx, t).addrspace;
    debug!("type_of_rooted %s in addrspace %u",
           ppaux::ty_to_str(ccx.tcx, t), addrspace as uint);
    return T_root(type_of(ccx, t), addrspace);
}

pub fn type_of_glue_fn(ccx: @CrateContext, t: ty::t) -> TypeRef {
    let tydescpp = T_ptr(T_ptr(ccx.tydesc_type));
    let llty = T_ptr(type_of(ccx, t));
    return T_fn([T_ptr(T_nil()), tydescpp, llty], T_nil());
}
