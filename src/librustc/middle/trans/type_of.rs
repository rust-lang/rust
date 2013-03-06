// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use lib::llvm::llvm;
use lib::llvm::{TypeRef};
use middle::trans::base;
use middle::trans::common::*;
use middle::trans::common;
use middle::trans::machine;
use middle::ty;
use util::ppaux;

use core::option::None;
use core::vec;
use syntax::ast;

pub fn type_of_explicit_arg(ccx: @CrateContext, arg: ty::arg) -> TypeRef {
    let llty = type_of(ccx, arg.ty);
    match ty::resolved_mode(ccx.tcx, arg.mode) {
        ast::by_val => llty,
        ast::by_copy => {
            if ty::type_is_immediate(arg.ty) {
                llty
            } else {
                T_ptr(llty)
            }
        }
        _ => T_ptr(llty)
    }
}

pub fn type_of_explicit_args(ccx: @CrateContext,
                             inputs: &[ty::arg]) -> ~[TypeRef] {
    inputs.map(|arg| type_of_explicit_arg(ccx, *arg))
}

pub fn type_of_fn(cx: @CrateContext, inputs: &[ty::arg],
                  output: ty::t) -> TypeRef {
    unsafe {
        let mut atys: ~[TypeRef] = ~[];

        // Arg 0: Output pointer.
        atys.push(T_ptr(type_of(cx, output)));

        // Arg 1: Environment
        atys.push(T_opaque_box_ptr(cx));

        // ... then explicit args.
        atys.push_all(type_of_explicit_args(cx, inputs));
        return T_fn(atys, llvm::LLVMVoidType());
    }
}

// Given a function type and a count of ty params, construct an llvm type
pub fn type_of_fn_from_ty(cx: @CrateContext, fty: ty::t) -> TypeRef {
    match ty::get(fty).sty {
        ty::ty_closure(ref f) => type_of_fn(cx, f.sig.inputs, f.sig.output),
        ty::ty_bare_fn(ref f) => type_of_fn(cx, f.sig.inputs, f.sig.output),
        _ => {
            cx.sess.bug(~"type_of_fn_from_ty given non-closure, non-bare-fn")
        }
    }
}

pub fn type_of_non_gc_box(cx: @CrateContext, t: ty::t) -> TypeRef {
    assert !ty::type_needs_infer(t);

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
            cx.sess.bug(~"non-box in type_of_non_gc_box");
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
    if cx.llsizingtypes.contains_key(&t) {
        return cx.llsizingtypes.get(&t);
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
            T_struct(~[T_ptr(T_i8()), T_ptr(T_i8())])
        }

        ty::ty_bare_fn(*) => T_ptr(T_i8()),
        ty::ty_closure(*) => T_struct(~[T_ptr(T_i8()), T_ptr(T_i8())]),
        ty::ty_trait(_, _, vstore) => T_opaque_trait(cx, vstore),

        ty::ty_estr(ty::vstore_fixed(size)) => T_array(T_i8(), size),
        ty::ty_evec(mt, ty::vstore_fixed(size)) => {
            T_array(sizing_type_of(cx, mt.ty), size)
        }

        ty::ty_unboxed_vec(mt) => T_vec(cx, sizing_type_of(cx, mt.ty)),

        ty::ty_tup(ref elems) => {
            T_struct(elems.map(|&t| sizing_type_of(cx, t)))
        }

        ty::ty_struct(def_id, ref substs) => {
            let fields = ty::lookup_struct_fields(cx.tcx, def_id);
            let lltype = T_struct(fields.map(|field| {
                let field_type = ty::lookup_field_type(cx.tcx,
                                                       def_id,
                                                       field.id,
                                                       substs);
                sizing_type_of(cx, field_type)
            }));
            if ty::ty_dtor(cx.tcx, def_id).is_present() {
                T_struct(~[lltype, T_i8()])
            } else {
                lltype
            }
        }

        ty::ty_enum(def_id, _) => T_struct(enum_body_types(cx, def_id, t)),

        ty::ty_self | ty::ty_infer(*) | ty::ty_param(*) | ty::ty_err(*) => {
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
    if cx.lltypes.contains_key(&t) { return cx.lltypes.get(&t); }

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

    // XXX: This is a terrible terrible copy.
    let llty = match /*bad*/copy ty::get(t).sty {
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
                                              /*bad*/copy substs.tps))
      }
      ty::ty_estr(ty::vstore_box) => {
        T_box_ptr(T_box(cx, T_vec(cx, T_i8())))
      }
      ty::ty_evec(mt, ty::vstore_box) => {
        T_box_ptr(T_box(cx, T_vec(cx, type_of(cx, mt.ty))))
      }
      ty::ty_box(mt) => T_box_ptr(T_box(cx, type_of(cx, mt.ty))),
      ty::ty_opaque_box => T_box_ptr(T_box(cx, T_i8())),
      ty::ty_uniq(mt) => T_unique_ptr(T_unique(cx, type_of(cx, mt.ty))),
      ty::ty_evec(mt, ty::vstore_uniq) => {
        T_unique_ptr(T_unique(cx, T_vec(cx, type_of(cx, mt.ty))))
      }
      ty::ty_unboxed_vec(mt) => {
        T_vec(cx, type_of(cx, mt.ty))
      }
      ty::ty_ptr(mt) => T_ptr(type_of(cx, mt.ty)),
      ty::ty_rptr(_, mt) => T_ptr(type_of(cx, mt.ty)),

      ty::ty_evec(mt, ty::vstore_slice(_)) => {
        T_struct(~[T_ptr(type_of(cx, mt.ty)),
                   T_uint_ty(cx, ast::ty_u)])
      }

      ty::ty_estr(ty::vstore_slice(_)) => {
        T_struct(~[T_ptr(T_i8()),
                   T_uint_ty(cx, ast::ty_u)])
      }

      ty::ty_estr(ty::vstore_fixed(n)) => {
        T_array(T_i8(), n + 1u /* +1 for trailing null */)
      }

      ty::ty_evec(mt, ty::vstore_fixed(n)) => {
        T_array(type_of(cx, mt.ty), n)
      }

      ty::ty_bare_fn(_) => T_ptr(type_of_fn_from_ty(cx, t)),
      ty::ty_closure(_) => T_fn_pair(cx, type_of_fn_from_ty(cx, t)),
      ty::ty_trait(_, _, vstore) => T_opaque_trait(cx, vstore),
      ty::ty_type => T_ptr(cx.tydesc_type),
      ty::ty_tup(elts) => {
        let mut tys = ~[];
        for vec::each(elts) |elt| {
            tys.push(type_of(cx, *elt));
        }
        T_struct(tys)
      }
      ty::ty_opaque_closure_ptr(_) => T_opaque_box_ptr(cx),
      ty::ty_struct(did, ref substs) => {
        // Only create the named struct, but don't fill it in. We fill it
        // in *after* placing it into the type cache. This prevents
        // infinite recursion with recursive struct types.

        common::T_named_struct(llvm_type_name(cx,
                                              a_struct,
                                              did,
                                              /*bad*/ copy substs.tps))
      }
      ty::ty_self => cx.tcx.sess.unimpl(~"type_of: ty_self"),
      ty::ty_infer(*) => cx.tcx.sess.bug(~"type_of with ty_infer"),
      ty::ty_param(*) => cx.tcx.sess.bug(~"type_of with ty_param"),
      ty::ty_err(*) => cx.tcx.sess.bug(~"type_of with ty_err")
    };

    cx.lltypes.insert(t, llty);

    // If this was an enum or struct, fill in the type now.
    match ty::get(t).sty {
      ty::ty_enum(did, _) => {
        fill_type_of_enum(cx, did, t, llty);
      }
      ty::ty_struct(did, ref substs) => {
        // Only instance vars are record fields at runtime.
        let fields = ty::lookup_struct_fields(cx.tcx, did);
        let mut tys = do vec::map(fields) |f| {
            let t = ty::lookup_field_type(cx.tcx, did, f.id, substs);
            type_of(cx, t)
        };

        // include a byte flag if there is a dtor so that we know when we've
        // been dropped
        if ty::ty_dtor(cx.tcx, did).is_present() {
            common::set_struct_body(llty, ~[T_struct(tys), T_i8()]);
        } else {
            common::set_struct_body(llty, ~[T_struct(tys)]);
        }
      }
      _ => ()
    }

    return llty;
}

pub fn enum_body_types(cx: @CrateContext, did: ast::def_id, t: ty::t)
                    -> ~[TypeRef] {
    let univar = ty::enum_is_univariant(cx.tcx, did);
    if !univar {
        let size = machine::static_size_of_enum(cx, t);
        ~[T_enum_discrim(cx), T_array(T_i8(), size)]
    }
    else {
        // Use the actual fields, so we get the alignment right.
        match ty::get(t).sty {
            ty::ty_enum(_, ref substs) => {
                do ty::enum_variants(cx.tcx, did)[0].args.map |&field_ty| {
                    sizing_type_of(cx, ty::subst(cx.tcx, substs, field_ty))
                }
            }
            _ => cx.sess.bug(~"enum is not an enum")
        }
    }
}

pub fn fill_type_of_enum(cx: @CrateContext,
                         did: ast::def_id,
                         t: ty::t,
                         llty: TypeRef) {
    debug!("type_of_enum %?: %?", t, ty::get(t));
    common::set_struct_body(llty, enum_body_types(cx, did, t));
}

// Want refinements! (Or case classes, I guess
pub enum named_ty { a_struct, an_enum }

pub fn llvm_type_name(cx: @CrateContext,
                      what: named_ty,
                      did: ast::def_id,
                      tps: ~[ty::t]) -> ~str {
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
    unsafe {
        T_fn(~[T_ptr(type_of(ccx, ty::mk_nil(ccx.tcx))), // output pointer
               T_ptr(type_of(ccx, self_ty))],            // self arg
             llvm::LLVMVoidType())
    }
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
    return T_fn(~[T_ptr(T_nil()), T_ptr(T_nil()), tydescpp, llty],
                T_void());
}
