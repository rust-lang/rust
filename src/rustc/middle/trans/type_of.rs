import common::*;
import lib::llvm::{TypeRef};
import syntax::ast;
import lib::llvm::llvm;
import driver::session::session;

import ty::*;

fn type_of_explicit_args(cx: crate_ctxt, inputs: [ty::arg]) -> [TypeRef] {
    vec::map(inputs) {|arg|
        let arg_ty = arg.ty;
        let llty = type_of(cx, arg_ty);
        alt ty::resolved_mode(cx.tcx, arg.mode) {
          ast::by_val { llty }
          _ { T_ptr(llty) }
        }
    }
}

fn type_of_fn(cx: crate_ctxt, inputs: [ty::arg],
              output: ty::t, params: [ty::param_bounds]) -> TypeRef {
    let atys: [TypeRef] = [];

    // Arg 0: Output pointer.
    atys += [T_ptr(type_of(cx, output))];

    // Arg 1: Environment
    atys += [T_opaque_box_ptr(cx)];

    // Args >2: ty params, if not acquired via capture...
    for bounds in params {
        atys += [T_ptr(cx.tydesc_type)];
        for bound in *bounds {
            alt bound {
              ty::bound_iface(_) { atys += [T_ptr(T_dict())]; }
              _ {}
            }
        }
    }
    // ... then explicit args.
    atys += type_of_explicit_args(cx, inputs);
    ret T_fn(atys, llvm::LLVMVoidType());
}

// Given a function type and a count of ty params, construct an llvm type
fn type_of_fn_from_ty(cx: crate_ctxt, fty: ty::t,
                      param_bounds: [ty::param_bounds]) -> TypeRef {
    type_of_fn(cx, ty::ty_fn_args(fty), ty::ty_fn_ret(fty), param_bounds)
}

fn type_of(cx: crate_ctxt, t: ty::t) -> TypeRef {
    assert !ty::type_has_vars(t);
    // Check the cache.

    if cx.lltypes.contains_key(t) { ret cx.lltypes.get(t); }
    let llty = alt ty::get(t).struct {
      ty::ty_nil | ty::ty_bot { T_nil() }
      ty::ty_bool { T_bool() }
      ty::ty_int(t) { T_int_ty(cx, t) }
      ty::ty_uint(t) { T_uint_ty(cx, t) }
      ty::ty_float(t) { T_float_ty(cx, t) }
      ty::ty_str { T_ptr(T_vec(cx, T_i8())) }
      ty::ty_enum(did, _) { type_of_enum(cx, did, t) }
      ty::ty_box(mt) {
        let mt_ty = mt.ty;
        T_ptr(T_box(cx, type_of(cx, mt_ty))) }
      ty::ty_opaque_box { T_ptr(T_box(cx, T_i8())) }
      ty::ty_uniq(mt) {
        let mt_ty = mt.ty;
        T_ptr(type_of(cx, mt_ty)) }
      ty::ty_vec(mt) {
        let mt_ty = mt.ty;
        if ty::type_has_dynamic_size(cx.tcx, mt_ty) {
            T_ptr(cx.opaque_vec_type)
        } else {
            T_ptr(T_vec(cx, type_of(cx, mt_ty))) }
      }
      ty::ty_ptr(mt) {
        let mt_ty = mt.ty;
        T_ptr(type_of(cx, mt_ty)) }
      ty::ty_rec(fields) {
        let tys: [TypeRef] = [];
        for f: ty::field in fields {
            let mt_ty = f.mt.ty;
            tys += [type_of(cx, mt_ty)];
        }
        T_struct(tys)
      }
      ty::ty_fn(_) {
        T_fn_pair(cx, type_of_fn_from_ty(cx, t, []))
      }
      ty::ty_iface(_, _) { T_opaque_iface(cx) }
      ty::ty_res(_, sub, tps) {
        let sub1 = ty::substitute_type_params(cx.tcx, tps, sub);
        ret T_struct([T_i8(), type_of(cx, sub1)]);
      }
      ty::ty_param(_, _) { T_typaram(cx.tn) }
      ty::ty_send_type | ty::ty_type { T_ptr(cx.tydesc_type) }
      ty::ty_tup(elts) {
        let tys = [];
        for elt in elts {
            tys += [type_of(cx, elt)];
        }
        T_struct(tys)
      }
      ty::ty_opaque_closure_ptr(_) { T_opaque_box_ptr(cx) }
      ty::ty_constr(subt,_) { type_of(cx, subt) }
      ty::ty_class(did, _) {
        let tys: [TypeRef] = [];
        // TODO: only handles local classes
        let cls_items = lookup_class_items(cx.tcx, did);
        for ci in cls_items {
            // only instance vars are record fields at runtime
            alt ci.node.decl {
                ast::instance_var(_,_,_,_) {
                  let fty = type_of(cx, class_item_type(cx.tcx, ci));
                  tys += [fty];
                }
                _ {}
            }
        }
        T_struct(tys)
      }
      ty::ty_self(_) { cx.tcx.sess.unimpl("type_of: ty_self \
                         not implemented"); }
      ty::ty_var(_) { cx.tcx.sess.bug("type_of shouldn't see a ty_var"); }
    };
    cx.lltypes.insert(t, llty);
    ret llty;
}

fn type_of_enum(cx: crate_ctxt, did: ast::def_id, t: ty::t)
    -> TypeRef {
    let degen = (*ty::enum_variants(cx.tcx, did)).len() == 1u;
    if check type_has_static_size(cx, t) {
        let size = shape::static_size_of_enum(cx, t);
        if !degen { T_enum(cx, size) }
        else if size == 0u { T_struct([T_enum_variant(cx)]) }
        else { T_array(T_i8(), size) }
    }
    else {
        if degen { T_struct([T_enum_variant(cx)]) }
        else { T_opaque_enum(cx) }
    }
}

fn type_of_ty_param_bounds_and_ty
    (ccx: crate_ctxt, tpt: ty::ty_param_bounds_and_ty) -> TypeRef {
    let t = tpt.ty;
    alt ty::get(t).struct {
      ty::ty_fn(_) {
        ret type_of_fn_from_ty(ccx, t, *tpt.bounds);
      }
      _ {
        // fall through
      }
    }
    type_of(ccx, t)
}

fn type_of_or_i8(ccx: crate_ctxt, typ: ty::t) -> TypeRef {
    if check type_has_static_size(ccx, typ) {
        type_of(ccx, typ)
    } else { T_i8() }
}
