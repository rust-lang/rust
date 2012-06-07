import std::map::{hashmap,str_hash};
import driver::session::session;
import lib::llvm::{TypeRef, ValueRef};
import syntax::ast;
import back::abi;
import common::*;
import build::*;
import base::*;
import type_of::*;
import ast::def_id;
import util::ppaux::ty_to_str;

fn visit_ty_steps(bcx: block, t: ty::t,
                  step: fn(bcx: block,
                           tyname: str,
                           args: [ValueRef]) -> block,
                  sub: fn(bcx: block, t: ty::t) -> block) -> block {

    let ccx = bcx.ccx();

    alt ty::get(t).struct {
      ty::ty_bot { step(bcx, "visit_bot", []) }
      ty::ty_nil { step(bcx, "visit_nil", []) }
      ty::ty_bool { step(bcx, "visit_bool", []) }
      ty::ty_int(ast::ty_i) { step(bcx, "visit_int", []) }
      ty::ty_int(ast::ty_char) { step(bcx, "visit_char", []) }
      ty::ty_int(ast::ty_i8) { step(bcx, "visit_i8", []) }
      ty::ty_int(ast::ty_i16) { step(bcx, "visit_i16", []) }
      ty::ty_int(ast::ty_i32) { step(bcx, "visit_i32", []) }
      ty::ty_int(ast::ty_i64) { step(bcx, "visit_i64", []) }
      ty::ty_uint(ast::ty_u) { step(bcx, "visit_uint", []) }
      ty::ty_uint(ast::ty_u8) { step(bcx, "visit_u8", []) }
      ty::ty_uint(ast::ty_u16) { step(bcx, "visit_u16", []) }
      ty::ty_uint(ast::ty_u32) { step(bcx, "visit_u32", []) }
      ty::ty_uint(ast::ty_u64) { step(bcx, "visit_u64", []) }
      ty::ty_float(ast::ty_f) { step(bcx, "visit_float", []) }
      ty::ty_float(ast::ty_f32) { step(bcx, "visit_f32", []) }
      ty::ty_float(ast::ty_f64) { step(bcx, "visit_f64", []) }
      ty::ty_str { step(bcx, "visit_str", []) }

      ty::ty_vec(mt) {
        let bcx = step(bcx, "visit_vec_of",
                       [C_uint(ccx, mt.mutbl as uint)]);
        sub(bcx, mt.ty)
      }

      _ {
        // Ideally this would be an unimpl, but sadly we have
        // to pretend we can visit everything at this point.
        step(bcx, "visit_bot", [])
      }
    }
}

// Emit a sequence of calls to visit_ty::visit_foo
fn emit_calls_to_iface_visit_ty(bcx: block, t: ty::t,
                                visitor_val: ValueRef,
                                visitor_iid: def_id) -> block {
    let tcx = bcx.tcx();
    let methods = ty::iface_methods(tcx, visitor_iid);
    visit_ty_steps(bcx, t,
                   {|bcx, mth_name, args|
                       let mth_idx = option::get(ty::method_idx(mth_name,
                                                                *methods));
                       let mth_ty = ty::mk_fn(tcx, methods[mth_idx].fty);
                       let get_lval = {|bcx|
                           impl::trans_iface_callee(bcx, visitor_val,
                                                    mth_ty, mth_idx)
                       };
                       trans_call_inner(bcx, none, mth_ty, ty::mk_bool(tcx),
                                        get_lval, arg_vals(args), ignore)
                   },
                   {|bcx, t_sub|
                       call_tydesc_glue(bcx, visitor_val, t_sub,
                                        abi::tydesc_field_visit_glue)})
}
