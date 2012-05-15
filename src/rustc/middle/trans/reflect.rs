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

fn visit_ty_steps<T>(bcx: block, t: ty::t,
                     step: fn(tyname: str, args: [ValueRef]) -> T) -> T {
    alt ty::get(t).struct {
      ty::ty_bot { step("visit_bot", []) }
      ty::ty_nil { step("visit_nil", []) }
      ty::ty_bool { step("visit_bool", []) }
      ty::ty_int(ast::ty_i) { step("visit_int", []) }
      ty::ty_int(ast::ty_char) { step("visit_char", []) }
          ty::ty_int(ast::ty_i8) { step("visit_i8", []) }
          ty::ty_int(ast::ty_i16) { step("visit_i16", []) }
          ty::ty_int(ast::ty_i32) { step("visit_i32", []) }
          ty::ty_int(ast::ty_i64) { step("visit_i64", []) }
          ty::ty_uint(ast::ty_u) { step("visit_uint", []) }
          ty::ty_uint(ast::ty_u8) { step("visit_u8", []) }
          ty::ty_uint(ast::ty_u16) { step("visit_u16", []) }
          ty::ty_uint(ast::ty_u32) { step("visit_u32", []) }
          ty::ty_uint(ast::ty_u64) { step("visit_u64", []) }
          ty::ty_float(ast::ty_f) { step("visit_float", []) }
          ty::ty_float(ast::ty_f32) { step("visit_f32", []) }
          ty::ty_float(ast::ty_f64) { step("visit_f64", []) }
          ty::ty_str { step("visit_str", []) }
      _ {
        bcx.sess().unimpl("trans::reflect::visit_ty_args on "
                          + ty_to_str(bcx.ccx().tcx, t));
      }
    }
}

// Emit a sequence of calls to visit_ty::visit_foo
fn emit_calls_to_iface_visit_ty(bcx: block, t: ty::t,
                                visitor_val: ValueRef,
                                visitor_iid: def_id) -> block {
    let tcx = bcx.tcx();
    visit_ty_steps(bcx, t) {|mth_name, args|
        let methods = ty::iface_methods(tcx, visitor_iid);
        let mth_idx = option::get(ty::method_idx(mth_name, *methods));
        let mth_ty = ty::mk_fn(tcx, methods[mth_idx].fty);
        let get_lval = {|bcx|
            impl::trans_iface_callee(bcx, visitor_val, mth_ty, mth_idx)
        };
        trans_call_inner(bcx, none, mth_ty, ty::mk_bool(tcx),
                         get_lval, arg_vals(args), ignore)
    }
}


fn find_intrinsic_ifaces(crate: @ast::crate)
    -> hashmap<str, ast::def_id> {

    let ifaces : hashmap<str, ast::def_id> = str_hash();

    // FIXME: hooking into the "intrinsic" root module is crude.
    // there ought to be a better approach. Attributes?

    for crate.node.module.items.each {|crate_item|
        if crate_item.ident == "intrinsic" {
            alt crate_item.node {
              ast::item_mod(m) {
                for m.items.each {|intrinsic_item|
                    alt intrinsic_item.node {
                      ast::item_iface(_, _, _) {
                        let def_id = { crate: ast::local_crate,
                                       node: intrinsic_item.id };
                        ifaces.insert(intrinsic_item.ident,
                                      def_id);
                      }
                      _ { }
                    }
                }
              }
              _ { }
            }
            break;
        }
    }

    // Assert whatever ifaces we are expecting to get from mod intrinsic.
    // assert ifaces.contains_key("visit_ty");

    ret ifaces;
}