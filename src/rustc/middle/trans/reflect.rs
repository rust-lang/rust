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

enum reflector = {
    visitor_val: ValueRef,
    visitor_methods: @[ty::method],
    mut bcx: block
};

impl methods for reflector {

    fn c_uint(u: uint) -> ValueRef {
        let bcx = self.bcx;
        C_uint(bcx.ccx(), u)
    }

    fn visit(ty_name: str, args: [ValueRef]) {
        let bcx = self.bcx;
        let tcx = bcx.tcx();
        let mth_idx = option::get(ty::method_idx("visit_" + ty_name,
                                                 *self.visitor_methods));
        let mth_ty = ty::mk_fn(tcx, self.visitor_methods[mth_idx].fty);
        let v = self.visitor_val;
        let get_lval = {|bcx|
            impl::trans_iface_callee(bcx, v, mth_ty, mth_idx)
        };
        self.bcx =
            trans_call_inner(self.bcx, none, mth_ty, ty::mk_bool(tcx),
                             get_lval, arg_vals(args), ignore);
    }

    fn visit_tydesc(t: ty::t) {
        self.bcx =
            call_tydesc_glue(self.bcx, self.visitor_val, t,
                             abi::tydesc_field_visit_glue);
    }

    fn bracketed_mt(bracket_name: str, mt: ty::mt, extra: [ValueRef]) {
        self.visit("enter_" + bracket_name,
                   [self.c_uint(mt.mutbl as uint)] + extra);
        self.visit_tydesc(mt.ty);
        self.visit("leave_" + bracket_name,
                   [self.c_uint(mt.mutbl as uint)] + extra);
    }

    fn vstore_name_and_extra(vstore: ty::vstore,
                             f: fn(str,[ValueRef])) {
        alt vstore {
          ty::vstore_fixed(n) { f("fixed", [self.c_uint(n)]) }
          ty::vstore_slice(_) { f("slice", []) }
          ty::vstore_uniq { f("uniq", []);}
          ty::vstore_box { f("box", []); }
        }
    }

    fn leaf(name: str) {
        self.visit(name, []);
    }

    // Entrypoint
    fn visit_ty(t: ty::t) {

        alt ty::get(t).struct {
          ty::ty_bot { self.leaf("bot") }
          ty::ty_nil { self.leaf("nil") }
          ty::ty_bool { self.leaf("bool") }
          ty::ty_int(ast::ty_i) { self.leaf("int") }
          ty::ty_int(ast::ty_char) { self.leaf("char") }
          ty::ty_int(ast::ty_i8) { self.leaf("i8") }
          ty::ty_int(ast::ty_i16) { self.leaf("i16") }
          ty::ty_int(ast::ty_i32) { self.leaf("i32") }
          ty::ty_int(ast::ty_i64) { self.leaf("i64") }
          ty::ty_uint(ast::ty_u) { self.leaf("uint") }
          ty::ty_uint(ast::ty_u8) { self.leaf("u8") }
          ty::ty_uint(ast::ty_u16) { self.leaf("u16") }
          ty::ty_uint(ast::ty_u32) { self.leaf("u32") }
          ty::ty_uint(ast::ty_u64) { self.leaf("u64") }
          ty::ty_float(ast::ty_f) { self.leaf("float") }
          ty::ty_float(ast::ty_f32) { self.leaf("f32") }
          ty::ty_float(ast::ty_f64) { self.leaf("f64") }
          ty::ty_str { self.leaf("str") }

          ty::ty_vec(mt) { self.bracketed_mt("vec", mt, []) }
          ty::ty_estr(vst) {
            self.vstore_name_and_extra(vst) {|name, extra|
                self.visit("estr_" + name, extra)
            }
          }
          ty::ty_evec(mt, vst) {
            self.vstore_name_and_extra(vst) {|name, extra|
                self.bracketed_mt("evec_" + name, mt, extra)
            }
          }
          ty::ty_box(mt) { self.bracketed_mt("box", mt, []) }
          ty::ty_uniq(mt) { self.bracketed_mt("uniq", mt, []) }
          ty::ty_ptr(mt) { self.bracketed_mt("ptr", mt, []) }
          ty::ty_rptr(_, mt) { self.bracketed_mt("rptr", mt, []) }

          // FIXME: finish these.
          _ { self.visit("bot", []) }
        }
    }
}

// Emit a sequence of calls to visit_ty::visit_foo
fn emit_calls_to_iface_visit_ty(bcx: block, t: ty::t,
                                visitor_val: ValueRef,
                                visitor_iid: def_id) -> block {

    let r = reflector({
        visitor_val: visitor_val,
        visitor_methods: ty::iface_methods(bcx.tcx(), visitor_iid),
        mut bcx: bcx
    });

    r.visit_ty(t);
    ret r.bcx;
}
