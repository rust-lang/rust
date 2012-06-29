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
    visitor_methods: @~[ty::method],
    mut bcx: block
};

impl methods for reflector {

    fn c_uint(u: uint) -> ValueRef {
        C_uint(self.bcx.ccx(), u)
    }

    fn c_int(i: int) -> ValueRef {
        C_int(self.bcx.ccx(), i)
    }

    fn c_slice(s: str) -> ValueRef {
        let ss = C_estr_slice(self.bcx.ccx(), s);
        do_spill_noroot(self.bcx, ss)
    }

    fn c_size_and_align(t: ty::t) -> ~[ValueRef] {
        let tr = type_of::type_of(self.bcx.ccx(), t);
        let s = shape::llsize_of_real(self.bcx.ccx(), tr);
        let a = shape::llalign_of_min(self.bcx.ccx(), tr);
        ret ~[self.c_uint(s),
             self.c_uint(a)];
    }

    fn visit(ty_name: str, args: ~[ValueRef]) {
        let tcx = self.bcx.tcx();
        let mth_idx = option::get(ty::method_idx(@("visit_" + ty_name),
                                                 *self.visitor_methods));
        let mth_ty = ty::mk_fn(tcx, self.visitor_methods[mth_idx].fty);
        let v = self.visitor_val;
        let get_lval = {|bcx|
            let callee =
                impl::trans_iface_callee(bcx, v, mth_ty, mth_idx);
            #debug("calling mth ty %s, lltype %s",
                   ty_to_str(bcx.ccx().tcx, mth_ty),
                   val_str(bcx.ccx().tn, callee.val));
            callee
        };
        #debug("passing %u args:", vec::len(args));
        let bcx = self.bcx;
        for args.eachi {|i, a|
            #debug("arg %u: %s", i, val_str(bcx.ccx().tn, a));
        }
        self.bcx =
            trans_call_inner(self.bcx, none, mth_ty, ty::mk_bool(tcx),
                             get_lval, arg_vals(args), ignore);
    }

    fn visit_tydesc(t: ty::t) {
        self.bcx =
            call_tydesc_glue(self.bcx, self.visitor_val, t,
                             abi::tydesc_field_visit_glue);
    }

    fn bracketed_t(bracket_name: str, t: ty::t, extra: ~[ValueRef]) {
        self.visit("enter_" + bracket_name, extra);
        self.visit_tydesc(t);
        self.visit("leave_" + bracket_name, extra);
    }

    fn bracketed_mt(bracket_name: str, mt: ty::mt, extra: ~[ValueRef]) {
        self.bracketed_t(bracket_name, mt.ty,
                         vec::append(~[self.c_uint(mt.mutbl as uint)],
                                     extra));
    }

    fn vstore_name_and_extra(t: ty::t,
                             vstore: ty::vstore,
                             f: fn(str,~[ValueRef])) {
        alt vstore {
          ty::vstore_fixed(n) {
            let extra = vec::append(~[self.c_uint(n)],
                                    self.c_size_and_align(t));
            f("fixed", extra)
          }
          ty::vstore_slice(_) { f("slice", ~[]) }
          ty::vstore_uniq { f("uniq", ~[]);}
          ty::vstore_box { f("box", ~[]); }
        }
    }

    fn leaf(name: str) {
        self.visit(name, ~[]);
    }

    // Entrypoint
    fn visit_ty(t: ty::t) {

        let bcx = self.bcx;
        #debug("reflect::visit_ty %s",
               ty_to_str(bcx.ccx().tcx, t));

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

          ty::ty_vec(mt) { self.bracketed_mt("vec", mt, ~[]) }
          ty::ty_estr(vst) {
            self.vstore_name_and_extra(t, vst) {|name, extra|
                self.visit("estr_" + name, extra)
            }
          }
          ty::ty_evec(mt, vst) {
            self.vstore_name_and_extra(t, vst) {|name, extra|
                self.bracketed_mt("evec_" + name, mt, extra)
            }
          }
          ty::ty_box(mt) { self.bracketed_mt("box", mt, ~[]) }
          ty::ty_uniq(mt) { self.bracketed_mt("uniq", mt, ~[]) }
          ty::ty_ptr(mt) { self.bracketed_mt("ptr", mt, ~[]) }
          ty::ty_rptr(_, mt) { self.bracketed_mt("rptr", mt, ~[]) }

          ty::ty_rec(fields) {
            let extra = (vec::append(~[self.c_uint(vec::len(fields))],
                                     self.c_size_and_align(t)));
            self.visit("enter_rec", extra);
            for fields.eachi {|i, field|
                self.bracketed_mt("rec_field", field.mt,
                                  ~[self.c_uint(i),
                                   self.c_slice(*field.ident)]);
            }
            self.visit("leave_rec", extra);
          }

          ty::ty_tup(tys) {
            let extra = (vec::append(~[self.c_uint(vec::len(tys))],
                                     self.c_size_and_align(t)));
            self.visit("enter_tup", extra);
            for tys.eachi {|i, t|
                self.bracketed_t("tup_field", t, ~[self.c_uint(i)]);
            }
            self.visit("leave_tup", extra);
          }

          // FIXME (#2594): fetch constants out of intrinsic:: for the
          // numbers.
          ty::ty_fn(fty) {
            let pureval = alt fty.purity {
              ast::pure_fn { 0u }
              ast::unsafe_fn { 1u }
              ast::impure_fn { 2u }
              ast::extern_fn { 3u }
            };
            let protoval = alt fty.proto {
              ast::proto_bare { 0u }
              ast::proto_any { 1u }
              ast::proto_uniq { 2u }
              ast::proto_box { 3u }
              ast::proto_block { 4u }
            };
            let retval = alt fty.ret_style {
              ast::noreturn { 0u }
              ast::return_val { 1u }
            };
            let extra = ~[self.c_uint(pureval),
                         self.c_uint(protoval),
                         self.c_uint(vec::len(fty.inputs)),
                         self.c_uint(retval)];
            self.visit("enter_fn", extra);
            for fty.inputs.eachi {|i, arg|
                let modeval = alt arg.mode {
                  ast::infer(_) { 0u }
                  ast::expl(e) {
                    alt e {
                      ast::by_ref { 1u }
                      ast::by_val { 2u }
                      ast::by_mutbl_ref { 3u }
                      ast::by_move { 4u }
                      ast::by_copy { 5u }
                    }
                  }
                };
                self.bracketed_t("fn_input", arg.ty,
                                 ~[self.c_uint(i),
                                  self.c_uint(modeval)]);
            }
            self.bracketed_t("fn_output", fty.output,
                             ~[self.c_uint(retval)]);
            self.visit("leave_fn", extra);
          }

          ty::ty_class(did, substs) {
            let bcx = self.bcx;
            let tcx = bcx.ccx().tcx;
            let fields = ty::class_items_as_fields(tcx, did, substs);
            let extra = vec::append(~[self.c_uint(vec::len(fields))],
                                    self.c_size_and_align(t));

            self.visit("enter_class", extra);
            for fields.eachi {|i, field|
                self.bracketed_mt("class_field", field.mt,
                                  ~[self.c_uint(i),
                                   self.c_slice(*field.ident)]);
            }
            self.visit("leave_class", extra);
          }

          // FIXME (#2595): visiting all the variants in turn is probably
          // not ideal. It'll work but will get costly on big enums. Maybe
          // let the visitor tell us if it wants to visit only a particular
          // variant?
          ty::ty_enum(did, substs) {
            let bcx = self.bcx;
            let tcx = bcx.ccx().tcx;
            let variants = ty::substd_enum_variants(tcx, did, substs);
            let extra = vec::append(~[self.c_uint(vec::len(variants))],
                                    self.c_size_and_align(t));

            self.visit("enter_enum", extra);
            for variants.eachi {|i, v|
                let extra = ~[self.c_uint(i),
                             self.c_int(v.disr_val),
                             self.c_uint(vec::len(v.args)),
                             self.c_slice(*v.name)];
                self.visit("enter_enum_variant", extra);
                for v.args.eachi {|j, a|
                    self.bracketed_t("enum_variant_field", a,
                                     ~[self.c_uint(j)]);
                }
                self.visit("leave_enum_variant", extra);
            }
            self.visit("leave_enum", extra);
          }

          // Miscallaneous extra types
          ty::ty_iface(_, _) { self.leaf("iface") }
          ty::ty_var(_) { self.leaf("var") }
          ty::ty_var_integral(_) { self.leaf("var_integral") }
          ty::ty_param(n, _) { self.visit("param", ~[self.c_uint(n)]) }
          ty::ty_self { self.leaf("self") }
          ty::ty_type { self.leaf("type") }
          ty::ty_opaque_box { self.leaf("opaque_box") }
          ty::ty_constr(t, _) { self.bracketed_t("constr", t, ~[]) }
          ty::ty_opaque_closure_ptr(ck) {
            let ckval = alt ck {
              ty::ck_block { 0u }
              ty::ck_box { 1u }
              ty::ck_uniq { 2u }
            };
            self.visit("closure_ptr", ~[self.c_uint(ckval)])
          }
          ty::ty_unboxed_vec(mt) { self.bracketed_mt("vec", mt, ~[]) }
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
