// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::link::mangle_internal_name_by_path_and_seq;
use lib::llvm::{ValueRef, llvm};
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee::{ArgVals, DontAutorefArg};
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::glue;
use middle::trans::machine;
use middle::trans::meth;
use middle::trans::type_of::*;
use middle::ty;
use util::ppaux::ty_to_str;

use std::libc::c_uint;
use std::option::None;
use std::vec;
use syntax::ast::DefId;
use syntax::ast;
use syntax::ast_map::path_name;
use syntax::parse::token::special_idents;

use middle::trans::type_::Type;

pub struct Reflector {
    visitor_val: ValueRef,
    visitor_methods: @~[@ty::Method],
    final_bcx: @mut Block,
    tydesc_ty: Type,
    bcx: @mut Block
}

impl Reflector {
    pub fn c_uint(&mut self, u: uint) -> ValueRef {
        C_uint(self.bcx.ccx(), u)
    }

    pub fn c_int(&mut self, i: int) -> ValueRef {
        C_int(self.bcx.ccx(), i)
    }

    pub fn c_bool(&mut self, b: bool) -> ValueRef {
        C_bool(b)
    }

    pub fn c_slice(&mut self, s: @str) -> ValueRef {
        // We're careful to not use first class aggregates here because that
        // will kick us off fast isel. (Issue #4352.)
        let bcx = self.bcx;
        let str_vstore = ty::vstore_slice(ty::re_static);
        let str_ty = ty::mk_estr(bcx.tcx(), str_vstore);
        let scratch = scratch_datum(bcx, str_ty, "", false);
        let len = C_uint(bcx.ccx(), s.len());
        let c_str = PointerCast(bcx, C_cstr(bcx.ccx(), s), Type::i8p());
        Store(bcx, c_str, GEPi(bcx, scratch.val, [ 0, 0 ]));
        Store(bcx, len, GEPi(bcx, scratch.val, [ 0, 1 ]));
        scratch.val
    }

    pub fn c_size_and_align(&mut self, t: ty::t) -> ~[ValueRef] {
        let tr = type_of(self.bcx.ccx(), t);
        let s = machine::llsize_of_real(self.bcx.ccx(), tr);
        let a = machine::llalign_of_min(self.bcx.ccx(), tr);
        return ~[self.c_uint(s),
             self.c_uint(a)];
    }

    pub fn c_tydesc(&mut self, t: ty::t) -> ValueRef {
        let bcx = self.bcx;
        let static_ti = get_tydesc(bcx.ccx(), t);
        glue::lazily_emit_all_tydesc_glue(bcx.ccx(), static_ti);
        PointerCast(bcx, static_ti.tydesc, self.tydesc_ty.ptr_to())
    }

    pub fn c_mt(&mut self, mt: &ty::mt) -> ~[ValueRef] {
        ~[self.c_uint(mt.mutbl as uint),
          self.c_tydesc(mt.ty)]
    }

    pub fn visit(&mut self, ty_name: &str, args: &[ValueRef]) {
        let tcx = self.bcx.tcx();
        let mth_idx = ty::method_idx(
            tcx.sess.ident_of(~"visit_" + ty_name),
            *self.visitor_methods).expect(fmt!("Couldn't find visit method \
                                                for %s", ty_name));
        let mth_ty =
            ty::mk_bare_fn(tcx, self.visitor_methods[mth_idx].fty.clone());
        let v = self.visitor_val;
        debug!("passing %u args:", args.len());
        let mut bcx = self.bcx;
        for (i, a) in args.iter().enumerate() {
            debug!("arg %u: %s", i, bcx.val_to_str(*a));
        }
        let bool_ty = ty::mk_bool();
        let result = unpack_result!(bcx, callee::trans_call_inner(
            self.bcx, None, mth_ty, bool_ty,
            |bcx| meth::trans_trait_callee_from_llval(bcx,
                                                      mth_ty,
                                                      mth_idx,
                                                      v,
                                                      None),
            ArgVals(args), None, DontAutorefArg));
        let result = bool_to_i1(bcx, result);
        let next_bcx = sub_block(bcx, "next");
        CondBr(bcx, result, next_bcx.llbb, self.final_bcx.llbb);
        self.bcx = next_bcx
    }

    pub fn bracketed(&mut self,
                     bracket_name: &str,
                     extra: &[ValueRef],
                     inner: &fn(&mut Reflector)) {
        self.visit("enter_" + bracket_name, extra);
        inner(self);
        self.visit("leave_" + bracket_name, extra);
    }

    pub fn vstore_name_and_extra(&mut self,
                                 t: ty::t,
                                 vstore: ty::vstore)
                                 -> (~str, ~[ValueRef]) {
        match vstore {
            ty::vstore_fixed(n) => {
                let extra = vec::append(~[self.c_uint(n)],
                                        self.c_size_and_align(t));
                (~"fixed", extra)
            }
            ty::vstore_slice(_) => (~"slice", ~[]),
            ty::vstore_uniq => (~"uniq", ~[]),
            ty::vstore_box => (~"box", ~[])
        }
    }

    pub fn leaf(&mut self, name: &str) {
        self.visit(name, []);
    }

    // Entrypoint
    pub fn visit_ty(&mut self, t: ty::t) {
        let bcx = self.bcx;
        let tcx = bcx.ccx().tcx;
        debug!("reflect::visit_ty %s", ty_to_str(bcx.ccx().tcx, t));

        match ty::get(t).sty {
          ty::ty_bot => self.leaf("bot"),
          ty::ty_nil => self.leaf("nil"),
          ty::ty_bool => self.leaf("bool"),
          ty::ty_char => self.leaf("char"),
          ty::ty_int(ast::ty_i) => self.leaf("int"),
          ty::ty_int(ast::ty_i8) => self.leaf("i8"),
          ty::ty_int(ast::ty_i16) => self.leaf("i16"),
          ty::ty_int(ast::ty_i32) => self.leaf("i32"),
          ty::ty_int(ast::ty_i64) => self.leaf("i64"),
          ty::ty_uint(ast::ty_u) => self.leaf("uint"),
          ty::ty_uint(ast::ty_u8) => self.leaf("u8"),
          ty::ty_uint(ast::ty_u16) => self.leaf("u16"),
          ty::ty_uint(ast::ty_u32) => self.leaf("u32"),
          ty::ty_uint(ast::ty_u64) => self.leaf("u64"),
          ty::ty_float(ast::ty_f) => self.leaf("float"),
          ty::ty_float(ast::ty_f32) => self.leaf("f32"),
          ty::ty_float(ast::ty_f64) => self.leaf("f64"),

          ty::ty_unboxed_vec(ref mt) => {
              let values = self.c_mt(mt);
              self.visit("vec", values)
          }

          ty::ty_estr(vst) => {
              let (name, extra) = self.vstore_name_and_extra(t, vst);
              self.visit(~"estr_" + name, extra)
          }
          ty::ty_evec(ref mt, vst) => {
              let (name, extra) = self.vstore_name_and_extra(t, vst);
              let extra = extra + self.c_mt(mt);
              if "uniq" == name && ty::type_contents(bcx.tcx(), t).contains_managed() {
                  self.visit("evec_uniq_managed", extra)
              } else {
                  self.visit(~"evec_" + name, extra)
              }
          }
          ty::ty_box(ref mt) => {
              let extra = self.c_mt(mt);
              self.visit("box", extra)
          }
          ty::ty_uniq(ref mt) => {
              let extra = self.c_mt(mt);
              if ty::type_contents(bcx.tcx(), t).contains_managed() {
                  self.visit("uniq_managed", extra)
              } else {
                  self.visit("uniq", extra)
              }
          }
          ty::ty_ptr(ref mt) => {
              let extra = self.c_mt(mt);
              self.visit("ptr", extra)
          }
          ty::ty_rptr(_, ref mt) => {
              let extra = self.c_mt(mt);
              self.visit("rptr", extra)
          }

          ty::ty_tup(ref tys) => {
              let extra = ~[self.c_uint(tys.len())]
                  + self.c_size_and_align(t);
              do self.bracketed("tup", extra) |this| {
                  for (i, t) in tys.iter().enumerate() {
                      let extra = ~[this.c_uint(i), this.c_tydesc(*t)];
                      this.visit("tup_field", extra);
                  }
              }
          }

          // FIXME (#2594): fetch constants out of intrinsic
          // FIXME (#4809): visitor should break out bare fns from other fns
          ty::ty_closure(ref fty) => {
            let pureval = ast_purity_constant(fty.purity);
            let sigilval = ast_sigil_constant(fty.sigil);
            let retval = if ty::type_is_bot(fty.sig.output) {0u} else {1u};
            let extra = ~[self.c_uint(pureval),
                          self.c_uint(sigilval),
                          self.c_uint(fty.sig.inputs.len()),
                          self.c_uint(retval)];
            self.visit("enter_fn", extra);
            self.visit_sig(retval, &fty.sig);
            self.visit("leave_fn", extra);
          }

          // FIXME (#2594): fetch constants out of intrinsic:: for the
          // numbers.
          ty::ty_bare_fn(ref fty) => {
            let pureval = ast_purity_constant(fty.purity);
            let sigilval = 0u;
            let retval = if ty::type_is_bot(fty.sig.output) {0u} else {1u};
            let extra = ~[self.c_uint(pureval),
                          self.c_uint(sigilval),
                          self.c_uint(fty.sig.inputs.len()),
                          self.c_uint(retval)];
            self.visit("enter_fn", extra);
            self.visit_sig(retval, &fty.sig);
            self.visit("leave_fn", extra);
          }

          ty::ty_struct(did, ref substs) => {
              let fields = ty::struct_fields(tcx, did, substs);
              let mut named_fields = false;
              if !fields.is_empty() {
                  named_fields = fields[0].ident != special_idents::unnamed_field;
              }

              let extra = ~[self.c_slice(ty_to_str(tcx, t).to_managed()),
                            self.c_bool(named_fields),
                            self.c_uint(fields.len())] + self.c_size_and_align(t);
              do self.bracketed("class", extra) |this| {
                  for (i, field) in fields.iter().enumerate() {
                      let extra = ~[this.c_uint(i),
                                    this.c_slice(bcx.ccx().sess.str_of(field.ident)),
                                    this.c_bool(named_fields)]
                          + this.c_mt(&field.mt);
                      this.visit("class_field", extra);
                  }
              }
          }

          // FIXME (#2595): visiting all the variants in turn is probably
          // not ideal. It'll work but will get costly on big enums. Maybe
          // let the visitor tell us if it wants to visit only a particular
          // variant?
          ty::ty_enum(did, ref substs) => {
            let ccx = bcx.ccx();
            let repr = adt::represent_type(bcx.ccx(), t);
            let variants = ty::substd_enum_variants(ccx.tcx, did, substs);
            let llptrty = type_of(ccx, t).ptr_to();
            let opaquety = ty::get_opaque_ty(ccx.tcx).unwrap();
            let opaqueptrty = ty::mk_ptr(ccx.tcx, ty::mt { ty: opaquety,
                                                           mutbl: ast::MutImmutable });

            let make_get_disr = || {
                let sub_path = bcx.fcx.path + &[path_name(special_idents::anon)];
                let sym = mangle_internal_name_by_path_and_seq(ccx,
                                                               sub_path,
                                                               "get_disr");

                let llfty = type_of_rust_fn(ccx, [opaqueptrty], ty::mk_int());
                let llfdecl = decl_internal_cdecl_fn(ccx.llmod, sym, llfty);
                let fcx = new_fn_ctxt(ccx,
                                      ~[],
                                      llfdecl,
                                      ty::mk_uint(),
                                      None);
                let arg = unsafe {
                    //
                    // we know the return type of llfdecl is an int here, so
                    // no need for a special check to see if the return type
                    // is immediate.
                    //
                    llvm::LLVMGetParam(llfdecl, fcx.arg_pos(0u) as c_uint)
                };
                let mut bcx = fcx.entry_bcx.unwrap();
                let arg = BitCast(bcx, arg, llptrty);
                let ret = adt::trans_get_discr(bcx, repr, arg);
                Store(bcx, ret, fcx.llretptr.unwrap());
                match fcx.llreturn {
                    Some(llreturn) => cleanup_and_Br(bcx, bcx, llreturn),
                    None => bcx = cleanup_block(bcx, Some(bcx.llbb))
                };
                finish_fn(fcx, bcx);
                llfdecl
            };

            let enum_args = ~[self.c_uint(variants.len()), make_get_disr()]
                + self.c_size_and_align(t);
            do self.bracketed("enum", enum_args) |this| {
                for (i, v) in variants.iter().enumerate() {
                    let name = ccx.sess.str_of(v.name);
                    let variant_args = ~[this.c_uint(i),
                                         C_integral(self.bcx.ccx().int_type, v.disr_val, false),
                                         this.c_uint(v.args.len()),
                                         this.c_slice(name)];
                    do this.bracketed("enum_variant", variant_args) |this| {
                        for (j, a) in v.args.iter().enumerate() {
                            let bcx = this.bcx;
                            let null = C_null(llptrty);
                            let ptr = adt::trans_field_ptr(bcx, repr, null, v.disr_val, j);
                            let offset = p2i(ccx, ptr);
                            let field_args = ~[this.c_uint(j),
                                               offset,
                                               this.c_tydesc(*a)];
                            this.visit("enum_variant_field", field_args);
                        }
                    }
                }
            }
          }

          ty::ty_trait(_, _, _, _, _) => {
              let extra = [self.c_slice(ty_to_str(tcx, t).to_managed())];
              self.visit("trait", extra);
          }

          // Miscellaneous extra types
          ty::ty_infer(_) => self.leaf("infer"),
          ty::ty_err => self.leaf("err"),
          ty::ty_param(ref p) => {
              let extra = ~[self.c_uint(p.idx)];
              self.visit("param", extra)
          }
          ty::ty_self(*) => self.leaf("self"),
          ty::ty_type => self.leaf("type"),
          ty::ty_opaque_box => self.leaf("opaque_box"),
          ty::ty_opaque_closure_ptr(ck) => {
              let ckval = ast_sigil_constant(ck);
              let extra = ~[self.c_uint(ckval)];
              self.visit("closure_ptr", extra)
          }
        }
    }

    pub fn visit_sig(&mut self, retval: uint, sig: &ty::FnSig) {
        for (i, arg) in sig.inputs.iter().enumerate() {
            let modeval = 5u;   // "by copy"
            let extra = ~[self.c_uint(i),
                         self.c_uint(modeval),
                         self.c_tydesc(*arg)];
            self.visit("fn_input", extra);
        }
        let extra = ~[self.c_uint(retval),
                      self.c_tydesc(sig.output)];
        self.visit("fn_output", extra);
    }
}

// Emit a sequence of calls to visit_ty::visit_foo
pub fn emit_calls_to_trait_visit_ty(bcx: @mut Block,
                                    t: ty::t,
                                    visitor_val: ValueRef,
                                    visitor_trait_id: DefId)
                                 -> @mut Block {
    let final = sub_block(bcx, "final");
    let tydesc_ty = ty::get_tydesc_ty(bcx.ccx().tcx).unwrap();
    let tydesc_ty = type_of(bcx.ccx(), tydesc_ty);
    let mut r = Reflector {
        visitor_val: visitor_val,
        visitor_methods: ty::trait_methods(bcx.tcx(), visitor_trait_id),
        final_bcx: final,
        tydesc_ty: tydesc_ty,
        bcx: bcx
    };
    r.visit_ty(t);
    Br(r.bcx, final.llbb);
    return final;
}

pub fn ast_sigil_constant(sigil: ast::Sigil) -> uint {
    match sigil {
        ast::OwnedSigil => 2u,
        ast::ManagedSigil => 3u,
        ast::BorrowedSigil => 4u,
    }
}

pub fn ast_purity_constant(purity: ast::purity) -> uint {
    match purity {
        ast::unsafe_fn => 1u,
        ast::impure_fn => 2u,
        ast::extern_fn => 3u
    }
}
