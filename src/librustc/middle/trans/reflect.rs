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
use lib::llvm::llvm;
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::glue;
use middle::trans::machine;
use middle::trans::type_::Type;
use middle::trans::type_of::*;
use middle::ty;
use util::ppaux::ty_to_str;

use arena::TypedArena;
use std::libc::c_uint;
use syntax::ast::DefId;
use syntax::ast;
use syntax::ast_map;
use syntax::parse::token::{InternedString, special_idents};
use syntax::parse::token;

struct Reflector<'a> {
    visitor_datum: Datum<Lvalue>,
    visitor_methods: @Vec<@ty::Method>,
    final_bcx: &'a Block<'a>,
    tydesc_ptr: ty::t,
    tydesc_ptr_llty: Type,
    bcx: &'a Block<'a>
}

impl<'a> Reflector<'a> {
    fn c_uint(&self, u: uint) -> Datum<PodValue> {
        pod_value(self.bcx.tcx(), C_uint(self.bcx.ccx(), u), ty::mk_uint())
    }

    fn c_bool(&self, b: bool) -> Datum<PodValue> {
        pod_value(self.bcx.tcx(), C_bool(self.bcx.ccx(), b), ty::mk_bool())
    }

    fn c_slice(&self, s: InternedString) -> Datum<Rvalue> {
        // We're careful to not use first class aggregates here because that
        // will kick us off fast isel. (Issue #4352.)
        let bcx = self.bcx;
        let str_vstore = ty::vstore_slice(ty::ReStatic);
        let str_ty = ty::mk_str(bcx.tcx(), str_vstore);
        let scratch = rvalue_scratch_datum(bcx, str_ty, "");
        let len = C_uint(bcx.ccx(), s.get().len());
        let c_str = PointerCast(bcx, C_cstr(bcx.ccx(), s), Type::i8p(bcx.ccx()));
        Store(bcx, c_str, GEPi(bcx, scratch.val, [ 0, 0 ]));
        Store(bcx, len, GEPi(bcx, scratch.val, [ 0, 1 ]));
        scratch
    }

    fn c_size_and_align(&self, t: ty::t) -> [Datum<PodValue>, ..2] {
        let tr = type_of(self.bcx.ccx(), t);
        [
            self.c_uint(machine::llsize_of_real(self.bcx.ccx(), tr) as uint),
            self.c_uint(machine::llalign_of_min(self.bcx.ccx(), tr) as uint)
        ]
    }

    fn c_tydesc(&self, t: ty::t) -> Datum<PodValue> {
        let bcx = self.bcx;
        let static_ti = get_tydesc(bcx.ccx(), t);
        glue::lazily_emit_visit_glue(bcx.ccx(), static_ti);
        let ptr = PointerCast(bcx, static_ti.tydesc, self.tydesc_ptr_llty);
        pod_value(self.bcx.tcx(), ptr, self.tydesc_ptr)
    }

    fn c_mt(&self, mt: &ty::mt) -> [Datum<PodValue>, ..2] {
        [self.c_uint(mt.mutbl as uint), self.c_tydesc(mt.ty)]
    }

    fn vstore_name_and_extra(&self,
                             t: ty::t,
                             vstore: ty::vstore)
                             -> (&'static str, Option<[Datum<PodValue>, ..3]>) {
        match vstore {
            ty::vstore_fixed(n) => {
                let size_align = self.c_size_and_align(t);
                ("fixed", Some([self.c_uint(n), size_align[0], size_align[1]]))
            }
            ty::vstore_slice(_) => ("slice", None),
            ty::vstore_uniq => ("uniq", None),
        }
    }

    fn visit_inner<I: Iterator<Datum<Expr>>>(self, ty_name: &str, args: I) -> Reflector<'a> {
        let tcx = self.bcx.tcx();
        let mth_idx = ty::method_idx(token::str_to_ident(~"visit_" + ty_name),
                                     self.visitor_methods.as_slice()).expect(
                format!("couldn't find visit method for {}", ty_name));
        let mth_ty =
            ty::mk_bare_fn(tcx,
                           self.visitor_methods.get(mth_idx).fty.clone());
        let visitor = self.visitor_datum.to_expr_datum();
        let mut bcx = self.bcx;
        let result = unpack_result!(bcx, callee::trans_call(
            bcx, None,
            callee::TraitMethod(mth_ty, mth_idx),
            Some(visitor).move_iter().chain(args),
            |bcx, arg_datum| DatumBlock(bcx, arg_datum),
            None));
        let result = bool_to_i1(bcx, result);
        let next_bcx = self.bcx.fcx.new_temp_block("next");
        CondBr(bcx, result, next_bcx.llbb, self.final_bcx.llbb);

        Reflector {
            bcx: next_bcx,
            ..self
        }
    }

    fn visit(self, ty_name: &str, args: &[Datum<PodValue>]) -> Reflector<'a> {
        self.visit_inner(ty_name, args.iter().map(|arg| arg.to_expr_datum()))
    }

    fn bracketed<I: Iterator<Datum<Expr>>>(self,
                                           bracket_name: &str, extra: || -> I,
                                           inner: |Reflector<'a>| -> Reflector<'a>)
                                           -> Reflector<'a> {
        inner(self.visit_inner("enter_" + bracket_name, extra()))
                  .visit_inner("leave_" + bracket_name, extra())
    }

    fn leaf(self, name: &str) -> Reflector<'a> {
        self.visit(name, [])
    }

    // Entrypoint
    fn visit_ty(self, t: ty::t) -> Reflector<'a> {
        let bcx = self.bcx;
        let tcx = bcx.tcx();
        debug!("reflect::visit_ty {}", ty_to_str(bcx.tcx(), t));

        match ty::get(t).sty {
          ty::ty_bot => self.leaf("bot"),
          ty::ty_nil => self.leaf("nil"),
          ty::ty_bool => self.leaf("bool"),
          ty::ty_char => self.leaf("char"),
          ty::ty_int(ast::TyI) => self.leaf("int"),
          ty::ty_int(ast::TyI8) => self.leaf("i8"),
          ty::ty_int(ast::TyI16) => self.leaf("i16"),
          ty::ty_int(ast::TyI32) => self.leaf("i32"),
          ty::ty_int(ast::TyI64) => self.leaf("i64"),
          ty::ty_uint(ast::TyU) => self.leaf("uint"),
          ty::ty_uint(ast::TyU8) => self.leaf("u8"),
          ty::ty_uint(ast::TyU16) => self.leaf("u16"),
          ty::ty_uint(ast::TyU32) => self.leaf("u32"),
          ty::ty_uint(ast::TyU64) => self.leaf("u64"),
          ty::ty_float(ast::TyF32) => self.leaf("f32"),
          ty::ty_float(ast::TyF64) => self.leaf("f64"),

          ty::ty_unboxed_vec(ref mt) => {
              self.visit("vec", self.c_mt(mt))
          }

          // Should rename to str_*/vec_*.
          ty::ty_str(vst) => {
              let (name, extra) = self.vstore_name_and_extra(t, vst);
              match extra {
                  Some(extra) => self.visit(~"estr_" + name, extra),
                  None => self.visit(~"estr_" + name, [])
              }
          }
          ty::ty_vec(ref mt, vst) => {
              let (name, extra) = self.vstore_name_and_extra(t, vst);
              let mt = self.c_mt(mt);
              match extra {
                  Some([n, size, align]) => {
                      self.visit(~"evec_" + name, [n, size, align, mt[0], mt[1]])
                  }
                  None => self.visit(~"evec_" + name, [mt[0], mt[1]])
              }
          }
          // Should remove mt from box and uniq.
          ty::ty_box(typ) => {
              let extra = self.c_mt(&ty::mt {
                  ty: typ,
                  mutbl: ast::MutImmutable,
              });
              self.visit("box", extra)
          }
          ty::ty_uniq(typ) => {
              let extra = self.c_mt(&ty::mt {
                  ty: typ,
                  mutbl: ast::MutImmutable,
              });
              self.visit("uniq", extra)
          }
          ty::ty_ptr(ref mt) => {
              self.visit("ptr", self.c_mt(mt))
          }
          ty::ty_rptr(_, ref mt) => {
              self.visit("rptr", self.c_mt(mt))
          }

          ty::ty_tup(ref tys) => {
              let size_align = self.c_size_and_align(t);
              let extra = [
                self.c_uint(tys.len()),
                size_align[0],
                size_align[1]
              ];
              self.bracketed("tup",
                || extra.iter().map(|arg| arg.to_expr_datum()),
                |mut this| {
                  for (i, t) in tys.iter().enumerate() {
                      this = this.visit("tup_field", [this.c_uint(i), this.c_tydesc(*t)]);
                  }
                  this
              })
          }

          // FIXME (#2594): fetch constants out of intrinsic
          // FIXME (#4809): visitor should break out bare fns from other fns
          ty::ty_closure(ref fty) => {
            let pureval = ast_purity_constant(fty.purity);
            let sigilval = match fty.sigil {
                ast::OwnedSigil => 2u,
                ast::ManagedSigil => 3u,
                ast::BorrowedSigil => 4u,
            };
            let retval = if ty::type_is_bot(fty.sig.output) {0u} else {1u};
            let extra = [self.c_uint(pureval),
                         self.c_uint(sigilval),
                         self.c_uint(fty.sig.inputs.len()),
                         self.c_uint(retval)];
            self.visit("enter_fn", extra)
                .visit_sig(retval, &fty.sig)
                .visit("leave_fn", extra)
          }

          // FIXME (#2594): fetch constants out of intrinsic:: for the
          // numbers.
          ty::ty_bare_fn(ref fty) => {
            let pureval = ast_purity_constant(fty.purity);
            let sigilval = 0u;
            let retval = if ty::type_is_bot(fty.sig.output) {0u} else {1u};
            let extra = [self.c_uint(pureval),
                         self.c_uint(sigilval),
                         self.c_uint(fty.sig.inputs.len()),
                         self.c_uint(retval)];
            self.visit("enter_fn", extra)
                 .visit_sig(retval, &fty.sig)
                 .visit("leave_fn", extra)
          }

          ty::ty_struct(did, ref substs) => {
              let fields = ty::struct_fields(tcx, did, substs);
              let mut named_fields = false;
              if !fields.is_empty() {
                  named_fields = fields.get(0).ident.name !=
                      special_idents::unnamed_field.name;
              }

              let size_align = self.c_size_and_align(t);
              let name = self.c_slice(token::intern_and_get_ident(ty_to_str(tcx, t)));
              let extra = [
                self.c_bool(named_fields),
                self.c_uint(fields.len()),
                size_align[0], size_align[1]
              ];
              self.bracketed("class",
                || {
                    let name = Datum(name.val, name.ty, RvalueExpr(Rvalue(name.kind.mode)));
                    Some(name).move_iter().chain(extra.iter().map(|arg| arg.to_expr_datum()))
                }, |mut this| {
                  for (i, field) in fields.iter().enumerate() {
                      let mt = this.c_mt(&field.mt);
                      let name = this.c_slice(token::get_ident(field.ident)).to_expr_datum();
                      let extra = [this.c_bool(named_fields), mt[0], mt[1]];
                      this = this.visit_inner("class_field",
                        Some(this.c_uint(i).to_expr_datum()).move_iter()
                            .chain(Some(name).move_iter())
                            .chain(extra.iter().map(|arg| arg.to_expr_datum()))
                      );
                  }
                  this
              })
          }

          // FIXME (#2595): visiting all the variants in turn is probably
          // not ideal. It'll work but will get costly on big enums. Maybe
          // let the visitor tell us if it wants to visit only a particular
          // variant?
          ty::ty_enum(did, ref substs) => {
            let ccx = bcx.ccx();
            let repr = adt::represent_type(bcx.ccx(), t);
            let variants = ty::substd_enum_variants(ccx.tcx(), did, substs);
            let llptrty = type_of(ccx, t).ptr_to();
            let opaquety = ty::get_opaque_ty(ccx.tcx()).unwrap();
            let opaqueptrty = ty::mk_imm_ptr(ccx.tcx(), opaquety);

            let get_disr = {
                let sym = mangle_internal_name_by_path_and_seq(
                    ast_map::Values([].iter()).chain(None), "get_disr");

                let llfdecl = decl_internal_rust_fn(ccx, false, [opaqueptrty], ty::mk_u64(), sym);
                let arena = TypedArena::new();
                let fcx = new_fn_ctxt(ccx, llfdecl, -1, false,
                                      ty::mk_u64(), None, None, &arena);
                init_function(&fcx, false, ty::mk_u64(), None);

                let arg = unsafe {
                    //
                    // we know the return type of llfdecl is an int here, so
                    // no need for a special check to see if the return type
                    // is immediate.
                    //
                    llvm::LLVMGetParam(llfdecl, fcx.arg_pos(0u) as c_uint)
                };
                let bcx = fcx.entry_bcx.get().unwrap();
                let arg = BitCast(bcx, arg, llptrty);
                let ret = adt::trans_get_discr(bcx, repr, arg, Some(Type::i64(ccx)));
                Store(bcx, ret, fcx.llretptr.get().unwrap());
                match fcx.llreturn.get() {
                    Some(llreturn) => Br(bcx, llreturn),
                    None => {}
                };
                finish_fn(&fcx, bcx);
                pod_value(bcx.tcx(), llfdecl, ty::mk_imm_ptr(bcx.tcx(), ty::mk_nil()))
            };

            let size_align = self.c_size_and_align(t);
            let enum_args = [
                self.c_uint(variants.len()),
                get_disr,
                size_align[0],
                size_align[1]
            ];
            self.bracketed("enum",
              || enum_args.iter().map(|arg| arg.to_expr_datum()),
              |mut this| {
                for (i, v) in variants.iter().enumerate() {
                    let args = [
                        this.c_uint(i),
                        pod_value(this.bcx.tcx(), C_u64(ccx, v.disr_val), ty::mk_u64()),
                        this.c_uint(v.args.len()),
                    ];
                    let name = this.c_slice(token::get_ident(v.name));
                    this = this.bracketed("enum_variant", || {
                        let name = Datum(name.val, name.ty, RvalueExpr(Rvalue(name.kind.mode)));
                        args.iter().map(|arg| arg.to_expr_datum()).chain(Some(name).move_iter())
                    }, |mut this| {
                        for (j, a) in v.args.iter().enumerate() {
                            let offset = p2i(ccx, adt::trans_field_ptr(this.bcx, repr,
                                                                       C_null(llptrty),
                                                                       v.disr_val, j));
                            this = this.visit("enum_variant_field", [
                                this.c_uint(j),
                                pod_value(this.bcx.tcx(), offset, ty::mk_int()),
                                this.c_tydesc(*a)
                            ]);
                        }
                        this
                    })
                }
                this
            })
          }

          ty::ty_trait(_) => {
              let name = self.c_slice(token::intern_and_get_ident(ty_to_str(tcx, t)));
              self.visit_inner("trait", Some(name.to_expr_datum()).move_iter())
          }

          // Miscellaneous extra types
          ty::ty_infer(_) => self.leaf("infer"),
          ty::ty_err => self.leaf("err"),
          ty::ty_param(ref p) => self.visit("param", [self.c_uint(p.idx)]),
          ty::ty_self(..) => self.leaf("self")
        }
    }

    fn visit_sig(mut self, retval: uint, sig: &ty::FnSig) -> Reflector<'a> {
        for (i, arg) in sig.inputs.iter().enumerate() {
            let modeval = 5u;   // "by copy"
            self = self.visit("fn_input", [
                self.c_uint(i),
                self.c_uint(modeval),
                self.c_tydesc(*arg)
            ]);
        }
        self.visit("fn_output", [
            self.c_uint(retval),
            self.c_bool(sig.variadic),
            self.c_tydesc(sig.output)
        ])
    }
}

// Emit a sequence of calls to visit_ty::visit_foo
pub fn emit_calls_to_trait_visit_ty<'a>(
                                    bcx: &'a Block<'a>,
                                    t: ty::t,
                                    visitor_datum: Datum<Lvalue>,
                                    visitor_trait_id: DefId)
                                    -> &'a Block<'a> {
    let final = bcx.fcx.new_temp_block("final");
    let tydesc_ptr = ty::mk_imm_ptr(bcx.tcx(), ty::get_tydesc_ty(bcx.tcx()).unwrap());
    let r = Reflector {
        visitor_datum: visitor_datum,
        visitor_methods: ty::trait_methods(bcx.tcx(), visitor_trait_id),
        final_bcx: final,
        tydesc_ptr: tydesc_ptr,
        tydesc_ptr_llty: type_of(bcx.ccx(), tydesc_ptr),
        bcx: bcx
    };
    Br(r.visit_ty(t).bcx, final.llbb);
    final
}

fn ast_purity_constant(purity: ast::Purity) -> uint {
    match purity {
        ast::UnsafeFn => 1u,
        ast::ImpureFn => 2u,
        ast::ExternFn => 3u
    }
}
