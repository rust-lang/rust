// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use lib::llvm::{llvm, ConstFCmp, ConstICmp, SetLinkage, PrivateLinkage, ValueRef, Bool, True};
use lib::llvm::{IntEQ, IntNE, IntUGT, IntUGE, IntULT, IntULE, IntSGT, IntSGE, IntSLT, IntSLE,
    RealOEQ, RealOGT, RealOGE, RealOLT, RealOLE, RealONE};

use metadata::csearch;
use middle::const_eval;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::base::push_ctxt;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::expr;
use middle::trans::inline;
use middle::trans::machine;
use middle::trans::type_of;
use middle::ty;
use util::ppaux::{Repr, ty_to_str};

use middle::trans::type_::Type;

use std::c_str::ToCStr;
use std::libc::c_uint;
use std::vec;
use syntax::{ast, ast_util, ast_map};

pub fn const_lit(cx: &mut CrateContext, e: &ast::Expr, lit: ast::lit)
    -> ValueRef {
    let _icx = push_ctxt("trans_lit");
    match lit.node {
      ast::lit_char(i) => C_integral(Type::char(), i as u64, false),
      ast::lit_int(i, t) => C_integral(Type::int_from_ty(cx, t), i as u64, true),
      ast::lit_uint(u, t) => C_integral(Type::uint_from_ty(cx, t), u, false),
      ast::lit_int_unsuffixed(i) => {
        let lit_int_ty = ty::node_id_to_type(cx.tcx, e.id);
        match ty::get(lit_int_ty).sty {
          ty::ty_int(t) => {
            C_integral(Type::int_from_ty(cx, t), i as u64, true)
          }
          ty::ty_uint(t) => {
            C_integral(Type::uint_from_ty(cx, t), i as u64, false)
          }
          _ => cx.sess.span_bug(lit.span,
                   fmt!("integer literal has type %s (expected int or uint)",
                        ty_to_str(cx.tcx, lit_int_ty)))
        }
      }
      ast::lit_float(fs, t) => C_floating(fs, Type::float_from_ty(cx, t)),
      ast::lit_float_unsuffixed(fs) => {
        let lit_float_ty = ty::node_id_to_type(cx.tcx, e.id);
        match ty::get(lit_float_ty).sty {
          ty::ty_float(t) => {
            C_floating(fs, Type::float_from_ty(cx, t))
          }
          _ => {
            cx.sess.span_bug(lit.span,
                             "floating point literal doesn't have the right type");
          }
        }
      }
      ast::lit_bool(b) => C_bool(b),
      ast::lit_nil => C_nil(),
      ast::lit_str(s) => C_estr_slice(cx, s)
    }
}

pub fn const_ptrcast(cx: &mut CrateContext, a: ValueRef, t: Type) -> ValueRef {
    unsafe {
        let b = llvm::LLVMConstPointerCast(a, t.ptr_to().to_ref());
        assert!(cx.const_globals.insert(b as int, a));
        b
    }
}

pub fn const_vec(cx: @mut CrateContext, e: &ast::Expr, es: &[@ast::Expr])
    -> (ValueRef, ValueRef, Type) {
    unsafe {
        let vec_ty = ty::expr_ty(cx.tcx, e);
        let unit_ty = ty::sequence_element_type(cx.tcx, vec_ty);
        let llunitty = type_of::type_of(cx, unit_ty);
        let unit_sz = machine::llsize_of(cx, llunitty);
        let sz = llvm::LLVMConstMul(C_uint(cx, es.len()), unit_sz);
        let vs = es.map(|e| const_expr(cx, *e));
        // If the vector contains enums, an LLVM array won't work.
        let v = if vs.iter().any(|vi| val_ty(*vi) != llunitty) {
            C_struct(vs)
        } else {
            C_array(llunitty, vs)
        };
        return (v, sz, llunitty);
    }
}

fn const_addr_of(cx: &mut CrateContext, cv: ValueRef) -> ValueRef {
    unsafe {
        let gv = do "const".with_c_str |name| {
            llvm::LLVMAddGlobal(cx.llmod, val_ty(cv).to_ref(), name)
        };
        llvm::LLVMSetInitializer(gv, cv);
        llvm::LLVMSetGlobalConstant(gv, True);
        SetLinkage(gv, PrivateLinkage);
        gv
    }
}

fn const_deref_ptr(cx: &mut CrateContext, v: ValueRef) -> ValueRef {
    let v = match cx.const_globals.find(&(v as int)) {
        Some(&v) => v,
        None => v
    };
    unsafe {
        assert_eq!(llvm::LLVMIsGlobalConstant(v), True);
        llvm::LLVMGetInitializer(v)
    }
}

fn const_deref_newtype(cx: &mut CrateContext, v: ValueRef, t: ty::t)
    -> ValueRef {
    let repr = adt::represent_type(cx, t);
    adt::const_get_field(cx, repr, v, 0, 0)
}

fn const_deref(cx: &mut CrateContext, v: ValueRef, t: ty::t, explicit: bool)
    -> (ValueRef, ty::t) {
    match ty::deref(cx.tcx, t, explicit) {
        Some(ref mt) => {
            assert!(mt.mutbl != ast::MutMutable);
            let dv = match ty::get(t).sty {
                ty::ty_ptr(*) | ty::ty_rptr(*) => {
                     const_deref_ptr(cx, v)
                }
                ty::ty_enum(*) | ty::ty_struct(*) => {
                    const_deref_newtype(cx, v, t)
                }
                _ => {
                    cx.sess.bug(fmt!("Unexpected dereferenceable type %s",
                                     ty_to_str(cx.tcx, t)))
                }
            };
            (dv, mt.ty)
        }
        None => {
            cx.sess.bug(fmt!("Can't dereference const of type %s",
                             ty_to_str(cx.tcx, t)))
        }
    }
}

pub fn get_const_val(cx: @mut CrateContext, mut def_id: ast::DefId) -> ValueRef {
    let contains_key = cx.const_values.contains_key(&def_id.node);
    if !ast_util::is_local(def_id) || !contains_key {
        if !ast_util::is_local(def_id) {
            def_id = inline::maybe_instantiate_inline(cx, def_id);
        }
        match cx.tcx.items.get_copy(&def_id.node) {
            ast_map::node_item(@ast::item {
                node: ast::item_static(_, ast::MutImmutable, _), _
            }, _) => {
                trans_const(cx, ast::MutImmutable, def_id.node);
            }
            _ => cx.tcx.sess.bug("expected a const to be an item")
        }
    }
    cx.const_values.get_copy(&def_id.node)
}

pub fn const_expr(cx: @mut CrateContext, e: @ast::Expr) -> ValueRef {
    let mut llconst = const_expr_unadjusted(cx, e);
    let ety = ty::expr_ty(cx.tcx, e);
    let adjustment = cx.tcx.adjustments.find_copy(&e.id);
    match adjustment {
        None => { }
        Some(@ty::AutoAddEnv(ty::re_static, ast::BorrowedSigil)) => {
            llconst = C_struct([llconst, C_null(Type::opaque_box(cx).ptr_to())])
        }
        Some(@ty::AutoAddEnv(ref r, ref s)) => {
            cx.sess.span_bug(e.span, fmt!("unexpected static function: \
                                           region %? sigil %?", *r, *s))
        }
        Some(@ty::AutoDerefRef(ref adj)) => {
            let mut ty = ety;
            let mut maybe_ptr = None;
            do adj.autoderefs.times {
                let (dv, dt) = const_deref(cx, llconst, ty, false);
                maybe_ptr = Some(llconst);
                llconst = dv;
                ty = dt;
            }

            match adj.autoref {
                None => { }
                Some(ref autoref) => {
                    // Don't copy data to do a deref+ref.
                    let llptr = match maybe_ptr {
                        Some(ptr) => ptr,
                        None => const_addr_of(cx, llconst)
                    };
                    match *autoref {
                        ty::AutoUnsafe(m) |
                        ty::AutoPtr(ty::re_static, m) => {
                            assert!(m != ast::MutMutable);
                            llconst = llptr;
                        }
                        ty::AutoBorrowVec(ty::re_static, m) => {
                            assert!(m != ast::MutMutable);
                            assert_eq!(abi::slice_elt_base, 0);
                            assert_eq!(abi::slice_elt_len, 1);

                            match ty::get(ty).sty {
                                ty::ty_evec(_, ty::vstore_fixed(*)) => {
                                    let size = machine::llsize_of(cx, val_ty(llconst));
                                    llconst = C_struct([llptr, size]);
                                }
                                _ => {}
                            }
                        }
                        _ => {
                            cx.sess.span_bug(e.span,
                                             fmt!("unimplemented const \
                                                   autoref %?", autoref))
                        }
                    }
                }
            }
        }
    }

    let ety_adjusted = ty::expr_ty_adjusted(cx.tcx, e);
    let llty = type_of::sizing_type_of(cx, ety_adjusted);
    let csize = machine::llsize_of_alloc(cx, val_ty(llconst));
    let tsize = machine::llsize_of_alloc(cx, llty);
    if csize != tsize {
        unsafe {
            // XXX these values could use some context
            llvm::LLVMDumpValue(llconst);
            llvm::LLVMDumpValue(C_undef(llty));
        }
        cx.sess.bug(fmt!("const %s of type %s has size %u instead of %u",
                         e.repr(cx.tcx), ty_to_str(cx.tcx, ety),
                         csize, tsize));
    }
    llconst
}

fn const_expr_unadjusted(cx: @mut CrateContext, e: &ast::Expr) -> ValueRef {
    unsafe {
        let _icx = push_ctxt("const_expr");
        return match e.node {
          ast::ExprLit(lit) => consts::const_lit(cx, e, *lit),
          ast::ExprBinary(_, b, e1, e2) => {
            let te1 = const_expr(cx, e1);
            let te2 = const_expr(cx, e2);

            let te2 = base::cast_shift_const_rhs(b, te1, te2);

            /* Neither type is bottom, and we expect them to be unified
             * already, so the following is safe. */
            let ty = ty::expr_ty(cx.tcx, e1);
            let is_float = ty::type_is_fp(ty);
            let signed = ty::type_is_signed(ty);
            return match b {
              ast::BiAdd   => {
                if is_float { llvm::LLVMConstFAdd(te1, te2) }
                else        { llvm::LLVMConstAdd(te1, te2) }
              }
              ast::BiSub => {
                if is_float { llvm::LLVMConstFSub(te1, te2) }
                else        { llvm::LLVMConstSub(te1, te2) }
              }
              ast::BiMul    => {
                if is_float { llvm::LLVMConstFMul(te1, te2) }
                else        { llvm::LLVMConstMul(te1, te2) }
              }
              ast::BiDiv    => {
                if is_float    { llvm::LLVMConstFDiv(te1, te2) }
                else if signed { llvm::LLVMConstSDiv(te1, te2) }
                else           { llvm::LLVMConstUDiv(te1, te2) }
              }
              ast::BiRem    => {
                if is_float    { llvm::LLVMConstFRem(te1, te2) }
                else if signed { llvm::LLVMConstSRem(te1, te2) }
                else           { llvm::LLVMConstURem(te1, te2) }
              }
              ast::BiAnd    => llvm::LLVMConstAnd(te1, te2),
              ast::BiOr     => llvm::LLVMConstOr(te1, te2),
              ast::BiBitXor => llvm::LLVMConstXor(te1, te2),
              ast::BiBitAnd => llvm::LLVMConstAnd(te1, te2),
              ast::BiBitOr  => llvm::LLVMConstOr(te1, te2),
              ast::BiShl    => llvm::LLVMConstShl(te1, te2),
              ast::BiShr    => {
                if signed { llvm::LLVMConstAShr(te1, te2) }
                else      { llvm::LLVMConstLShr(te1, te2) }
              }
              ast::BiEq     => {
                  if is_float { ConstFCmp(RealOEQ, te1, te2) }
                  else        { ConstICmp(IntEQ, te1, te2)   }
              },
              ast::BiLt     => {
                  if is_float { ConstFCmp(RealOLT, te1, te2) }
                  else        {
                      if signed { ConstICmp(IntSLT, te1, te2) }
                      else      { ConstICmp(IntULT, te1, te2) }
                  }
              },
              ast::BiLe     => {
                  if is_float { ConstFCmp(RealOLE, te1, te2) }
                  else        {
                      if signed { ConstICmp(IntSLE, te1, te2) }
                      else      { ConstICmp(IntULE, te1, te2) }
                  }
              },
              ast::BiNe     => {
                  if is_float { ConstFCmp(RealONE, te1, te2) }
                  else        { ConstICmp(IntNE, te1, te2) }
              },
              ast::BiGe     => {
                  if is_float { ConstFCmp(RealOGE, te1, te2) }
                  else        {
                      if signed { ConstICmp(IntSGE, te1, te2) }
                      else      { ConstICmp(IntUGE, te1, te2) }
                  }
              },
              ast::BiGt     => {
                  if is_float { ConstFCmp(RealOGT, te1, te2) }
                  else        {
                      if signed { ConstICmp(IntSGT, te1, te2) }
                      else      { ConstICmp(IntUGT, te1, te2) }
                  }
              },
            };
          },
          ast::ExprUnary(_, u, e) => {
            let te = const_expr(cx, e);
            let ty = ty::expr_ty(cx.tcx, e);
            let is_float = ty::type_is_fp(ty);
            return match u {
              ast::UnBox(_)  |
              ast::UnUniq |
              ast::UnDeref  => {
                let (dv, _dt) = const_deref(cx, te, ty, true);
                dv
              }
              ast::UnNot    => {
                match ty::get(ty).sty {
                    ty::ty_bool => {
                        // Somewhat questionable, but I believe this is
                        // correct.
                        let te = llvm::LLVMConstTrunc(te, Type::i1().to_ref());
                        let te = llvm::LLVMConstNot(te);
                        llvm::LLVMConstZExt(te, Type::bool().to_ref())
                    }
                    _ => llvm::LLVMConstNot(te),
                }
              }
              ast::UnNeg    => {
                if is_float { llvm::LLVMConstFNeg(te) }
                else        { llvm::LLVMConstNeg(te) }
              }
            }
          }
          ast::ExprField(base, field, _) => {
              let bt = ty::expr_ty_adjusted(cx.tcx, base);
              let brepr = adt::represent_type(cx, bt);
              let bv = const_expr(cx, base);
              do expr::with_field_tys(cx.tcx, bt, None) |discr, field_tys| {
                  let ix = ty::field_idx_strict(cx.tcx, field, field_tys);
                  adt::const_get_field(cx, brepr, bv, discr, ix)
              }
          }

          ast::ExprIndex(_, base, index) => {
              let bt = ty::expr_ty_adjusted(cx.tcx, base);
              let bv = const_expr(cx, base);
              let iv = match const_eval::eval_const_expr(cx.tcx, index) {
                  const_eval::const_int(i) => i as u64,
                  const_eval::const_uint(u) => u,
                  _ => cx.sess.span_bug(index.span,
                                        "index is not an integer-constant expression")
              };
              let (arr, len) = match ty::get(bt).sty {
                  ty::ty_evec(_, vstore) | ty::ty_estr(vstore) =>
                      match vstore {
                      ty::vstore_fixed(u) =>
                          (bv, C_uint(cx, u)),

                      ty::vstore_slice(_) => {
                          let unit_ty = ty::sequence_element_type(cx.tcx, bt);
                          let llunitty = type_of::type_of(cx, unit_ty);
                          let unit_sz = machine::llsize_of(cx, llunitty);

                          let e1 = const_get_elt(cx, bv, [0]);
                          (const_deref_ptr(cx, e1),
                           llvm::LLVMConstUDiv(const_get_elt(cx, bv, [1]),
                                               unit_sz))
                      },
                      _ => cx.sess.span_bug(base.span,
                                            "index-expr base must be fixed-size or slice")
                  },
                  _ =>  cx.sess.span_bug(base.span,
                                         "index-expr base must be a vector or string type")
              };

              let len = llvm::LLVMConstIntGetZExtValue(len) as u64;
              let len = match ty::get(bt).sty {
                  ty::ty_estr(*) => {assert!(len > 0); len - 1},
                  _ => len
              };
              if iv >= len {
                  // FIXME #3170: report this earlier on in the const-eval
                  // pass. Reporting here is a bit late.
                  cx.sess.span_err(e.span,
                                   "const index-expr is out of bounds");
              }
              const_get_elt(cx, arr, [iv as c_uint])
          }
          ast::ExprCast(base, _) => {
            let ety = ty::expr_ty(cx.tcx, e);
            let llty = type_of::type_of(cx, ety);
            let basety = ty::expr_ty(cx.tcx, base);
            let v = const_expr(cx, base);
            match (expr::cast_type_kind(basety),
                   expr::cast_type_kind(ety)) {

              (expr::cast_integral, expr::cast_integral) => {
                let s = ty::type_is_signed(basety) as Bool;
                llvm::LLVMConstIntCast(v, llty.to_ref(), s)
              }
              (expr::cast_integral, expr::cast_float) => {
                if ty::type_is_signed(basety) {
                    llvm::LLVMConstSIToFP(v, llty.to_ref())
                } else {
                    llvm::LLVMConstUIToFP(v, llty.to_ref())
                }
              }
              (expr::cast_float, expr::cast_float) => {
                llvm::LLVMConstFPCast(v, llty.to_ref())
              }
              (expr::cast_float, expr::cast_integral) => {
                if ty::type_is_signed(ety) { llvm::LLVMConstFPToSI(v, llty.to_ref()) }
                else { llvm::LLVMConstFPToUI(v, llty.to_ref()) }
              }
              (expr::cast_enum, expr::cast_integral) |
              (expr::cast_enum, expr::cast_float)  => {
                let repr = adt::represent_type(cx, basety);
                let discr = adt::const_get_discrim(cx, repr, v);
                let iv = C_integral(cx.int_type, discr, false);
                let ety_cast = expr::cast_type_kind(ety);
                match ety_cast {
                    expr::cast_integral => {
                        let s = ty::type_is_signed(ety) as Bool;
                        llvm::LLVMConstIntCast(iv, llty.to_ref(), s)
                    }
                    expr::cast_float => llvm::LLVMConstUIToFP(iv, llty.to_ref()),
                    _ => cx.sess.bug("enum cast destination is not \
                                      integral or float")
                }
              }
              (expr::cast_pointer, expr::cast_pointer) => {
                llvm::LLVMConstPointerCast(v, llty.to_ref())
              }
              (expr::cast_integral, expr::cast_pointer) => {
                llvm::LLVMConstIntToPtr(v, llty.to_ref())
              }
              _ => {
                cx.sess.impossible_case(e.span,
                                        "bad combination of types for cast")
              }
            }
          }
          ast::ExprAddrOf(ast::MutImmutable, sub) => {
              let e = const_expr(cx, sub);
              const_addr_of(cx, e)
          }
          ast::ExprTup(ref es) => {
              let ety = ty::expr_ty(cx.tcx, e);
              let repr = adt::represent_type(cx, ety);
              let vals = es.map(|&e| const_expr(cx, e));
              adt::trans_const(cx, repr, 0, vals)
          }
          ast::ExprStruct(_, ref fs, ref base_opt) => {
              let ety = ty::expr_ty(cx.tcx, e);
              let repr = adt::represent_type(cx, ety);
              let tcx = cx.tcx;

              let base_val = match *base_opt {
                Some(base) => Some(const_expr(cx, base)),
                None => None
              };

              do expr::with_field_tys(tcx, ety, Some(e.id))
                  |discr, field_tys| {
                  let cs: ~[ValueRef] = field_tys.iter().enumerate()
                      .map(|(ix, &field_ty)| {
                      match fs.iter().find(|f| field_ty.ident == f.ident) {
                          Some(f) => const_expr(cx, (*f).expr),
                          None => {
                              match base_val {
                                Some(bv) => adt::const_get_field(cx, repr, bv, discr, ix),
                                None => cx.tcx.sess.span_bug(e.span, "missing struct field")
                              }
                          }
                      }
                  }).collect();
                  adt::trans_const(cx, repr, discr, cs)
              }
          }
          ast::ExprVec(ref es, ast::MutImmutable) => {
            let (v, _, _) = const_vec(cx, e, *es);
            v
          }
          ast::ExprVstore(sub, ast::ExprVstoreSlice) => {
            match sub.node {
              ast::ExprLit(ref lit) => {
                match lit.node {
                  ast::lit_str(*) => { const_expr(cx, sub) }
                  _ => { cx.sess.span_bug(e.span, "bad const-slice lit") }
                }
              }
              ast::ExprVec(ref es, ast::MutImmutable) => {
                let (cv, sz, llunitty) = const_vec(cx, e, *es);
                let llty = val_ty(cv);
                let gv = do "const".with_c_str |name| {
                    llvm::LLVMAddGlobal(cx.llmod, llty.to_ref(), name)
                };
                llvm::LLVMSetInitializer(gv, cv);
                llvm::LLVMSetGlobalConstant(gv, True);
                SetLinkage(gv, PrivateLinkage);
                let p = const_ptrcast(cx, gv, llunitty);
                C_struct([p, sz])
              }
              _ => cx.sess.span_bug(e.span, "bad const-slice expr")
            }
          }
          ast::ExprRepeat(elem, count, _) => {
            let vec_ty = ty::expr_ty(cx.tcx, e);
            let unit_ty = ty::sequence_element_type(cx.tcx, vec_ty);
            let llunitty = type_of::type_of(cx, unit_ty);
            let n = match const_eval::eval_const_expr(cx.tcx, count) {
                const_eval::const_int(i)  => i as uint,
                const_eval::const_uint(i) => i as uint,
                _ => cx.sess.span_bug(count.span, "count must be integral const expression.")
            };
            let vs = vec::from_elem(n, const_expr(cx, elem));
            let v = if vs.iter().any(|vi| val_ty(*vi) != llunitty) {
                C_struct(vs)
            } else {
                C_array(llunitty, vs)
            };
            v
          }
          ast::ExprPath(ref pth) => {
            // Assert that there are no type parameters in this path.
            assert!(pth.segments.iter().all(|seg| seg.types.is_empty()));

            let tcx = cx.tcx;
            match tcx.def_map.find(&e.id) {
                Some(&ast::DefFn(def_id, _purity)) => {
                    if !ast_util::is_local(def_id) {
                        let ty = csearch::get_type(cx.tcx, def_id).ty;
                        base::trans_external_path(cx, def_id, ty)
                    } else {
                        assert!(ast_util::is_local(def_id));
                        base::get_item_val(cx, def_id.node)
                    }
                }
                Some(&ast::DefStatic(def_id, false)) => {
                    get_const_val(cx, def_id)
                }
                Some(&ast::DefVariant(enum_did, variant_did)) => {
                    let ety = ty::expr_ty(cx.tcx, e);
                    let repr = adt::represent_type(cx, ety);
                    let vinfo = ty::enum_variant_with_id(cx.tcx,
                                                         enum_did,
                                                         variant_did);
                    adt::trans_const(cx, repr, vinfo.disr_val, [])
                }
                Some(&ast::DefStruct(_)) => {
                    let ety = ty::expr_ty(cx.tcx, e);
                    let llty = type_of::type_of(cx, ety);
                    C_null(llty)
                }
                _ => {
                    cx.sess.span_bug(e.span, "expected a const, fn, struct, or variant def")
                }
            }
          }
          ast::ExprCall(callee, ref args, _) => {
              let tcx = cx.tcx;
              match tcx.def_map.find(&callee.id) {
                  Some(&ast::DefStruct(_)) => {
                      let ety = ty::expr_ty(cx.tcx, e);
                      let repr = adt::represent_type(cx, ety);
                      let arg_vals = args.map(|a| const_expr(cx, *a));
                      adt::trans_const(cx, repr, 0, arg_vals)
                  }
                  Some(&ast::DefVariant(enum_did, variant_did)) => {
                      let ety = ty::expr_ty(cx.tcx, e);
                      let repr = adt::represent_type(cx, ety);
                      let vinfo = ty::enum_variant_with_id(cx.tcx,
                                                           enum_did,
                                                           variant_did);
                      let arg_vals = args.map(|a| const_expr(cx, *a));
                      adt::trans_const(cx, repr, vinfo.disr_val, arg_vals)
                  }
                  _ => cx.sess.span_bug(e.span, "expected a struct or variant def")
              }
          }
          ast::ExprParen(e) => { return const_expr(cx, e); }
          _ => cx.sess.span_bug(e.span,
                  "bad constant expression type in consts::const_expr")
        };
    }
}

pub fn trans_const(ccx: @mut CrateContext, m: ast::Mutability, id: ast::NodeId) {
    unsafe {
        let _icx = push_ctxt("trans_const");
        let g = base::get_item_val(ccx, id);
        // At this point, get_item_val has already translated the
        // constant's initializer to determine its LLVM type.
        let v = ccx.const_values.get_copy(&id);
        llvm::LLVMSetInitializer(g, v);
        if m != ast::MutMutable {
            llvm::LLVMSetGlobalConstant(g, True);
        }
    }
}
