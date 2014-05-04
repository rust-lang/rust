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
use lib::llvm::{llvm, ConstFCmp, ConstICmp, SetLinkage, PrivateLinkage, ValueRef, Bool, True,
    False};
use lib::llvm::{IntEQ, IntNE, IntUGT, IntUGE, IntULT, IntULE, IntSGT, IntSGE, IntSLT, IntSLE,
    RealOEQ, RealOGT, RealOGE, RealOLT, RealOLE, RealONE};

use metadata::csearch;
use middle::const_eval;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::base::push_ctxt;
use middle::trans::closure;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::expr;
use middle::trans::inline;
use middle::trans::machine;
use middle::trans::type_::Type;
use middle::trans::type_of;
use middle::trans::debuginfo;
use middle::ty;
use util::ppaux::{Repr, ty_to_str};

use std::c_str::ToCStr;
use std::vec;
use std::vec::Vec;
use libc::c_uint;
use syntax::{ast, ast_util};

pub fn const_lit(cx: &CrateContext, e: &ast::Expr, lit: ast::Lit)
    -> ValueRef {
    let _icx = push_ctxt("trans_lit");
    match lit.node {
        ast::LitChar(i) => C_integral(Type::char(cx), i as u64, false),
        ast::LitInt(i, t) => C_integral(Type::int_from_ty(cx, t), i as u64, true),
        ast::LitUint(u, t) => C_integral(Type::uint_from_ty(cx, t), u, false),
        ast::LitIntUnsuffixed(i) => {
            let lit_int_ty = ty::node_id_to_type(cx.tcx(), e.id);
            match ty::get(lit_int_ty).sty {
                ty::ty_int(t) => {
                    C_integral(Type::int_from_ty(cx, t), i as u64, true)
                }
                ty::ty_uint(t) => {
                    C_integral(Type::uint_from_ty(cx, t), i as u64, false)
                }
                _ => cx.sess().span_bug(lit.span,
                        format!("integer literal has type {} (expected int or uint)",
                                ty_to_str(cx.tcx(), lit_int_ty)))
            }
        }
        ast::LitFloat(ref fs, t) => {
            C_floating(fs.get(), Type::float_from_ty(cx, t))
        }
        ast::LitFloatUnsuffixed(ref fs) => {
            let lit_float_ty = ty::node_id_to_type(cx.tcx(), e.id);
            match ty::get(lit_float_ty).sty {
                ty::ty_float(t) => {
                    C_floating(fs.get(), Type::float_from_ty(cx, t))
                }
                _ => {
                    cx.sess().span_bug(lit.span,
                        "floating point literal doesn't have the right type");
                }
            }
        }
        ast::LitBool(b) => C_bool(cx, b),
        ast::LitNil => C_nil(cx),
        ast::LitStr(ref s, _) => C_str_slice(cx, (*s).clone()),
        ast::LitBinary(ref data) => C_binary_slice(cx, data.as_slice()),
    }
}

pub fn const_ptrcast(cx: &CrateContext, a: ValueRef, t: Type) -> ValueRef {
    unsafe {
        let b = llvm::LLVMConstPointerCast(a, t.ptr_to().to_ref());
        assert!(cx.const_globals.borrow_mut().insert(b as int, a));
        b
    }
}

fn const_vec(cx: &CrateContext, e: &ast::Expr,
             es: &[@ast::Expr], is_local: bool) -> (ValueRef, Type, bool) {
    let vec_ty = ty::expr_ty(cx.tcx(), e);
    let unit_ty = ty::sequence_element_type(cx.tcx(), vec_ty);
    let llunitty = type_of::type_of(cx, unit_ty);
    let (vs, inlineable) = vec::unzip(es.iter().map(|e| const_expr(cx, *e, is_local)));
    // If the vector contains enums, an LLVM array won't work.
    let v = if vs.iter().any(|vi| val_ty(*vi) != llunitty) {
        C_struct(cx, vs.as_slice(), false)
    } else {
        C_array(llunitty, vs.as_slice())
    };
    (v, llunitty, inlineable.iter().fold(true, |a, &b| a && b))
}

fn const_addr_of(cx: &CrateContext, cv: ValueRef) -> ValueRef {
    unsafe {
        let gv = "const".with_c_str(|name| {
            llvm::LLVMAddGlobal(cx.llmod, val_ty(cv).to_ref(), name)
        });
        llvm::LLVMSetInitializer(gv, cv);
        llvm::LLVMSetGlobalConstant(gv, True);
        SetLinkage(gv, PrivateLinkage);
        gv
    }
}

fn const_deref_ptr(cx: &CrateContext, v: ValueRef) -> ValueRef {
    let v = match cx.const_globals.borrow().find(&(v as int)) {
        Some(&v) => v,
        None => v
    };
    unsafe {
        assert_eq!(llvm::LLVMIsGlobalConstant(v), True);
        llvm::LLVMGetInitializer(v)
    }
}

fn const_deref_newtype(cx: &CrateContext, v: ValueRef, t: ty::t)
    -> ValueRef {
    let repr = adt::represent_type(cx, t);
    adt::const_get_field(cx, &*repr, v, 0, 0)
}

fn const_deref(cx: &CrateContext, v: ValueRef, t: ty::t, explicit: bool)
    -> (ValueRef, ty::t) {
    match ty::deref(t, explicit) {
        Some(ref mt) => {
            assert!(mt.mutbl != ast::MutMutable);
            let dv = match ty::get(t).sty {
                ty::ty_ptr(mt) | ty::ty_rptr(_, mt) => {
                    match ty::get(mt.ty).sty {
                        ty::ty_vec(_, None) | ty::ty_str => cx.sess().bug("unexpected slice"),
                        _ => const_deref_ptr(cx, v),
                    }
                }
                ty::ty_enum(..) | ty::ty_struct(..) => {
                    const_deref_newtype(cx, v, t)
                }
                _ => {
                    cx.sess().bug(format!("unexpected dereferenceable type {}",
                                          ty_to_str(cx.tcx(), t)))
                }
            };
            (dv, mt.ty)
        }
        None => {
            cx.sess().bug(format!("can't dereference const of type {}",
                                  ty_to_str(cx.tcx(), t)))
        }
    }
}

pub fn get_const_val(cx: &CrateContext,
                     mut def_id: ast::DefId) -> (ValueRef, bool) {
    let contains_key = cx.const_values.borrow().contains_key(&def_id.node);
    if !ast_util::is_local(def_id) || !contains_key {
        if !ast_util::is_local(def_id) {
            def_id = inline::maybe_instantiate_inline(cx, def_id);
        }

        match cx.tcx.map.expect_item(def_id.node).node {
            ast::ItemStatic(_, ast::MutImmutable, _) => {
                trans_const(cx, ast::MutImmutable, def_id.node);
            }
            _ => {}
        }
    }

    (cx.const_values.borrow().get_copy(&def_id.node),
     !cx.non_inlineable_statics.borrow().contains(&def_id.node))
}

pub fn const_expr(cx: &CrateContext, e: &ast::Expr, is_local: bool) -> (ValueRef, bool) {
    let (llconst, inlineable) = const_expr_unadjusted(cx, e, is_local);
    let mut llconst = llconst;
    let mut inlineable = inlineable;
    let ety = ty::expr_ty(cx.tcx(), e);
    let ety_adjusted = ty::expr_ty_adjusted(cx.tcx(), e);
    let opt_adj = cx.tcx.adjustments.borrow().find_copy(&e.id);
    match opt_adj {
        None => { }
        Some(adj) => {
            match adj {
                ty::AutoAddEnv(ty::RegionTraitStore(ty::ReStatic, _)) => {
                    let def = ty::resolve_expr(cx.tcx(), e);
                    let wrapper = closure::get_wrapper_for_bare_fn(cx,
                                                                   ety_adjusted,
                                                                   def,
                                                                   llconst,
                                                                   is_local);
                    llconst = C_struct(cx, [wrapper, C_null(Type::i8p(cx))], false)
                }
                ty::AutoAddEnv(store) => {
                    cx.sess()
                      .span_bug(e.span,
                                format!("unexpected static function: {:?}",
                                        store))
                }
                ty::AutoObject(..) => {
                    cx.sess()
                      .span_unimpl(e.span,
                                   "unimplemented const coercion to trait \
                                    object");
                }
                ty::AutoDerefRef(ref adj) => {
                    let mut ty = ety;
                    let mut maybe_ptr = None;
                    for _ in range(0, adj.autoderefs) {
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
                                None => {
                                    inlineable = false;
                                    const_addr_of(cx, llconst)
                                }
                            };
                            match *autoref {
                                ty::AutoUnsafe(m) |
                                ty::AutoPtr(ty::ReStatic, m) => {
                                    assert!(m != ast::MutMutable);
                                    llconst = llptr;
                                }
                                ty::AutoBorrowVec(ty::ReStatic, m) => {
                                    assert!(m != ast::MutMutable);
                                    assert_eq!(abi::slice_elt_base, 0);
                                    assert_eq!(abi::slice_elt_len, 1);
                                    match ty::get(ty).sty {
                                        ty::ty_vec(_, Some(len)) => {
                                            llconst = C_struct(cx, [
                                                llptr,
                                                C_uint(cx, len)
                                            ], false);
                                        }
                                        _ => {}
                                    }
                                }
                                _ => {
                                    cx.sess().span_bug(e.span,
                                                       format!("unimplemented \
                                                                const autoref \
                                                                {:?}",
                                                               autoref))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let llty = type_of::sizing_type_of(cx, ety_adjusted);
    let csize = machine::llsize_of_alloc(cx, val_ty(llconst));
    let tsize = machine::llsize_of_alloc(cx, llty);
    if csize != tsize {
        unsafe {
            // FIXME these values could use some context
            llvm::LLVMDumpValue(llconst);
            llvm::LLVMDumpValue(C_undef(llty));
        }
        cx.sess().bug(format!("const {} of type {} has size {} instead of {}",
                         e.repr(cx.tcx()), ty_to_str(cx.tcx(), ety),
                         csize, tsize));
    }
    (llconst, inlineable)
}

// the bool returned is whether this expression can be inlined into other crates
// if it's assigned to a static.
fn const_expr_unadjusted(cx: &CrateContext, e: &ast::Expr,
                         is_local: bool) -> (ValueRef, bool) {
    let map_list = |exprs: &[@ast::Expr]| {
        exprs.iter().map(|&e| const_expr(cx, e, is_local))
             .fold((Vec::new(), true),
                   |(l, all_inlineable), (val, inlineable)| {
                (l.append_one(val), all_inlineable && inlineable)
             })
    };
    unsafe {
        let _icx = push_ctxt("const_expr");
        return match e.node {
          ast::ExprLit(lit) => {
              (consts::const_lit(cx, e, (*lit).clone()), true)
          }
          ast::ExprBinary(b, e1, e2) => {
            let (te1, _) = const_expr(cx, e1, is_local);
            let (te2, _) = const_expr(cx, e2, is_local);

            let te2 = base::cast_shift_const_rhs(b, te1, te2);

            /* Neither type is bottom, and we expect them to be unified
             * already, so the following is safe. */
            let ty = ty::expr_ty(cx.tcx(), e1);
            let is_float = ty::type_is_fp(ty);
            let signed = ty::type_is_signed(ty);
            return (match b {
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
            }, true)
          },
          ast::ExprUnary(u, e) => {
            let (te, _) = const_expr(cx, e, is_local);
            let ty = ty::expr_ty(cx.tcx(), e);
            let is_float = ty::type_is_fp(ty);
            return (match u {
              ast::UnBox | ast::UnUniq | ast::UnDeref => {
                let (dv, _dt) = const_deref(cx, te, ty, true);
                dv
              }
              ast::UnNot    => {
                match ty::get(ty).sty {
                    ty::ty_bool => {
                        // Somewhat questionable, but I believe this is
                        // correct.
                        let te = llvm::LLVMConstTrunc(te, Type::i1(cx).to_ref());
                        let te = llvm::LLVMConstNot(te);
                        llvm::LLVMConstZExt(te, Type::bool(cx).to_ref())
                    }
                    _ => llvm::LLVMConstNot(te),
                }
              }
              ast::UnNeg    => {
                if is_float { llvm::LLVMConstFNeg(te) }
                else        { llvm::LLVMConstNeg(te) }
              }
            }, true)
          }
          ast::ExprField(base, field, _) => {
              let bt = ty::expr_ty_adjusted(cx.tcx(), base);
              let brepr = adt::represent_type(cx, bt);
              let (bv, inlineable) = const_expr(cx, base, is_local);
              expr::with_field_tys(cx.tcx(), bt, None, |discr, field_tys| {
                  let ix = ty::field_idx_strict(cx.tcx(), field.name, field_tys);
                  (adt::const_get_field(cx, &*brepr, bv, discr, ix), inlineable)
              })
          }

          ast::ExprIndex(base, index) => {
              let bt = ty::expr_ty_adjusted(cx.tcx(), base);
              let (bv, inlineable) = const_expr(cx, base, is_local);
              let iv = match const_eval::eval_const_expr(cx.tcx(), index) {
                  const_eval::const_int(i) => i as u64,
                  const_eval::const_uint(u) => u,
                  _ => cx.sess().span_bug(index.span,
                                          "index is not an integer-constant expression")
              };
              let (arr, len) = match ty::get(bt).sty {
                  ty::ty_vec(_, Some(u)) => (bv, C_uint(cx, u)),
                  ty::ty_rptr(_, mt) => match ty::get(mt.ty).sty {
                      ty::ty_vec(_, None) | ty::ty_str => {
                          let e1 = const_get_elt(cx, bv, [0]);
                          (const_deref_ptr(cx, e1), const_get_elt(cx, bv, [1]))
                      },
                      _ => cx.sess().span_bug(base.span,
                                              "index-expr base must be a vector or string type")
                  },
                  _ => cx.sess().span_bug(base.span,
                                          "index-expr base must be a vector or string type")
              };

              let len = llvm::LLVMConstIntGetZExtValue(len) as u64;
              let len = match ty::get(bt).sty {
                  ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) => match ty::get(ty).sty {
                      ty::ty_str => {
                          assert!(len > 0);
                          len - 1
                      }
                      _ => len
                  },
                  _ => len
              };
              if iv >= len {
                  // FIXME #3170: report this earlier on in the const-eval
                  // pass. Reporting here is a bit late.
                  cx.sess().span_err(e.span,
                                     "const index-expr is out of bounds");
              }
              (const_get_elt(cx, arr, [iv as c_uint]), inlineable)
          }
          ast::ExprCast(base, _) => {
            let ety = ty::expr_ty(cx.tcx(), e);
            let llty = type_of::type_of(cx, ety);
            let basety = ty::expr_ty(cx.tcx(), base);
            let (v, inlineable) = const_expr(cx, base, is_local);
            return (match (expr::cast_type_kind(basety),
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
                let discr = adt::const_get_discrim(cx, &*repr, v);
                let iv = C_integral(cx.int_type, discr, false);
                let ety_cast = expr::cast_type_kind(ety);
                match ety_cast {
                    expr::cast_integral => {
                        let s = ty::type_is_signed(ety) as Bool;
                        llvm::LLVMConstIntCast(iv, llty.to_ref(), s)
                    }
                    expr::cast_float => llvm::LLVMConstUIToFP(iv, llty.to_ref()),
                    _ => cx.sess().bug("enum cast destination is not \
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
                cx.sess().impossible_case(e.span,
                                          "bad combination of types for cast")
              }
            }, inlineable)
          }
          ast::ExprAddrOf(ast::MutImmutable, sub) => {
              let (e, _) = const_expr(cx, sub, is_local);
              (const_addr_of(cx, e), false)
          }
          ast::ExprTup(ref es) => {
              let ety = ty::expr_ty(cx.tcx(), e);
              let repr = adt::represent_type(cx, ety);
              let (vals, inlineable) = map_list(es.as_slice());
              (adt::trans_const(cx, &*repr, 0, vals.as_slice()), inlineable)
          }
          ast::ExprStruct(_, ref fs, ref base_opt) => {
              let ety = ty::expr_ty(cx.tcx(), e);
              let repr = adt::represent_type(cx, ety);
              let tcx = cx.tcx();

              let base_val = match *base_opt {
                Some(base) => Some(const_expr(cx, base, is_local)),
                None => None
              };

              expr::with_field_tys(tcx, ety, Some(e.id), |discr, field_tys| {
                  let (cs, inlineable) = vec::unzip(field_tys.iter().enumerate()
                      .map(|(ix, &field_ty)| {
                      match fs.iter().find(|f| field_ty.ident.name == f.ident.node.name) {
                          Some(f) => const_expr(cx, (*f).expr, is_local),
                          None => {
                              match base_val {
                                Some((bv, inlineable)) => {
                                    (adt::const_get_field(cx, &*repr, bv, discr, ix),
                                     inlineable)
                                }
                                None => cx.sess().span_bug(e.span, "missing struct field")
                              }
                          }
                      }
                  }));
                  (adt::trans_const(cx, &*repr, discr, cs.as_slice()),
                   inlineable.iter().fold(true, |a, &b| a && b))
              })
          }
          ast::ExprVec(ref es) => {
            let (v, _, inlineable) = const_vec(cx,
                                               e,
                                               es.as_slice(),
                                               is_local);
            (v, inlineable)
          }
          ast::ExprVstore(sub, store @ ast::ExprVstoreSlice) |
          ast::ExprVstore(sub, store @ ast::ExprVstoreMutSlice) => {
            match sub.node {
              ast::ExprLit(ref lit) => {
                match lit.node {
                    ast::LitStr(..) => { const_expr(cx, sub, is_local) }
                    _ => { cx.sess().span_bug(e.span, "bad const-slice lit") }
                }
              }
              ast::ExprVec(ref es) => {
                let (cv, llunitty, _) = const_vec(cx,
                                                  e,
                                                  es.as_slice(),
                                                  is_local);
                let llty = val_ty(cv);
                let gv = "const".with_c_str(|name| {
                    llvm::LLVMAddGlobal(cx.llmod, llty.to_ref(), name)
                });
                llvm::LLVMSetInitializer(gv, cv);
                llvm::LLVMSetGlobalConstant(gv,
                      if store == ast::ExprVstoreMutSlice { False } else { True });
                SetLinkage(gv, PrivateLinkage);
                let p = const_ptrcast(cx, gv, llunitty);
                (C_struct(cx, [p, C_uint(cx, es.len())], false), false)
              }
              _ => cx.sess().span_bug(e.span, "bad const-slice expr")
            }
          }
          ast::ExprRepeat(elem, count) => {
            let vec_ty = ty::expr_ty(cx.tcx(), e);
            let unit_ty = ty::sequence_element_type(cx.tcx(), vec_ty);
            let llunitty = type_of::type_of(cx, unit_ty);
            let n = match const_eval::eval_const_expr(cx.tcx(), count) {
                const_eval::const_int(i)  => i as uint,
                const_eval::const_uint(i) => i as uint,
                _ => cx.sess().span_bug(count.span, "count must be integral const expression.")
            };
            let vs = Vec::from_elem(n, const_expr(cx, elem, is_local).val0());
            let v = if vs.iter().any(|vi| val_ty(*vi) != llunitty) {
                C_struct(cx, vs.as_slice(), false)
            } else {
                C_array(llunitty, vs.as_slice())
            };
            (v, true)
          }
          ast::ExprPath(ref pth) => {
            // Assert that there are no type parameters in this path.
            assert!(pth.segments.iter().all(|seg| seg.types.is_empty()));

            let opt_def = cx.tcx().def_map.borrow().find_copy(&e.id);
            match opt_def {
                Some(ast::DefFn(def_id, _fn_style)) => {
                    if !ast_util::is_local(def_id) {
                        let ty = csearch::get_type(cx.tcx(), def_id).ty;
                        (base::trans_external_path(cx, def_id, ty), true)
                    } else {
                        assert!(ast_util::is_local(def_id));
                        (base::get_item_val(cx, def_id.node), true)
                    }
                }
                Some(ast::DefStatic(def_id, false)) => {
                    get_const_val(cx, def_id)
                }
                Some(ast::DefVariant(enum_did, variant_did, _)) => {
                    let ety = ty::expr_ty(cx.tcx(), e);
                    let repr = adt::represent_type(cx, ety);
                    let vinfo = ty::enum_variant_with_id(cx.tcx(),
                                                         enum_did,
                                                         variant_did);
                    (adt::trans_const(cx, &*repr, vinfo.disr_val, []), true)
                }
                Some(ast::DefStruct(_)) => {
                    let ety = ty::expr_ty(cx.tcx(), e);
                    let llty = type_of::type_of(cx, ety);
                    (C_null(llty), true)
                }
                _ => {
                    cx.sess().span_bug(e.span, "expected a const, fn, struct, or variant def")
                }
            }
          }
          ast::ExprCall(callee, ref args) => {
              let opt_def = cx.tcx().def_map.borrow().find_copy(&callee.id);
              match opt_def {
                  Some(ast::DefStruct(_)) => {
                      let ety = ty::expr_ty(cx.tcx(), e);
                      let repr = adt::represent_type(cx, ety);
                      let (arg_vals, inlineable) = map_list(args.as_slice());
                      (adt::trans_const(cx, &*repr, 0, arg_vals.as_slice()),
                       inlineable)
                  }
                  Some(ast::DefVariant(enum_did, variant_did, _)) => {
                      let ety = ty::expr_ty(cx.tcx(), e);
                      let repr = adt::represent_type(cx, ety);
                      let vinfo = ty::enum_variant_with_id(cx.tcx(),
                                                           enum_did,
                                                           variant_did);
                      let (arg_vals, inlineable) = map_list(args.as_slice());
                      (adt::trans_const(cx,
                                        &*repr,
                                        vinfo.disr_val,
                                        arg_vals.as_slice()), inlineable)
                  }
                  _ => cx.sess().span_bug(e.span, "expected a struct or variant def")
              }
          }
          ast::ExprParen(e) => { const_expr(cx, e, is_local) }
          ast::ExprBlock(ref block) => {
            match block.expr {
                Some(ref expr) => const_expr(cx, &**expr, is_local),
                None => (C_nil(cx), true)
            }
          }
          _ => cx.sess().span_bug(e.span,
                  "bad constant expression type in consts::const_expr")
        };
    }
}

pub fn trans_const(ccx: &CrateContext, m: ast::Mutability, id: ast::NodeId) {
    unsafe {
        let _icx = push_ctxt("trans_const");
        let g = base::get_item_val(ccx, id);
        // At this point, get_item_val has already translated the
        // constant's initializer to determine its LLVM type.
        let v = ccx.const_values.borrow().get_copy(&id);
        llvm::LLVMSetInitializer(g, v);
        if m != ast::MutMutable {
            llvm::LLVMSetGlobalConstant(g, True);
        }
        debuginfo::create_global_var_metadata(ccx, id, g);
    }
}
