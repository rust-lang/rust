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
use llvm;
use llvm::{ConstFCmp, ConstICmp, SetLinkage, SetUnnamedAddr};
use llvm::{InternalLinkage, ValueRef, Bool, True};
use middle::{check_const, const_eval, def};
use trans::{adt, closure, debuginfo, expr, inline, machine};
use trans::base::{self, push_ctxt};
use trans::common::*;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use middle::subst::Substs;
use middle::ty::{self, Ty};
use util::ppaux::{Repr, ty_to_string};

use std::iter::repeat;
use libc::c_uint;
use syntax::{ast, ast_util};
use syntax::ptr::P;

pub fn const_lit(cx: &CrateContext, e: &ast::Expr, lit: &ast::Lit)
    -> ValueRef {
    let _icx = push_ctxt("trans_lit");
    debug!("const_lit: {:?}", lit);
    match lit.node {
        ast::LitByte(b) => C_integral(Type::uint_from_ty(cx, ast::TyU8), b as u64, false),
        ast::LitChar(i) => C_integral(Type::char(cx), i as u64, false),
        ast::LitInt(i, ast::SignedIntLit(t, _)) => {
            C_integral(Type::int_from_ty(cx, t), i, true)
        }
        ast::LitInt(u, ast::UnsignedIntLit(t)) => {
            C_integral(Type::uint_from_ty(cx, t), u, false)
        }
        ast::LitInt(i, ast::UnsuffixedIntLit(_)) => {
            let lit_int_ty = ty::node_id_to_type(cx.tcx(), e.id);
            match lit_int_ty.sty {
                ty::ty_int(t) => {
                    C_integral(Type::int_from_ty(cx, t), i as u64, true)
                }
                ty::ty_uint(t) => {
                    C_integral(Type::uint_from_ty(cx, t), i as u64, false)
                }
                _ => cx.sess().span_bug(lit.span,
                        &format!("integer literal has type {} (expected int \
                                 or uint)",
                                ty_to_string(cx.tcx(), lit_int_ty))[])
            }
        }
        ast::LitFloat(ref fs, t) => {
            C_floating(&fs, Type::float_from_ty(cx, t))
        }
        ast::LitFloatUnsuffixed(ref fs) => {
            let lit_float_ty = ty::node_id_to_type(cx.tcx(), e.id);
            match lit_float_ty.sty {
                ty::ty_float(t) => {
                    C_floating(&fs, Type::float_from_ty(cx, t))
                }
                _ => {
                    cx.sess().span_bug(lit.span,
                        "floating point literal doesn't have the right type");
                }
            }
        }
        ast::LitBool(b) => C_bool(cx, b),
        ast::LitStr(ref s, _) => C_str_slice(cx, (*s).clone()),
        ast::LitBinary(ref data) => {
            let g = addr_of(cx, C_bytes(cx, &data[..]), "binary", e.id);
            let base = ptrcast(g, Type::i8p(cx));
            let prev_const = cx.const_unsized().borrow_mut()
                               .insert(base, g);
            assert!(prev_const.is_none() || prev_const == Some(g));
            assert_eq!(abi::FAT_PTR_ADDR, 0);
            assert_eq!(abi::FAT_PTR_EXTRA, 1);
            C_struct(cx, &[base, C_uint(cx, data.len())], false)
        }
    }
}

pub fn ptrcast(val: ValueRef, ty: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstPointerCast(val, ty.to_ref())
    }
}

fn addr_of_mut(ccx: &CrateContext,
               cv: ValueRef,
               kind: &str,
               id: ast::NodeId)
               -> ValueRef {
    unsafe {
        let name = format!("{}{}\0", kind, id);
        let gv = llvm::LLVMAddGlobal(ccx.llmod(), val_ty(cv).to_ref(),
                                     name.as_ptr() as *const _);
        llvm::LLVMSetInitializer(gv, cv);
        SetLinkage(gv, InternalLinkage);
        SetUnnamedAddr(gv, true);
        gv
    }
}

pub fn addr_of(ccx: &CrateContext,
               cv: ValueRef,
               kind: &str,
               id: ast::NodeId)
               -> ValueRef {
    match ccx.const_globals().borrow().get(&cv) {
        Some(&gv) => return gv,
        None => {}
    }
    let gv = addr_of_mut(ccx, cv, kind, id);
    unsafe {
        llvm::LLVMSetGlobalConstant(gv, True);
    }
    ccx.const_globals().borrow_mut().insert(cv, gv);
    gv
}

fn const_deref_ptr(cx: &CrateContext, v: ValueRef) -> ValueRef {
    let v = match cx.const_unsized().borrow().get(&v) {
        Some(&v) => v,
        None => v
    };
    unsafe {
        llvm::LLVMGetInitializer(v)
    }
}

fn const_deref<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                         v: ValueRef,
                         ty: Ty<'tcx>)
                         -> (ValueRef, Ty<'tcx>) {
    match ty::deref(ty, true) {
        Some(mt) => {
            if type_is_sized(cx.tcx(), mt.ty) {
                (const_deref_ptr(cx, v), mt.ty)
            } else {
                // Derefing a fat pointer does not change the representation,
                // just the type to ty_open.
                (v, ty::mk_open(cx.tcx(), mt.ty))
            }
        }
        None => {
            cx.sess().bug(&format!("unexpected dereferenceable type {}",
                                   ty_to_string(cx.tcx(), ty))[])
        }
    }
}

pub fn get_const_expr<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                def_id: ast::DefId,
                                ref_expr: &ast::Expr)
                                -> &'tcx ast::Expr {
    let def_id = inline::maybe_instantiate_inline(ccx, def_id);

    if def_id.krate != ast::LOCAL_CRATE {
        ccx.sess().span_bug(ref_expr.span,
                            "cross crate constant could not be inlined");
    }

    let item = ccx.tcx().map.expect_item(def_id.node);
    if let ast::ItemConst(_, ref expr) = item.node {
        &**expr
    } else {
        ccx.sess().span_bug(ref_expr.span,
                            &format!("get_const_val given non-constant item {}",
                                     item.repr(ccx.tcx()))[]);
    }
}

fn get_const_val(ccx: &CrateContext,
                 def_id: ast::DefId,
                 ref_expr: &ast::Expr) -> ValueRef {
    let expr = get_const_expr(ccx, def_id, ref_expr);
    let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
    get_const_expr_as_global(ccx, expr, check_const::PURE_CONST, empty_substs)
}

pub fn get_const_expr_as_global<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                          expr: &ast::Expr,
                                          qualif: check_const::ConstQualif,
                                          param_substs: &'tcx Substs<'tcx>)
                                          -> ValueRef {
    // Special-case constants to cache a common global for all uses.
    match expr.node {
        ast::ExprPath(_) => {
            let def = ccx.tcx().def_map.borrow()[expr.id];
            match def {
                def::DefConst(def_id) => {
                    if !ccx.tcx().adjustments.borrow().contains_key(&expr.id) {
                        return get_const_val(ccx, def_id, expr);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }

    let key = (expr.id, param_substs);
    match ccx.const_values().borrow().get(&key) {
        Some(&val) => return val,
        None => {}
    }
    let val = if qualif.intersects(check_const::NON_STATIC_BORROWS) {
        // Avoid autorefs as they would create global instead of stack
        // references, even when only the latter are correct.
        let ty = monomorphize::apply_param_substs(ccx.tcx(), param_substs,
                                                  &ty::expr_ty(ccx.tcx(), expr));
        const_expr_unadjusted(ccx, expr, ty, param_substs)
    } else {
        const_expr(ccx, expr, param_substs).0
    };

    // boolean SSA values are i1, but they have to be stored in i8 slots,
    // otherwise some LLVM optimization passes don't work as expected
    let val = unsafe {
        if llvm::LLVMTypeOf(val) == Type::i1(ccx).to_ref() {
            llvm::LLVMConstZExt(val, Type::i8(ccx).to_ref())
        } else {
            val
        }
    };

    let lvalue = addr_of(ccx, val, "const", expr.id);
    ccx.const_values().borrow_mut().insert(key, lvalue);
    lvalue
}

pub fn const_expr<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                            e: &ast::Expr,
                            param_substs: &'tcx Substs<'tcx>)
                            -> (ValueRef, Ty<'tcx>) {
    let ety = monomorphize::apply_param_substs(cx.tcx(), param_substs,
                                               &ty::expr_ty(cx.tcx(), e));
    let llconst = const_expr_unadjusted(cx, e, ety, param_substs);
    let mut llconst = llconst;
    let mut ety_adjusted = monomorphize::apply_param_substs(cx.tcx(), param_substs,
                                                            &ty::expr_ty_adjusted(cx.tcx(), e));
    let opt_adj = cx.tcx().adjustments.borrow().get(&e.id).cloned();
    match opt_adj {
        Some(ty::AdjustReifyFnPointer(_def_id)) => {
            // FIXME(#19925) once fn item types are
            // zero-sized, we'll need to do something here
        }
        Some(ty::AdjustDerefRef(adj)) => {
            let mut ty = ety;
            // Save the last autoderef in case we can avoid it.
            if adj.autoderefs > 0 {
                for _ in 0..adj.autoderefs-1 {
                    let (dv, dt) = const_deref(cx, llconst, ty);
                    llconst = dv;
                    ty = dt;
                }
            }

            let second_autoref = match adj.autoref {
                None => {
                    let (dv, dt) = const_deref(cx, llconst, ty);
                    llconst = dv;

                    // If we derefed a fat pointer then we will have an
                    // open type here. So we need to update the type with
                    // the one returned from const_deref.
                    ety_adjusted = dt;
                    None
                }
                Some(ty::AutoUnsafe(_, opt_autoref)) |
                Some(ty::AutoPtr(_, _, opt_autoref)) => {
                    if adj.autoderefs == 0 {
                        // Don't copy data to do a deref+ref
                        // (i.e., skip the last auto-deref).
                        llconst = addr_of(cx, llconst, "autoref", e.id);
                    } else {
                        // Seeing as we are deref'ing here and take a reference
                        // again to make the pointer part of the far pointer below,
                        // we just skip the whole thing. We still need the type
                        // though. This works even if we don't need to deref
                        // because of byref semantics. Note that this is not just
                        // an optimisation, it is necessary for mutable vectors to
                        // work properly.
                        ty = match ty::deref(ty, true) {
                            Some(mt) => {
                                if type_is_sized(cx.tcx(), mt.ty) {
                                    mt.ty
                                } else {
                                    // Derefing a fat pointer does not change the representation,
                                    // just the type to ty_open.
                                    ty::mk_open(cx.tcx(), mt.ty)
                                }
                            }
                            None => {
                                cx.sess().bug(&format!("unexpected dereferenceable type {}",
                                                       ty_to_string(cx.tcx(), ty))[])
                            }
                        }
                    }
                    opt_autoref
                }
                Some(autoref) => {
                    cx.sess().span_bug(e.span,
                        &format!("unimplemented const first autoref {:?}", autoref)[])
                }
            };
            match second_autoref {
                None => {}
                Some(box ty::AutoUnsafe(_, None)) |
                Some(box ty::AutoPtr(_, _, None)) => {
                    llconst = addr_of(cx, llconst, "autoref", e.id);
                }
                Some(box ty::AutoUnsize(ref k)) => {
                    let unsized_ty = ty::unsize_ty(cx.tcx(), ty, k, e.span);
                    let info = expr::unsized_info(cx, k, e.id, ty, param_substs,
                        |t| ty::mk_imm_rptr(cx.tcx(), cx.tcx().mk_region(ty::ReStatic), t));

                    let base = ptrcast(llconst, type_of::type_of(cx, unsized_ty).ptr_to());
                    let prev_const = cx.const_unsized().borrow_mut()
                                       .insert(base, llconst);
                    assert!(prev_const.is_none() || prev_const == Some(llconst));
                    assert_eq!(abi::FAT_PTR_ADDR, 0);
                    assert_eq!(abi::FAT_PTR_EXTRA, 1);
                    llconst = C_struct(cx, &[base, info], false);
                }
                Some(autoref) => {
                    cx.sess().span_bug(e.span,
                        &format!("unimplemented const second autoref {:?}", autoref)[])
                }
            }
        }
        None => {}
    };

    let llty = type_of::sizing_type_of(cx, ety_adjusted);
    let csize = machine::llsize_of_alloc(cx, val_ty(llconst));
    let tsize = machine::llsize_of_alloc(cx, llty);
    if csize != tsize {
        unsafe {
            // FIXME these values could use some context
            llvm::LLVMDumpValue(llconst);
            llvm::LLVMDumpValue(C_undef(llty));
        }
        cx.sess().bug(&format!("const {} of type {} has size {} instead of {}",
                         e.repr(cx.tcx()), ty_to_string(cx.tcx(), ety_adjusted),
                         csize, tsize)[]);
    }
    (llconst, ety_adjusted)
}

fn const_expr_unadjusted<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                   e: &ast::Expr,
                                   ety: Ty<'tcx>,
                                   param_substs: &'tcx Substs<'tcx>) -> ValueRef {
    let map_list = |exprs: &[P<ast::Expr>]| {
        exprs.iter().map(|e| const_expr(cx, &**e, param_substs).0)
             .fold(Vec::new(), |mut l, val| { l.push(val); l })
    };
    unsafe {
        let _icx = push_ctxt("const_expr");
        return match e.node {
          ast::ExprLit(ref lit) => {
              const_lit(cx, e, &**lit)
          }
          ast::ExprBinary(b, ref e1, ref e2) => {
            /* Neither type is bottom, and we expect them to be unified
             * already, so the following is safe. */
            let (te1, ty) = const_expr(cx, &**e1, param_substs);
            let is_simd = ty::type_is_simd(cx.tcx(), ty);
            let intype = if is_simd {
                ty::simd_type(cx.tcx(), ty)
            } else {
                ty
            };
            let is_float = ty::type_is_fp(intype);
            let signed = ty::type_is_signed(intype);

            let (te2, _) = const_expr(cx, &**e2, param_substs);
            let te2 = base::cast_shift_const_rhs(b, te1, te2);

            return match b.node {
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
              ast::BiEq | ast::BiNe | ast::BiLt | ast::BiLe | ast::BiGt | ast::BiGe => {
                  if is_float {
                      let cmp = base::bin_op_to_fcmp_predicate(cx, b.node);
                      ConstFCmp(cmp, te1, te2)
                  } else {
                      let cmp = base::bin_op_to_icmp_predicate(cx, b.node, signed);
                      let bool_val = ConstICmp(cmp, te1, te2);
                      if is_simd {
                          // LLVM outputs an `< size x i1 >`, so we need to perform
                          // a sign extension to get the correctly sized type.
                          llvm::LLVMConstIntCast(bool_val, val_ty(te1).to_ref(), True)
                      } else {
                          bool_val
                      }
                  }
              }
            }
          },
          ast::ExprUnary(u, ref e) => {
            let (te, ty) = const_expr(cx, &**e, param_substs);
            let is_float = ty::type_is_fp(ty);
            return match u {
              ast::UnUniq | ast::UnDeref => {
                const_deref(cx, te, ty).0
              }
              ast::UnNot    => llvm::LLVMConstNot(te),
              ast::UnNeg    => {
                if is_float { llvm::LLVMConstFNeg(te) }
                else        { llvm::LLVMConstNeg(te) }
              }
            }
          }
          ast::ExprField(ref base, field) => {
              let (bv, bt) = const_expr(cx, &**base, param_substs);
              let brepr = adt::represent_type(cx, bt);
              expr::with_field_tys(cx.tcx(), bt, None, |discr, field_tys| {
                  let ix = ty::field_idx_strict(cx.tcx(), field.node.name, field_tys);
                  adt::const_get_field(cx, &*brepr, bv, discr, ix)
              })
          }
          ast::ExprTupField(ref base, idx) => {
              let (bv, bt) = const_expr(cx, &**base, param_substs);
              let brepr = adt::represent_type(cx, bt);
              expr::with_field_tys(cx.tcx(), bt, None, |discr, _| {
                  adt::const_get_field(cx, &*brepr, bv, discr, idx.node)
              })
          }

          ast::ExprIndex(ref base, ref index) => {
              let (bv, bt) = const_expr(cx, &**base, param_substs);
              let iv = match const_eval::eval_const_expr(cx.tcx(), &**index) {
                  const_eval::const_int(i) => i as u64,
                  const_eval::const_uint(u) => u,
                  _ => cx.sess().span_bug(index.span,
                                          "index is not an integer-constant expression")
              };
              let (arr, len) = match bt.sty {
                  ty::ty_vec(_, Some(u)) => (bv, C_uint(cx, u)),
                  ty::ty_open(ty) => match ty.sty {
                      ty::ty_vec(_, None) | ty::ty_str => {
                          let e1 = const_get_elt(cx, bv, &[0]);
                          (const_deref_ptr(cx, e1), const_get_elt(cx, bv, &[1]))
                      },
                      _ => cx.sess().span_bug(base.span,
                                              &format!("index-expr base must be a vector \
                                                       or string type, found {}",
                                                      ty_to_string(cx.tcx(), bt))[])
                  },
                  ty::ty_rptr(_, mt) => match mt.ty.sty {
                      ty::ty_vec(_, Some(u)) => {
                          (const_deref_ptr(cx, bv), C_uint(cx, u))
                      },
                      _ => cx.sess().span_bug(base.span,
                                              &format!("index-expr base must be a vector \
                                                       or string type, found {}",
                                                      ty_to_string(cx.tcx(), bt))[])
                  },
                  _ => cx.sess().span_bug(base.span,
                                          &format!("index-expr base must be a vector \
                                                   or string type, found {}",
                                                  ty_to_string(cx.tcx(), bt))[])
              };

              let len = llvm::LLVMConstIntGetZExtValue(len) as u64;
              let len = match bt.sty {
                  ty::ty_uniq(ty) | ty::ty_rptr(_, ty::mt{ty, ..}) => match ty.sty {
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
              const_get_elt(cx, arr, &[iv as c_uint])
          }
          ast::ExprCast(ref base, _) => {
            let llty = type_of::type_of(cx, ety);
            let (v, basety) = const_expr(cx, &**base, param_substs);
            if expr::cast_is_noop(basety, ety) {
                return v;
            }
            return match (expr::cast_type_kind(cx.tcx(), basety),
                           expr::cast_type_kind(cx.tcx(), ety)) {

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
              (expr::cast_enum, expr::cast_integral) => {
                let repr = adt::represent_type(cx, basety);
                let discr = adt::const_get_discrim(cx, &*repr, v);
                let iv = C_integral(cx.int_type(), discr, false);
                let ety_cast = expr::cast_type_kind(cx.tcx(), ety);
                match ety_cast {
                    expr::cast_integral => {
                        let s = ty::type_is_signed(ety) as Bool;
                        llvm::LLVMConstIntCast(iv, llty.to_ref(), s)
                    }
                    _ => cx.sess().bug("enum cast destination is not \
                                        integral")
                }
              }
              (expr::cast_pointer, expr::cast_pointer) => {
                ptrcast(v, llty)
              }
              (expr::cast_integral, expr::cast_pointer) => {
                llvm::LLVMConstIntToPtr(v, llty.to_ref())
              }
              (expr::cast_pointer, expr::cast_integral) => {
                llvm::LLVMConstPtrToInt(v, llty.to_ref())
              }
              _ => {
                cx.sess().impossible_case(e.span,
                                          "bad combination of types for cast")
              }
            }
          }
          ast::ExprAddrOf(ast::MutImmutable, ref sub) => {
              // If this is the address of some static, then we need to return
              // the actual address of the static itself (short circuit the rest
              // of const eval).
              let mut cur = sub;
              loop {
                  match cur.node {
                      ast::ExprParen(ref sub) => cur = sub,
                      ast::ExprBlock(ref blk) => {
                        if let Some(ref sub) = blk.expr {
                            cur = sub;
                        } else {
                            break;
                        }
                      }
                      _ => break,
                  }
              }
              let opt_def = cx.tcx().def_map.borrow().get(&cur.id).cloned();
              if let Some(def::DefStatic(def_id, _)) = opt_def {
                  return get_static_val(cx, def_id, ety);
              }

              // If this isn't the address of a static, then keep going through
              // normal constant evaluation.
              let (v, _) = const_expr(cx, &**sub, param_substs);
              addr_of(cx, v, "ref", e.id)
          }
          ast::ExprAddrOf(ast::MutMutable, ref sub) => {
              let (v, _) = const_expr(cx, &**sub, param_substs);
              addr_of_mut(cx, v, "ref_mut_slice", e.id)
          }
          ast::ExprTup(ref es) => {
              let repr = adt::represent_type(cx, ety);
              let vals = map_list(&es[..]);
              adt::trans_const(cx, &*repr, 0, &vals[..])
          }
          ast::ExprStruct(_, ref fs, ref base_opt) => {
              let repr = adt::represent_type(cx, ety);

              let base_val = match *base_opt {
                Some(ref base) => Some(const_expr(cx, &**base, param_substs)),
                None => None
              };

              expr::with_field_tys(cx.tcx(), ety, Some(e.id), |discr, field_tys| {
                  let cs = field_tys.iter().enumerate()
                                    .map(|(ix, &field_ty)| {
                      match fs.iter().find(|f| field_ty.name == f.ident.node.name) {
                          Some(ref f) => const_expr(cx, &*f.expr, param_substs).0,
                          None => {
                              match base_val {
                                  Some((bv, _)) => {
                                      adt::const_get_field(cx, &*repr, bv,
                                                           discr, ix)
                                  }
                                  None => {
                                      cx.sess().span_bug(e.span,
                                                         "missing struct field")
                                  }
                              }
                          }
                      }
                  }).collect::<Vec<_>>();
                  if ty::type_is_simd(cx.tcx(), ety) {
                      C_vector(&cs[..])
                  } else {
                      adt::trans_const(cx, &*repr, discr, &cs[..])
                  }
              })
          }
          ast::ExprVec(ref es) => {
            let unit_ty = ty::sequence_element_type(cx.tcx(), ety);
            let llunitty = type_of::type_of(cx, unit_ty);
            let vs = es.iter().map(|e| const_expr(cx, &**e, param_substs).0)
                              .collect::<Vec<_>>();
            // If the vector contains enums, an LLVM array won't work.
            if vs.iter().any(|vi| val_ty(*vi) != llunitty) {
                C_struct(cx, &vs[..], false)
            } else {
                C_array(llunitty, &vs[..])
            }
          }
          ast::ExprRepeat(ref elem, ref count) => {
            let unit_ty = ty::sequence_element_type(cx.tcx(), ety);
            let llunitty = type_of::type_of(cx, unit_ty);
            let n = match const_eval::eval_const_expr(cx.tcx(), &**count) {
                const_eval::const_int(i)  => i as uint,
                const_eval::const_uint(i) => i as uint,
                _ => cx.sess().span_bug(count.span, "count must be integral const expression.")
            };
            let unit_val = const_expr(cx, &**elem, param_substs).0;
            let vs: Vec<_> = repeat(unit_val).take(n).collect();
            if val_ty(unit_val) != llunitty {
                C_struct(cx, &vs[..], false)
            } else {
                C_array(llunitty, &vs[..])
            }
          }
          ast::ExprPath(_) | ast::ExprQPath(_) => {
            let def = cx.tcx().def_map.borrow()[e.id];
            match def {
                def::DefFn(..) | def::DefStaticMethod(..) | def::DefMethod(..) => {
                    expr::trans_def_fn_unadjusted(cx, e, def, param_substs).val
                }
                def::DefConst(def_id) => {
                    const_deref_ptr(cx, get_const_val(cx, def_id, e))
                }
                def::DefVariant(enum_did, variant_did, _) => {
                    let vinfo = ty::enum_variant_with_id(cx.tcx(),
                                                         enum_did,
                                                         variant_did);
                    if vinfo.args.len() > 0 {
                        // N-ary variant.
                        expr::trans_def_fn_unadjusted(cx, e, def, param_substs).val
                    } else {
                        // Nullary variant.
                        let repr = adt::represent_type(cx, ety);
                        adt::trans_const(cx, &*repr, vinfo.disr_val, &[])
                    }
                }
                def::DefStruct(_) => {
                    if let ty::ty_bare_fn(..) = ety.sty {
                        // Tuple struct.
                        expr::trans_def_fn_unadjusted(cx, e, def, param_substs).val
                    } else {
                        // Unit struct.
                        C_null(type_of::type_of(cx, ety))
                    }
                }
                _ => {
                    cx.sess().span_bug(e.span, "expected a const, fn, struct, \
                                                or variant def")
                }
            }
          }
          ast::ExprCall(ref callee, ref args) => {
              let opt_def = cx.tcx().def_map.borrow().get(&callee.id).cloned();
              let arg_vals = map_list(&args[..]);
              match opt_def {
                  Some(def::DefStruct(_)) => {
                      if ty::type_is_simd(cx.tcx(), ety) {
                          C_vector(&arg_vals[..])
                      } else {
                          let repr = adt::represent_type(cx, ety);
                          adt::trans_const(cx, &*repr, 0, &arg_vals[..])
                      }
                  }
                  Some(def::DefVariant(enum_did, variant_did, _)) => {
                      let repr = adt::represent_type(cx, ety);
                      let vinfo = ty::enum_variant_with_id(cx.tcx(),
                                                           enum_did,
                                                           variant_did);
                      adt::trans_const(cx,
                                       &*repr,
                                       vinfo.disr_val,
                                       &arg_vals[..])
                  }
                  _ => cx.sess().span_bug(e.span, "expected a struct or variant def")
              }
          }
          ast::ExprParen(ref e) => const_expr(cx, &**e, param_substs).0,
          ast::ExprBlock(ref block) => {
            match block.expr {
                Some(ref expr) => const_expr(cx, &**expr, param_substs).0,
                None => C_nil(cx)
            }
          }
          ast::ExprClosure(_, ref decl, ref body) => {
            closure::trans_closure_expr(closure::Dest::Ignore(cx),
                                        &**decl, &**body, e.id,
                                        param_substs);
            C_null(type_of::type_of(cx, ety))
          }
          _ => cx.sess().span_bug(e.span,
                  "bad constant expression type in consts::const_expr")
        };
    }
}

pub fn trans_static(ccx: &CrateContext, m: ast::Mutability, id: ast::NodeId) {
    unsafe {
        let _icx = push_ctxt("trans_static");
        let g = base::get_item_val(ccx, id);
        // At this point, get_item_val has already translated the
        // constant's initializer to determine its LLVM type.
        let v = ccx.static_values().borrow()[id].clone();
        // boolean SSA values are i1, but they have to be stored in i8 slots,
        // otherwise some LLVM optimization passes don't work as expected
        let v = if llvm::LLVMTypeOf(v) == Type::i1(ccx).to_ref() {
            llvm::LLVMConstZExt(v, Type::i8(ccx).to_ref())
        } else {
            v
        };
        llvm::LLVMSetInitializer(g, v);

        // As an optimization, all shared statics which do not have interior
        // mutability are placed into read-only memory.
        if m != ast::MutMutable {
            let node_ty = ty::node_id_to_type(ccx.tcx(), id);
            let tcontents = ty::type_contents(ccx.tcx(), node_ty);
            if !tcontents.interior_unsafe() {
                llvm::LLVMSetGlobalConstant(g, True);
            }
        }
        debuginfo::create_global_var_metadata(ccx, id, g);
    }
}

fn get_static_val<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, did: ast::DefId,
                            ty: Ty<'tcx>) -> ValueRef {
    if ast_util::is_local(did) { return base::get_item_val(ccx, did.node) }
    base::trans_external_path(ccx, did, ty)
}
