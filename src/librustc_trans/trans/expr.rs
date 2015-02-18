// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Translation of Expressions
//!
//! The expr module handles translation of expressions. The most general
//! translation routine is `trans()`, which will translate an expression
//! into a datum. `trans_into()` is also available, which will translate
//! an expression and write the result directly into memory, sometimes
//! avoiding the need for a temporary stack slot. Finally,
//! `trans_to_lvalue()` is available if you'd like to ensure that the
//! result has cleanup scheduled.
//!
//! Internally, each of these functions dispatches to various other
//! expression functions depending on the kind of expression. We divide
//! up expressions into:
//!
//! - **Datum expressions:** Those that most naturally yield values.
//!   Examples would be `22`, `box x`, or `a + b` (when not overloaded).
//! - **DPS expressions:** Those that most naturally write into a location
//!   in memory. Examples would be `foo()` or `Point { x: 3, y: 4 }`.
//! - **Statement expressions:** That that do not generate a meaningful
//!   result. Examples would be `while { ... }` or `return 44`.
//!
//! Public entry points:
//!
//! - `trans_into(bcx, expr, dest) -> bcx`: evaluates an expression,
//!   storing the result into `dest`. This is the preferred form, if you
//!   can manage it.
//!
//! - `trans(bcx, expr) -> DatumBlock`: evaluates an expression, yielding
//!   `Datum` with the result. You can then store the datum, inspect
//!   the value, etc. This may introduce temporaries if the datum is a
//!   structural type.
//!
//! - `trans_to_lvalue(bcx, expr, "...") -> DatumBlock`: evaluates an
//!   expression and ensures that the result has a cleanup associated with it,
//!   creating a temporary stack slot if necessary.
//!
//! - `trans_local_var -> Datum`: looks up a local variable or upvar.

#![allow(non_camel_case_types)]

pub use self::cast_kind::*;
pub use self::Dest::*;
use self::lazy_binop_ty::*;

use back::abi;
use llvm::{self, ValueRef};
use middle::check_const;
use middle::def;
use middle::mem_categorization::Typer;
use middle::subst::{self, Substs};
use trans::{_match, adt, asm, base, callee, closure, consts, controlflow};
use trans::base::*;
use trans::build::*;
use trans::cleanup::{self, CleanupMethods};
use trans::common::*;
use trans::datum::*;
use trans::debuginfo::{self, DebugLoc, ToDebugLoc};
use trans::glue;
use trans::machine;
use trans::meth;
use trans::monomorphize;
use trans::tvec;
use trans::type_of;
use middle::ty::{struct_fields, tup_fields};
use middle::ty::{AdjustDerefRef, AdjustReifyFnPointer, AutoUnsafe};
use middle::ty::{AutoPtr};
use middle::ty::{self, Ty};
use middle::ty::MethodCall;
use util::common::indenter;
use util::ppaux::Repr;
use trans::machine::{llsize_of, llsize_of_alloc};
use trans::type_::Type;

use syntax::{ast, ast_util, codemap};
use syntax::ptr::P;
use syntax::parse::token;
use std::iter::repeat;
use std::mem;
use std::rc::Rc;

// Destinations

// These are passed around by the code generating functions to track the
// destination of a computation's value.

#[derive(Copy, PartialEq)]
pub enum Dest {
    SaveIn(ValueRef),
    Ignore,
}

impl Dest {
    pub fn to_string(&self, ccx: &CrateContext) -> String {
        match *self {
            SaveIn(v) => format!("SaveIn({})", ccx.tn().val_to_string(v)),
            Ignore => "Ignore".to_string()
        }
    }
}

/// This function is equivalent to `trans(bcx, expr).store_to_dest(dest)` but it may generate
/// better optimized LLVM code.
pub fn trans_into<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &ast::Expr,
                              dest: Dest)
                              -> Block<'blk, 'tcx> {
    let mut bcx = bcx;

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    if bcx.tcx().adjustments.borrow().contains_key(&expr.id) {
        // use trans, which may be less efficient but
        // which will perform the adjustments:
        let datum = unpack_datum!(bcx, trans(bcx, expr));
        return datum.store_to_dest(bcx, dest, expr.id);
    }

    let qualif = bcx.tcx().const_qualif_map.borrow()[expr.id];
    if !qualif.intersects(check_const::NOT_CONST | check_const::NEEDS_DROP) {
        if !qualif.intersects(check_const::PREFER_IN_PLACE) {
            if let SaveIn(lldest) = dest {
                let global = consts::get_const_expr_as_global(bcx.ccx(), expr, qualif,
                                                            bcx.fcx.param_substs);
                // Cast pointer to destination, because constants
                // have different types.
                let lldest = PointerCast(bcx, lldest, val_ty(global));
                memcpy_ty(bcx, lldest, global, expr_ty_adjusted(bcx, expr));
            }
            // Don't do anything in the Ignore case, consts don't need drop.
            return bcx;
        } else {
            // The only way we're going to see a `const` at this point is if
            // it prefers in-place instantiation, likely because it contains
            // `[x; N]` somewhere within.
            match expr.node {
                ast::ExprPath(_) | ast::ExprQPath(_) => {
                    match bcx.def(expr.id) {
                        def::DefConst(did) => {
                            let expr = consts::get_const_expr(bcx.ccx(), did, expr);
                            // Temporarily get cleanup scopes out of the way,
                            // as they require sub-expressions to be contained
                            // inside the current AST scope.
                            // These should record no cleanups anyways, `const`
                            // can't have destructors.
                            let scopes = mem::replace(&mut *bcx.fcx.scopes.borrow_mut(),
                                                      vec![]);
                            bcx = trans_into(bcx, expr, dest);
                            let scopes = mem::replace(&mut *bcx.fcx.scopes.borrow_mut(),
                                                      scopes);
                            assert!(scopes.is_empty());
                            return bcx;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    debug!("trans_into() expr={}", expr.repr(bcx.tcx()));

    let cleanup_debug_loc = debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(),
                                                                          expr.id,
                                                                          expr.span,
                                                                          false);
    bcx.fcx.push_ast_cleanup_scope(cleanup_debug_loc);

    let kind = ty::expr_kind(bcx.tcx(), expr);
    bcx = match kind {
        ty::LvalueExpr | ty::RvalueDatumExpr => {
            trans_unadjusted(bcx, expr).store_to_dest(dest, expr.id)
        }
        ty::RvalueDpsExpr => {
            trans_rvalue_dps_unadjusted(bcx, expr, dest)
        }
        ty::RvalueStmtExpr => {
            trans_rvalue_stmt_unadjusted(bcx, expr)
        }
    };

    bcx.fcx.pop_and_trans_ast_cleanup_scope(bcx, expr.id)
}

/// Translates an expression, returning a datum (and new block) encapsulating the result. When
/// possible, it is preferred to use `trans_into`, as that may avoid creating a temporary on the
/// stack.
pub fn trans<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                         expr: &ast::Expr)
                         -> DatumBlock<'blk, 'tcx, Expr> {
    debug!("trans(expr={})", bcx.expr_to_string(expr));

    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let qualif = bcx.tcx().const_qualif_map.borrow()[expr.id];
    let adjusted_global = !qualif.intersects(check_const::NON_STATIC_BORROWS);
    let global = if !qualif.intersects(check_const::NOT_CONST | check_const::NEEDS_DROP) {
        let global = consts::get_const_expr_as_global(bcx.ccx(), expr, qualif,
                                                      bcx.fcx.param_substs);

        if qualif.intersects(check_const::HAS_STATIC_BORROWS) {
            // Is borrowed as 'static, must return lvalue.

            // Cast pointer to global, because constants have different types.
            let const_ty = expr_ty_adjusted(bcx, expr);
            let llty = type_of::type_of(bcx.ccx(), const_ty);
            let global = PointerCast(bcx, global, llty.ptr_to());
            let datum = Datum::new(global, const_ty, Lvalue);
            return DatumBlock::new(bcx, datum.to_expr_datum());
        }

        // Otherwise, keep around and perform adjustments, if needed.
        let const_ty = if adjusted_global {
            expr_ty_adjusted(bcx, expr)
        } else {
            expr_ty(bcx, expr)
        };

        // This could use a better heuristic.
        Some(if type_is_immediate(bcx.ccx(), const_ty) {
            // Cast pointer to global, because constants have different types.
            let llty = type_of::type_of(bcx.ccx(), const_ty);
            let global = PointerCast(bcx, global, llty.ptr_to());
            // Maybe just get the value directly, instead of loading it?
            immediate_rvalue(load_ty(bcx, global, const_ty), const_ty)
        } else {
            let llty = type_of::type_of(bcx.ccx(), const_ty);
            // HACK(eddyb) get around issues with lifetime intrinsics.
            let scratch = alloca_no_lifetime(bcx, llty, "const");
            let lldest = if !ty::type_is_structural(const_ty) {
                // Cast pointer to slot, because constants have different types.
                PointerCast(bcx, scratch, val_ty(global))
            } else {
                // In this case, memcpy_ty calls llvm.memcpy after casting both
                // source and destination to i8*, so we don't need any casts.
                scratch
            };
            memcpy_ty(bcx, lldest, global, const_ty);
            Datum::new(scratch, const_ty, Rvalue::new(ByRef))
        })
    } else {
        None
    };

    let cleanup_debug_loc = debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(),
                                                                          expr.id,
                                                                          expr.span,
                                                                          false);
    fcx.push_ast_cleanup_scope(cleanup_debug_loc);
    let datum = match global {
        Some(rvalue) => rvalue.to_expr_datum(),
        None => unpack_datum!(bcx, trans_unadjusted(bcx, expr))
    };
    let datum = if adjusted_global {
        datum // trans::consts already performed adjustments.
    } else {
        unpack_datum!(bcx, apply_adjustments(bcx, expr, datum))
    };
    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, expr.id);
    return DatumBlock::new(bcx, datum);
}

pub fn get_len(bcx: Block, fat_ptr: ValueRef) -> ValueRef {
    GEPi(bcx, fat_ptr, &[0, abi::FAT_PTR_EXTRA])
}

pub fn get_dataptr(bcx: Block, fat_ptr: ValueRef) -> ValueRef {
    GEPi(bcx, fat_ptr, &[0, abi::FAT_PTR_ADDR])
}

// Retrieve the information we are losing (making dynamic) in an unsizing
// adjustment.
// When making a dtor, we need to do different things depending on the
// ownership of the object.. mk_ty is a function for turning `unadjusted_ty`
// into a type to be destructed. If we want to end up with a Box pointer,
// then mk_ty should make a Box pointer (T -> Box<T>), if we want a
// borrowed reference then it should be T -> &T.
pub fn unsized_info<'a, 'tcx, F>(ccx: &CrateContext<'a, 'tcx>,
                                 kind: &ty::UnsizeKind<'tcx>,
                                 id: ast::NodeId,
                                 unadjusted_ty: Ty<'tcx>,
                                 param_substs: &'tcx subst::Substs<'tcx>,
                                 mk_ty: F) -> ValueRef where
    F: FnOnce(Ty<'tcx>) -> Ty<'tcx>,
{
    // FIXME(#19596) workaround: `|t| t` causes monomorphization recursion
    fn identity<T>(t: T) -> T { t }

    debug!("unsized_info(kind={:?}, id={}, unadjusted_ty={})",
           kind, id, unadjusted_ty.repr(ccx.tcx()));
    match kind {
        &ty::UnsizeLength(len) => C_uint(ccx, len),
        &ty::UnsizeStruct(box ref k, tp_index) => match unadjusted_ty.sty {
            ty::ty_struct(_, ref substs) => {
                let ty_substs = substs.types.get_slice(subst::TypeSpace);
                // The dtor for a field treats it like a value, so mk_ty
                // should just be the identity function.
                unsized_info(ccx, k, id, ty_substs[tp_index], param_substs, identity)
            }
            _ => ccx.sess().bug(&format!("UnsizeStruct with bad sty: {}",
                                         unadjusted_ty.repr(ccx.tcx()))[])
        },
        &ty::UnsizeVtable(ty::TyTrait { ref principal, .. }, _) => {
            // Note that we preserve binding levels here:
            let substs = principal.0.substs.with_self_ty(unadjusted_ty).erase_regions();
            let substs = ccx.tcx().mk_substs(substs);
            let trait_ref = ty::Binder(Rc::new(ty::TraitRef { def_id: principal.def_id(),
                                                             substs: substs }));
            let trait_ref = monomorphize::apply_param_substs(ccx.tcx(),
                                                             param_substs,
                                                             &trait_ref);
            let box_ty = mk_ty(unadjusted_ty);
            consts::ptrcast(meth::get_vtable(ccx, box_ty, trait_ref, param_substs),
                            Type::vtable_ptr(ccx))
        }
    }
}

/// Helper for trans that apply adjustments from `expr` to `datum`, which should be the unadjusted
/// translation of `expr`.
fn apply_adjustments<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 expr: &ast::Expr,
                                 datum: Datum<'tcx, Expr>)
                                 -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;
    let mut datum = datum;
    let adjustment = match bcx.tcx().adjustments.borrow().get(&expr.id).cloned() {
        None => {
            return DatumBlock::new(bcx, datum);
        }
        Some(adj) => { adj }
    };
    debug!("unadjusted datum for expr {}: {}, adjustment={}",
           expr.repr(bcx.tcx()),
           datum.to_string(bcx.ccx()),
           adjustment.repr(bcx.tcx()));
    match adjustment {
        AdjustReifyFnPointer(_def_id) => {
            // FIXME(#19925) once fn item types are
            // zero-sized, we'll need to do something here
        }
        AdjustDerefRef(ref adj) => {
            let (autoderefs, use_autoref) = match adj.autoref {
                // Extracting a value from a box counts as a deref, but if we are
                // just converting Box<[T, ..n]> to Box<[T]> we aren't really doing
                // a deref (and wouldn't if we could treat Box like a normal struct).
                Some(ty::AutoUnsizeUniq(..)) => (adj.autoderefs - 1, true),
                // We are a bit paranoid about adjustments and thus might have a re-
                // borrow here which merely derefs and then refs again (it might have
                // a different region or mutability, but we don't care here. It might
                // also be just in case we need to unsize. But if there are no nested
                // adjustments then it should be a no-op).
                Some(ty::AutoPtr(_, _, None)) |
                Some(ty::AutoUnsafe(_, None)) if adj.autoderefs == 1 => {
                    match datum.ty.sty {
                        // Don't skip a conversion from Box<T> to &T, etc.
                        ty::ty_rptr(..) => {
                            let method_call = MethodCall::autoderef(expr.id, adj.autoderefs-1);
                            let method = bcx.tcx().method_map.borrow().get(&method_call).is_some();
                            if method {
                                // Don't skip an overloaded deref.
                                (adj.autoderefs, true)
                            } else {
                                (adj.autoderefs - 1, false)
                            }
                        }
                        _ => (adj.autoderefs, true),
                    }
                }
                _ => (adj.autoderefs, true)
            };

            if autoderefs > 0 {
                // Schedule cleanup.
                let lval = unpack_datum!(bcx, datum.to_lvalue_datum(bcx, "auto_deref", expr.id));
                datum = unpack_datum!(
                    bcx, deref_multiple(bcx, expr, lval.to_expr_datum(), autoderefs));
            }

            // (You might think there is a more elegant way to do this than a
            // use_autoref bool, but then you remember that the borrow checker exists).
            if let (true, &Some(ref a)) = (use_autoref, &adj.autoref) {
                datum = unpack_datum!(bcx, apply_autoref(a,
                                                         bcx,
                                                         expr,
                                                         datum));
            }
        }
    }
    debug!("after adjustments, datum={}", datum.to_string(bcx.ccx()));
    return DatumBlock::new(bcx, datum);

    fn apply_autoref<'blk, 'tcx>(autoref: &ty::AutoRef<'tcx>,
                                 bcx: Block<'blk, 'tcx>,
                                 expr: &ast::Expr,
                                 datum: Datum<'tcx, Expr>)
                                 -> DatumBlock<'blk, 'tcx, Expr> {
        let mut bcx = bcx;
        let mut datum = datum;

        let datum = match autoref {
            &AutoPtr(_, _, ref a) | &AutoUnsafe(_, ref a) => {
                debug!("  AutoPtr");
                if let &Some(box ref a) = a {
                    datum = unpack_datum!(bcx, apply_autoref(a, bcx, expr, datum));
                }
                if !type_is_sized(bcx.tcx(), datum.ty) {
                    // Arrange cleanup
                    let lval = unpack_datum!(bcx,
                        datum.to_lvalue_datum(bcx, "ref_fat_ptr", expr.id));
                    unpack_datum!(bcx, ref_fat_ptr(bcx, lval))
                } else {
                    unpack_datum!(bcx, auto_ref(bcx, datum, expr))
                }
            }
            &ty::AutoUnsize(ref k) => {
                debug!("  AutoUnsize");
                unpack_datum!(bcx, unsize_expr(bcx, expr, datum, k))
            }

            &ty::AutoUnsizeUniq(ty::UnsizeLength(len)) => {
                debug!("  AutoUnsizeUniq(UnsizeLength)");
                unpack_datum!(bcx, unsize_unique_vec(bcx, expr, datum, len))
            }
            &ty::AutoUnsizeUniq(ref k) => {
                debug!("  AutoUnsizeUniq");
                unpack_datum!(bcx, unsize_unique_expr(bcx, expr, datum, k))
            }
        };

        DatumBlock::new(bcx, datum)
    }

    fn unsize_expr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               expr: &ast::Expr,
                               datum: Datum<'tcx, Expr>,
                               k: &ty::UnsizeKind<'tcx>)
                               -> DatumBlock<'blk, 'tcx, Expr> {
        let mut bcx = bcx;
        let tcx = bcx.tcx();
        let datum_ty = datum.ty;
        let unsized_ty = ty::unsize_ty(tcx, datum_ty, k, expr.span);
        debug!("unsized_ty={}", unsized_ty.repr(bcx.tcx()));
        let dest_ty = ty::mk_open(tcx, unsized_ty);
        debug!("dest_ty={}", unsized_ty.repr(bcx.tcx()));

        let info = unsized_info(bcx.ccx(), k, expr.id, datum_ty, bcx.fcx.param_substs,
                                |t| ty::mk_imm_rptr(tcx, tcx.mk_region(ty::ReStatic), t));

        // Arrange cleanup
        let lval = unpack_datum!(bcx,
                                 datum.to_lvalue_datum(bcx, "into_fat_ptr", expr.id));
        // Compute the base pointer. This doesn't change the pointer value,
        // but merely its type.
        let base = match *k {
            ty::UnsizeStruct(..) | ty::UnsizeVtable(..) => {
                PointerCast(bcx, lval.val, type_of::type_of(bcx.ccx(), unsized_ty).ptr_to())
            }
            ty::UnsizeLength(..) => {
                GEPi(bcx, lval.val, &[0, 0])
            }
        };

        let scratch = rvalue_scratch_datum(bcx, dest_ty, "__fat_ptr");
        Store(bcx, base, get_dataptr(bcx, scratch.val));
        Store(bcx, info, get_len(bcx, scratch.val));

        DatumBlock::new(bcx, scratch.to_expr_datum())
    }

    fn unsize_unique_vec<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     expr: &ast::Expr,
                                     datum: Datum<'tcx, Expr>,
                                     len: uint)
                                     -> DatumBlock<'blk, 'tcx, Expr> {
        let mut bcx = bcx;
        let tcx = bcx.tcx();

        let datum_ty = datum.ty;

        debug!("unsize_unique_vec expr.id={} datum_ty={} len={}",
               expr.id, datum_ty.repr(tcx), len);

        // We do not arrange cleanup ourselves; if we already are an
        // L-value, then cleanup will have already been scheduled (and
        // the `datum.store_to` call below will emit code to zero the
        // drop flag when moving out of the L-value). If we are an R-value,
        // then we do not need to schedule cleanup.

        let ll_len = C_uint(bcx.ccx(), len);
        let unit_ty = ty::sequence_element_type(tcx, ty::type_content(datum_ty));
        let vec_ty = ty::mk_uniq(tcx, ty::mk_vec(tcx, unit_ty, None));
        let scratch = rvalue_scratch_datum(bcx, vec_ty, "__unsize_unique");

        let base = get_dataptr(bcx, scratch.val);
        let base = PointerCast(bcx,
                               base,
                               type_of::type_of(bcx.ccx(), datum_ty).ptr_to());
        bcx = datum.store_to(bcx, base);

        Store(bcx, ll_len, get_len(bcx, scratch.val));
        DatumBlock::new(bcx, scratch.to_expr_datum())
    }

    fn unsize_unique_expr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      expr: &ast::Expr,
                                      datum: Datum<'tcx, Expr>,
                                      k: &ty::UnsizeKind<'tcx>)
                                      -> DatumBlock<'blk, 'tcx, Expr> {
        let mut bcx = bcx;
        let tcx = bcx.tcx();

        let datum_ty = datum.ty;
        let unboxed_ty = match datum_ty.sty {
            ty::ty_uniq(t) => t,
            _ => bcx.sess().bug(&format!("Expected ty_uniq, found {}",
                                        bcx.ty_to_string(datum_ty))[])
        };
        let result_ty = ty::mk_uniq(tcx, ty::unsize_ty(tcx, unboxed_ty, k, expr.span));

        // We do not arrange cleanup ourselves; if we already are an
        // L-value, then cleanup will have already been scheduled (and
        // the `datum.store_to` call below will emit code to zero the
        // drop flag when moving out of the L-value). If we are an R-value,
        // then we do not need to schedule cleanup.

        let scratch = rvalue_scratch_datum(bcx, result_ty, "__uniq_fat_ptr");
        let llbox_ty = type_of::type_of(bcx.ccx(), datum_ty);
        let base = PointerCast(bcx, get_dataptr(bcx, scratch.val), llbox_ty.ptr_to());
        bcx = datum.store_to(bcx, base);

        let info = unsized_info(bcx.ccx(), k, expr.id, unboxed_ty, bcx.fcx.param_substs,
                                |t| ty::mk_uniq(tcx, t));
        Store(bcx, info, get_len(bcx, scratch.val));

        DatumBlock::new(bcx, scratch.to_expr_datum())
    }
}

/// Translates an expression in "lvalue" mode -- meaning that it returns a reference to the memory
/// that the expr represents.
///
/// If this expression is an rvalue, this implies introducing a temporary.  In other words,
/// something like `x().f` is translated into roughly the equivalent of
///
///   { tmp = x(); tmp.f }
pub fn trans_to_lvalue<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   expr: &ast::Expr,
                                   name: &str)
                                   -> DatumBlock<'blk, 'tcx, Lvalue> {
    let mut bcx = bcx;
    let datum = unpack_datum!(bcx, trans(bcx, expr));
    return datum.to_lvalue_datum(bcx, name, expr.id);
}

/// A version of `trans` that ignores adjustments. You almost certainly do not want to call this
/// directly.
fn trans_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                expr: &ast::Expr)
                                -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;

    debug!("trans_unadjusted(expr={})", bcx.expr_to_string(expr));
    let _indenter = indenter();

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    return match ty::expr_kind(bcx.tcx(), expr) {
        ty::LvalueExpr | ty::RvalueDatumExpr => {
            let datum = unpack_datum!(bcx, {
                trans_datum_unadjusted(bcx, expr)
            });

            DatumBlock {bcx: bcx, datum: datum}
        }

        ty::RvalueStmtExpr => {
            bcx = trans_rvalue_stmt_unadjusted(bcx, expr);
            nil(bcx, expr_ty(bcx, expr))
        }

        ty::RvalueDpsExpr => {
            let ty = expr_ty(bcx, expr);
            if type_is_zero_size(bcx.ccx(), ty) {
                bcx = trans_rvalue_dps_unadjusted(bcx, expr, Ignore);
                nil(bcx, ty)
            } else {
                let scratch = rvalue_scratch_datum(bcx, ty, "");
                bcx = trans_rvalue_dps_unadjusted(
                    bcx, expr, SaveIn(scratch.val));

                // Note: this is not obviously a good idea.  It causes
                // immediate values to be loaded immediately after a
                // return from a call or other similar expression,
                // which in turn leads to alloca's having shorter
                // lifetimes and hence larger stack frames.  However,
                // in turn it can lead to more register pressure.
                // Still, in practice it seems to increase
                // performance, since we have fewer problems with
                // morestack churn.
                let scratch = unpack_datum!(
                    bcx, scratch.to_appropriate_datum(bcx));

                DatumBlock::new(bcx, scratch.to_expr_datum())
            }
        }
    };

    fn nil<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, ty: Ty<'tcx>)
                       -> DatumBlock<'blk, 'tcx, Expr> {
        let llval = C_undef(type_of::type_of(bcx.ccx(), ty));
        let datum = immediate_rvalue(llval, ty);
        DatumBlock::new(bcx, datum.to_expr_datum())
    }
}

fn trans_datum_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      expr: &ast::Expr)
                                      -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let _icx = push_ctxt("trans_datum_unadjusted");

    match expr.node {
        ast::ExprParen(ref e) => {
            trans(bcx, &**e)
        }
        ast::ExprPath(_) | ast::ExprQPath(_) => {
            trans_def(bcx, expr, bcx.def(expr.id))
        }
        ast::ExprField(ref base, ident) => {
            trans_rec_field(bcx, &**base, ident.node)
        }
        ast::ExprTupField(ref base, idx) => {
            trans_rec_tup_field(bcx, &**base, idx.node)
        }
        ast::ExprIndex(ref base, ref idx) => {
            trans_index(bcx, expr, &**base, &**idx, MethodCall::expr(expr.id))
        }
        ast::ExprBox(_, ref contents) => {
            // Special case for `Box<T>`
            let box_ty = expr_ty(bcx, expr);
            let contents_ty = expr_ty(bcx, &**contents);
            match box_ty.sty {
                ty::ty_uniq(..) => {
                    trans_uniq_expr(bcx, expr, box_ty, &**contents, contents_ty)
                }
                _ => bcx.sess().span_bug(expr.span,
                                         "expected unique box")
            }

        }
        ast::ExprLit(ref lit) => trans_immediate_lit(bcx, expr, &**lit),
        ast::ExprBinary(op, ref lhs, ref rhs) => {
            trans_binary(bcx, expr, op, &**lhs, &**rhs)
        }
        ast::ExprUnary(op, ref x) => {
            trans_unary(bcx, expr, op, &**x)
        }
        ast::ExprAddrOf(_, ref x) => {
            match x.node {
                ast::ExprRepeat(..) | ast::ExprVec(..) => {
                    // Special case for slices.
                    let cleanup_debug_loc =
                        debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(),
                                                                      x.id,
                                                                      x.span,
                                                                      false);
                    fcx.push_ast_cleanup_scope(cleanup_debug_loc);
                    let datum = unpack_datum!(
                        bcx, tvec::trans_slice_vec(bcx, expr, &**x));
                    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, x.id);
                    DatumBlock::new(bcx, datum)
                }
                _ => {
                    trans_addr_of(bcx, expr, &**x)
                }
            }
        }
        ast::ExprCast(ref val, _) => {
            // Datum output mode means this is a scalar cast:
            trans_imm_cast(bcx, &**val, expr.id)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                &format!("trans_rvalue_datum_unadjusted reached \
                         fall-through case: {:?}",
                        expr.node)[]);
        }
    }
}

fn trans_field<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                              base: &ast::Expr,
                              get_idx: F)
                              -> DatumBlock<'blk, 'tcx, Expr> where
    F: FnOnce(&'blk ty::ctxt<'tcx>, &[ty::field<'tcx>]) -> uint,
{
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rec_field");

    let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, base, "field"));
    let bare_ty = ty::unopen_type(base_datum.ty);
    let repr = adt::represent_type(bcx.ccx(), bare_ty);
    with_field_tys(bcx.tcx(), bare_ty, None, move |discr, field_tys| {
        let ix = get_idx(bcx.tcx(), field_tys);
        let d = base_datum.get_element(
            bcx,
            field_tys[ix].mt.ty,
            |srcval| adt::trans_field_ptr(bcx, &*repr, srcval, discr, ix));

        if type_is_sized(bcx.tcx(), d.ty) {
            DatumBlock { datum: d.to_expr_datum(), bcx: bcx }
        } else {
            let scratch = rvalue_scratch_datum(bcx, ty::mk_open(bcx.tcx(), d.ty), "");
            Store(bcx, d.val, get_dataptr(bcx, scratch.val));
            let info = Load(bcx, get_len(bcx, base_datum.val));
            Store(bcx, info, get_len(bcx, scratch.val));

            DatumBlock::new(bcx, scratch.to_expr_datum())

        }
    })

}

/// Translates `base.field`.
fn trans_rec_field<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               base: &ast::Expr,
                               field: ast::Ident)
                               -> DatumBlock<'blk, 'tcx, Expr> {
    trans_field(bcx, base, |tcx, field_tys| ty::field_idx_strict(tcx, field.name, field_tys))
}

/// Translates `base.<idx>`.
fn trans_rec_tup_field<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   base: &ast::Expr,
                                   idx: uint)
                                   -> DatumBlock<'blk, 'tcx, Expr> {
    trans_field(bcx, base, |_, _| idx)
}

fn trans_index<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           index_expr: &ast::Expr,
                           base: &ast::Expr,
                           idx: &ast::Expr,
                           method_call: MethodCall)
                           -> DatumBlock<'blk, 'tcx, Expr> {
    //! Translates `base[idx]`.

    let _icx = push_ctxt("trans_index");
    let ccx = bcx.ccx();
    let mut bcx = bcx;

    let index_expr_debug_loc = index_expr.debug_loc();

    // Check for overloaded index.
    let method_ty = ccx.tcx()
                       .method_map
                       .borrow()
                       .get(&method_call)
                       .map(|method| method.ty);
    let elt_datum = match method_ty {
        Some(method_ty) => {
            let method_ty = monomorphize_type(bcx, method_ty);

            let base_datum = unpack_datum!(bcx, trans(bcx, base));

            // Translate index expression.
            let ix_datum = unpack_datum!(bcx, trans(bcx, idx));

            let ref_ty = // invoked methods have LB regions instantiated:
                ty::no_late_bound_regions(
                    bcx.tcx(), &ty::ty_fn_ret(method_ty)).unwrap().unwrap();
            let elt_ty = match ty::deref(ref_ty, true) {
                None => {
                    bcx.tcx().sess.span_bug(index_expr.span,
                                            "index method didn't return a \
                                             dereferenceable type?!")
                }
                Some(elt_tm) => elt_tm.ty,
            };

            // Overloaded. Evaluate `trans_overloaded_op`, which will
            // invoke the user's index() method, which basically yields
            // a `&T` pointer.  We can then proceed down the normal
            // path (below) to dereference that `&T`.
            let scratch = rvalue_scratch_datum(bcx, ref_ty, "overloaded_index_elt");
            unpack_result!(bcx,
                           trans_overloaded_op(bcx,
                                               index_expr,
                                               method_call,
                                               base_datum,
                                               vec![(ix_datum, idx.id)],
                                               Some(SaveIn(scratch.val)),
                                               true));
            let datum = scratch.to_expr_datum();
            if type_is_sized(bcx.tcx(), elt_ty) {
                Datum::new(datum.to_llscalarish(bcx), elt_ty, LvalueExpr)
            } else {
                Datum::new(datum.val, ty::mk_open(bcx.tcx(), elt_ty), LvalueExpr)
            }
        }
        None => {
            let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx,
                                                                base,
                                                                "index"));

            // Translate index expression and cast to a suitable LLVM integer.
            // Rust is less strict than LLVM in this regard.
            let ix_datum = unpack_datum!(bcx, trans(bcx, idx));
            let ix_val = ix_datum.to_llscalarish(bcx);
            let ix_size = machine::llbitsize_of_real(bcx.ccx(),
                                                     val_ty(ix_val));
            let int_size = machine::llbitsize_of_real(bcx.ccx(),
                                                      ccx.int_type());
            let ix_val = {
                if ix_size < int_size {
                    if ty::type_is_signed(expr_ty(bcx, idx)) {
                        SExt(bcx, ix_val, ccx.int_type())
                    } else { ZExt(bcx, ix_val, ccx.int_type()) }
                } else if ix_size > int_size {
                    Trunc(bcx, ix_val, ccx.int_type())
                } else {
                    ix_val
                }
            };

            let vt =
                tvec::vec_types(bcx,
                                ty::sequence_element_type(bcx.tcx(),
                                                          base_datum.ty));

            let (base, len) = base_datum.get_vec_base_and_len(bcx);

            debug!("trans_index: base {}", bcx.val_to_string(base));
            debug!("trans_index: len {}", bcx.val_to_string(len));

            let bounds_check = ICmp(bcx,
                                    llvm::IntUGE,
                                    ix_val,
                                    len,
                                    index_expr_debug_loc);
            let expect = ccx.get_intrinsic(&("llvm.expect.i1"));
            let expected = Call(bcx,
                                expect,
                                &[bounds_check, C_bool(ccx, false)],
                                None,
                                index_expr_debug_loc);
            bcx = with_cond(bcx, expected, |bcx| {
                controlflow::trans_fail_bounds_check(bcx,
                                                     expr_info(index_expr),
                                                     ix_val,
                                                     len)
            });
            let elt = InBoundsGEP(bcx, base, &[ix_val]);
            let elt = PointerCast(bcx, elt, vt.llunit_ty.ptr_to());
            Datum::new(elt, vt.unit_ty, LvalueExpr)
        }
    };

    DatumBlock::new(bcx, elt_datum)
}

fn trans_def<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                         ref_expr: &ast::Expr,
                         def: def::Def)
                         -> DatumBlock<'blk, 'tcx, Expr> {
    //! Translates a reference to a path.

    let _icx = push_ctxt("trans_def_lvalue");
    match def {
        def::DefFn(..) | def::DefStaticMethod(..) | def::DefMethod(..) |
        def::DefStruct(_) | def::DefVariant(..) => {
            let datum = trans_def_fn_unadjusted(bcx.ccx(), ref_expr, def,
                                                bcx.fcx.param_substs);
            DatumBlock::new(bcx, datum.to_expr_datum())
        }
        def::DefStatic(did, _) => {
            // There are two things that may happen here:
            //  1) If the static item is defined in this crate, it will be
            //     translated using `get_item_val`, and we return a pointer to
            //     the result.
            //  2) If the static item is defined in another crate then we add
            //     (or reuse) a declaration of an external global, and return a
            //     pointer to that.
            let const_ty = expr_ty(bcx, ref_expr);

            // For external constants, we don't inline.
            let val = if did.krate == ast::LOCAL_CRATE {
                // Case 1.

                // The LLVM global has the type of its initializer,
                // which may not be equal to the enum's type for
                // non-C-like enums.
                let val = base::get_item_val(bcx.ccx(), did.node);
                let pty = type_of::type_of(bcx.ccx(), const_ty).ptr_to();
                PointerCast(bcx, val, pty)
            } else {
                // Case 2.
                base::get_extern_const(bcx.ccx(), did, const_ty)
            };
            DatumBlock::new(bcx, Datum::new(val, const_ty, LvalueExpr))
        }
        def::DefConst(_) => {
            bcx.sess().span_bug(ref_expr.span,
                "constant expression should not reach expr::trans_def")
        }
        _ => {
            DatumBlock::new(bcx, trans_local_var(bcx, def).to_expr_datum())
        }
    }
}

fn trans_rvalue_stmt_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                            expr: &ast::Expr)
                                            -> Block<'blk, 'tcx> {
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rvalue_stmt");

    if bcx.unreachable.get() {
        return bcx;
    }

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    match expr.node {
        ast::ExprParen(ref e) => {
            trans_into(bcx, &**e, Ignore)
        }
        ast::ExprBreak(label_opt) => {
            controlflow::trans_break(bcx, expr, label_opt)
        }
        ast::ExprAgain(label_opt) => {
            controlflow::trans_cont(bcx, expr, label_opt)
        }
        ast::ExprRet(ref ex) => {
            // Check to see if the return expression itself is reachable.
            // This can occur when the inner expression contains a return
            let reachable = if let Some(ref cfg) = bcx.fcx.cfg {
                cfg.node_is_reachable(expr.id)
            } else {
                true
            };

            if reachable {
                controlflow::trans_ret(bcx, expr, ex.as_ref().map(|e| &**e))
            } else {
                // If it's not reachable, just translate the inner expression
                // directly. This avoids having to manage a return slot when
                // it won't actually be used anyway.
                if let &Some(ref x) = ex {
                    bcx = trans_into(bcx, &**x, Ignore);
                }
                // Mark the end of the block as unreachable. Once we get to
                // a return expression, there's no more we should be doing
                // after this.
                Unreachable(bcx);
                bcx
            }
        }
        ast::ExprWhile(ref cond, ref body, _) => {
            controlflow::trans_while(bcx, expr, &**cond, &**body)
        }
        ast::ExprLoop(ref body, _) => {
            controlflow::trans_loop(bcx, expr, &**body)
        }
        ast::ExprAssign(ref dst, ref src) => {
            let src_datum = unpack_datum!(bcx, trans(bcx, &**src));
            let dst_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, &**dst, "assign"));

            if type_needs_drop(bcx.tcx(), dst_datum.ty) {
                // If there are destructors involved, make sure we
                // are copying from an rvalue, since that cannot possible
                // alias an lvalue. We are concerned about code like:
                //
                //   a = a
                //
                // but also
                //
                //   a = a.b
                //
                // where e.g. a : Option<Foo> and a.b :
                // Option<Foo>. In that case, freeing `a` before the
                // assignment may also free `a.b`!
                //
                // We could avoid this intermediary with some analysis
                // to determine whether `dst` may possibly own `src`.
                debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);
                let src_datum = unpack_datum!(
                    bcx, src_datum.to_rvalue_datum(bcx, "ExprAssign"));
                bcx = glue::drop_ty(bcx,
                                    dst_datum.val,
                                    dst_datum.ty,
                                    expr.debug_loc());
                src_datum.store_to(bcx, dst_datum.val)
            } else {
                src_datum.store_to(bcx, dst_datum.val)
            }
        }
        ast::ExprAssignOp(op, ref dst, ref src) => {
            trans_assign_op(bcx, expr, op, &**dst, &**src)
        }
        ast::ExprInlineAsm(ref a) => {
            asm::trans_inline_asm(bcx, a)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                &format!("trans_rvalue_stmt_unadjusted reached \
                         fall-through case: {:?}",
                        expr.node)[]);
        }
    }
}

fn trans_rvalue_dps_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                           expr: &ast::Expr,
                                           dest: Dest)
                                           -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_rvalue_dps_unadjusted");
    let mut bcx = bcx;
    let tcx = bcx.tcx();

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    match expr.node {
        ast::ExprParen(ref e) => {
            trans_into(bcx, &**e, dest)
        }
        ast::ExprPath(_) | ast::ExprQPath(_) => {
            trans_def_dps_unadjusted(bcx, expr, bcx.def(expr.id), dest)
        }
        ast::ExprIf(ref cond, ref thn, ref els) => {
            controlflow::trans_if(bcx, expr.id, &**cond, &**thn, els.as_ref().map(|e| &**e), dest)
        }
        ast::ExprMatch(ref discr, ref arms, _) => {
            _match::trans_match(bcx, expr, &**discr, &arms[..], dest)
        }
        ast::ExprBlock(ref blk) => {
            controlflow::trans_block(bcx, &**blk, dest)
        }
        ast::ExprStruct(_, ref fields, ref base) => {
            trans_struct(bcx,
                         &fields[..],
                         base.as_ref().map(|e| &**e),
                         expr.span,
                         expr.id,
                         node_id_type(bcx, expr.id),
                         dest)
        }
        ast::ExprRange(ref start, ref end) => {
            // FIXME it is just not right that we are synthesising ast nodes in
            // trans. Shudder.
            fn make_field(field_name: &str, expr: P<ast::Expr>) -> ast::Field {
                ast::Field {
                    ident: codemap::dummy_spanned(token::str_to_ident(field_name)),
                    expr: expr,
                    span: codemap::DUMMY_SP,
                }
            }

            // A range just desugars into a struct.
            // Note that the type of the start and end may not be the same, but
            // they should only differ in their lifetime, which should not matter
            // in trans.
            let (did, fields, ty_params) = match (start, end) {
                (&Some(ref start), &Some(ref end)) => {
                    // Desugar to Range
                    let fields = vec![make_field("start", start.clone()),
                                      make_field("end", end.clone())];
                    (tcx.lang_items.range_struct(), fields, vec![node_id_type(bcx, start.id)])
                }
                (&Some(ref start), &None) => {
                    // Desugar to RangeFrom
                    let fields = vec![make_field("start", start.clone())];
                    (tcx.lang_items.range_from_struct(), fields, vec![node_id_type(bcx, start.id)])
                }
                (&None, &Some(ref end)) => {
                    // Desugar to RangeTo
                    let fields = vec![make_field("end", end.clone())];
                    (tcx.lang_items.range_to_struct(), fields, vec![node_id_type(bcx, end.id)])
                }
                _ => {
                    // Desugar to RangeFull
                    (tcx.lang_items.range_full_struct(), vec![], vec![])
                }
            };

            if let Some(did) = did {
                let substs = Substs::new_type(ty_params, vec![]);
                trans_struct(bcx,
                             &fields,
                             None,
                             expr.span,
                             expr.id,
                             ty::mk_struct(tcx, did, tcx.mk_substs(substs)),
                             dest)
            } else {
                tcx.sess.span_bug(expr.span,
                                  "No lang item for ranges (how did we get this far?)")
            }
        }
        ast::ExprTup(ref args) => {
            let numbered_fields: Vec<(uint, &ast::Expr)> =
                args.iter().enumerate().map(|(i, arg)| (i, &**arg)).collect();
            trans_adt(bcx,
                      expr_ty(bcx, expr),
                      0,
                      &numbered_fields[..],
                      None,
                      dest,
                      expr.debug_loc())
        }
        ast::ExprLit(ref lit) => {
            match lit.node {
                ast::LitStr(ref s, _) => {
                    tvec::trans_lit_str(bcx, expr, (*s).clone(), dest)
                }
                _ => {
                    bcx.tcx()
                       .sess
                       .span_bug(expr.span,
                                 "trans_rvalue_dps_unadjusted shouldn't be \
                                  translating this type of literal")
                }
            }
        }
        ast::ExprVec(..) | ast::ExprRepeat(..) => {
            tvec::trans_fixed_vstore(bcx, expr, dest)
        }
        ast::ExprClosure(_, ref decl, ref body) => {
            let dest = match dest {
                SaveIn(lldest) => closure::Dest::SaveIn(bcx, lldest),
                Ignore => closure::Dest::Ignore(bcx.ccx())
            };
            closure::trans_closure_expr(dest, &**decl, &**body, expr.id, bcx.fcx.param_substs)
                .unwrap_or(bcx)
        }
        ast::ExprCall(ref f, ref args) => {
            if bcx.tcx().is_method_call(expr.id) {
                trans_overloaded_call(bcx,
                                      expr,
                                      &**f,
                                      &args[..],
                                      Some(dest))
            } else {
                callee::trans_call(bcx,
                                   expr,
                                   &**f,
                                   callee::ArgExprs(&args[..]),
                                   dest)
            }
        }
        ast::ExprMethodCall(_, _, ref args) => {
            callee::trans_method_call(bcx,
                                      expr,
                                      &*args[0],
                                      callee::ArgExprs(&args[..]),
                                      dest)
        }
        ast::ExprBinary(op, ref lhs, ref rhs) => {
            // if not overloaded, would be RvalueDatumExpr
            let lhs = unpack_datum!(bcx, trans(bcx, &**lhs));
            let rhs_datum = unpack_datum!(bcx, trans(bcx, &**rhs));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id), lhs,
                                vec![(rhs_datum, rhs.id)], Some(dest),
                                !ast_util::is_by_value_binop(op.node)).bcx
        }
        ast::ExprUnary(op, ref subexpr) => {
            // if not overloaded, would be RvalueDatumExpr
            let arg = unpack_datum!(bcx, trans(bcx, &**subexpr));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id),
                                arg, Vec::new(), Some(dest), !ast_util::is_by_value_unop(op)).bcx
        }
        ast::ExprIndex(ref base, ref idx) => {
            // if not overloaded, would be RvalueDatumExpr
            let base = unpack_datum!(bcx, trans(bcx, &**base));
            let idx_datum = unpack_datum!(bcx, trans(bcx, &**idx));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id), base,
                                vec![(idx_datum, idx.id)], Some(dest), true).bcx
        }
        ast::ExprCast(ref val, _) => {
            // DPS output mode means this is a trait cast:
            if ty::type_is_trait(node_id_type(bcx, expr.id)) {
                let trait_ref =
                    bcx.tcx().object_cast_map.borrow()
                                             .get(&expr.id)
                                             .cloned()
                                             .unwrap();
                let trait_ref = bcx.monomorphize(&trait_ref);
                let datum = unpack_datum!(bcx, trans(bcx, &**val));
                meth::trans_trait_cast(bcx, datum, expr.id,
                                       trait_ref, dest)
            } else {
                bcx.tcx().sess.span_bug(expr.span,
                                        "expr_cast of non-trait");
            }
        }
        ast::ExprAssignOp(op, ref dst, ref src) => {
            trans_assign_op(bcx, expr, op, &**dst, &**src)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                &format!("trans_rvalue_dps_unadjusted reached fall-through \
                         case: {:?}",
                        expr.node)[]);
        }
    }
}

fn trans_def_dps_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        ref_expr: &ast::Expr,
                                        def: def::Def,
                                        dest: Dest)
                                        -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_def_dps_unadjusted");

    let lldest = match dest {
        SaveIn(lldest) => lldest,
        Ignore => { return bcx; }
    };

    match def {
        def::DefVariant(tid, vid, _) => {
            let variant_info = ty::enum_variant_with_id(bcx.tcx(), tid, vid);
            if variant_info.args.len() > 0 {
                // N-ary variant.
                let llfn = callee::trans_fn_ref(bcx.ccx(), vid,
                                                ExprId(ref_expr.id),
                                                bcx.fcx.param_substs).val;
                Store(bcx, llfn, lldest);
                return bcx;
            } else {
                // Nullary variant.
                let ty = expr_ty(bcx, ref_expr);
                let repr = adt::represent_type(bcx.ccx(), ty);
                adt::trans_set_discr(bcx, &*repr, lldest,
                                     variant_info.disr_val);
                return bcx;
            }
        }
        def::DefStruct(_) => {
            let ty = expr_ty(bcx, ref_expr);
            match ty.sty {
                ty::ty_struct(did, _) if ty::has_dtor(bcx.tcx(), did) => {
                    let repr = adt::represent_type(bcx.ccx(), ty);
                    adt::trans_set_discr(bcx, &*repr, lldest, 0);
                }
                _ => {}
            }
            bcx
        }
        _ => {
            bcx.tcx().sess.span_bug(ref_expr.span, &format!(
                "Non-DPS def {:?} referened by {}",
                def, bcx.node_id_to_string(ref_expr.id))[]);
        }
    }
}

pub fn trans_def_fn_unadjusted<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                         ref_expr: &ast::Expr,
                                         def: def::Def,
                                         param_substs: &'tcx subst::Substs<'tcx>)
                                         -> Datum<'tcx, Rvalue> {
    let _icx = push_ctxt("trans_def_datum_unadjusted");

    match def {
        def::DefFn(did, _) |
        def::DefStruct(did) | def::DefVariant(_, did, _) |
        def::DefStaticMethod(did, def::FromImpl(_)) |
        def::DefMethod(did, _, def::FromImpl(_)) => {
            callee::trans_fn_ref(ccx, did, ExprId(ref_expr.id), param_substs)
        }
        def::DefStaticMethod(impl_did, def::FromTrait(trait_did)) |
        def::DefMethod(impl_did, _, def::FromTrait(trait_did)) => {
            meth::trans_static_method_callee(ccx, impl_did,
                                             trait_did, ref_expr.id,
                                             param_substs)
        }
        _ => {
            ccx.tcx().sess.span_bug(ref_expr.span, &format!(
                    "trans_def_fn_unadjusted invoked on: {:?} for {}",
                    def,
                    ref_expr.repr(ccx.tcx()))[]);
        }
    }
}

/// Translates a reference to a local variable or argument. This always results in an lvalue datum.
pub fn trans_local_var<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   def: def::Def)
                                   -> Datum<'tcx, Lvalue> {
    let _icx = push_ctxt("trans_local_var");

    match def {
        def::DefUpvar(nid, _) => {
            // Can't move upvars, so this is never a ZeroMemLastUse.
            let local_ty = node_id_type(bcx, nid);
            match bcx.fcx.llupvars.borrow().get(&nid) {
                Some(&val) => Datum::new(val, local_ty, Lvalue),
                None => {
                    bcx.sess().bug(&format!(
                        "trans_local_var: no llval for upvar {} found",
                        nid)[]);
                }
            }
        }
        def::DefLocal(nid) => {
            let datum = match bcx.fcx.lllocals.borrow().get(&nid) {
                Some(&v) => v,
                None => {
                    bcx.sess().bug(&format!(
                        "trans_local_var: no datum for local/arg {} found",
                        nid)[]);
                }
            };
            debug!("take_local(nid={}, v={}, ty={})",
                   nid, bcx.val_to_string(datum.val), bcx.ty_to_string(datum.ty));
            datum
        }
        _ => {
            bcx.sess().unimpl(&format!(
                "unsupported def type in trans_local_var: {:?}",
                def)[]);
        }
    }
}

/// Helper for enumerating the field types of structs, enums, or records. The optional node ID here
/// is the node ID of the path identifying the enum variant in use. If none, this cannot possibly
/// an enum variant (so, if it is and `node_id_opt` is none, this function panics).
pub fn with_field_tys<'tcx, R, F>(tcx: &ty::ctxt<'tcx>,
                                  ty: Ty<'tcx>,
                                  node_id_opt: Option<ast::NodeId>,
                                  op: F)
                                  -> R where
    F: FnOnce(ty::Disr, &[ty::field<'tcx>]) -> R,
{
    match ty.sty {
        ty::ty_struct(did, substs) => {
            let fields = struct_fields(tcx, did, substs);
            let fields = monomorphize::normalize_associated_type(tcx, &fields);
            op(0, &fields[..])
        }

        ty::ty_tup(ref v) => {
            op(0, &tup_fields(&v[..])[])
        }

        ty::ty_enum(_, substs) => {
            // We want the *variant* ID here, not the enum ID.
            match node_id_opt {
                None => {
                    tcx.sess.bug(&format!(
                        "cannot get field types from the enum type {} \
                         without a node ID",
                        ty.repr(tcx))[]);
                }
                Some(node_id) => {
                    let def = tcx.def_map.borrow()[node_id].clone();
                    match def {
                        def::DefVariant(enum_id, variant_id, _) => {
                            let variant_info = ty::enum_variant_with_id(
                                tcx, enum_id, variant_id);
                            let fields = struct_fields(tcx, variant_id, substs);
                            let fields = monomorphize::normalize_associated_type(tcx, &fields);
                            op(variant_info.disr_val, &fields[..])
                        }
                        _ => {
                            tcx.sess.bug("resolve didn't map this expr to a \
                                          variant ID")
                        }
                    }
                }
            }
        }

        _ => {
            tcx.sess.bug(&format!(
                "cannot get field types from the type {}",
                ty.repr(tcx))[]);
        }
    }
}

fn trans_struct<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                            fields: &[ast::Field],
                            base: Option<&ast::Expr>,
                            expr_span: codemap::Span,
                            expr_id: ast::NodeId,
                            ty: Ty<'tcx>,
                            dest: Dest) -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_rec");

    let tcx = bcx.tcx();
    with_field_tys(tcx, ty, Some(expr_id), |discr, field_tys| {
        let mut need_base: Vec<bool> = repeat(true).take(field_tys.len()).collect();

        let numbered_fields = fields.iter().map(|field| {
            let opt_pos =
                field_tys.iter().position(|field_ty|
                                          field_ty.name == field.ident.node.name);
            let result = match opt_pos {
                Some(i) => {
                    need_base[i] = false;
                    (i, &*field.expr)
                }
                None => {
                    tcx.sess.span_bug(field.span,
                                      "Couldn't find field in struct type")
                }
            };
            result
        }).collect::<Vec<_>>();
        let optbase = match base {
            Some(base_expr) => {
                let mut leftovers = Vec::new();
                for (i, b) in need_base.iter().enumerate() {
                    if *b {
                        leftovers.push((i, field_tys[i].mt.ty));
                    }
                }
                Some(StructBaseInfo {expr: base_expr,
                                     fields: leftovers })
            }
            None => {
                if need_base.iter().any(|b| *b) {
                    tcx.sess.span_bug(expr_span, "missing fields and no base expr")
                }
                None
            }
        };

        trans_adt(bcx,
                  ty,
                  discr,
                  &numbered_fields,
                  optbase,
                  dest,
                  DebugLoc::At(expr_id, expr_span))
    })
}

/// Information that `trans_adt` needs in order to fill in the fields
/// of a struct copied from a base struct (e.g., from an expression
/// like `Foo { a: b, ..base }`.
///
/// Note that `fields` may be empty; the base expression must always be
/// evaluated for side-effects.
pub struct StructBaseInfo<'a, 'tcx> {
    /// The base expression; will be evaluated after all explicit fields.
    expr: &'a ast::Expr,
    /// The indices of fields to copy paired with their types.
    fields: Vec<(uint, Ty<'tcx>)>
}

/// Constructs an ADT instance:
///
/// - `fields` should be a list of field indices paired with the
/// expression to store into that field.  The initializers will be
/// evaluated in the order specified by `fields`.
///
/// - `optbase` contains information on the base struct (if any) from
/// which remaining fields are copied; see comments on `StructBaseInfo`.
pub fn trans_adt<'a, 'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                 ty: Ty<'tcx>,
                                 discr: ty::Disr,
                                 fields: &[(uint, &ast::Expr)],
                                 optbase: Option<StructBaseInfo<'a, 'tcx>>,
                                 dest: Dest,
                                 debug_location: DebugLoc)
                                 -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_adt");
    let fcx = bcx.fcx;
    let repr = adt::represent_type(bcx.ccx(), ty);

    debug_location.apply(bcx.fcx);

    // If we don't care about the result, just make a
    // temporary stack slot
    let addr = match dest {
        SaveIn(pos) => pos,
        Ignore => alloc_ty(bcx, ty, "temp"),
    };

    // This scope holds intermediates that must be cleaned should
    // panic occur before the ADT as a whole is ready.
    let custom_cleanup_scope = fcx.push_custom_cleanup_scope();

    // First we trans the base, if we have one, to the dest
    if let Some(base) = optbase {
        assert_eq!(discr, 0);

        match ty::expr_kind(bcx.tcx(), &*base.expr) {
            ty::RvalueDpsExpr | ty::RvalueDatumExpr if !type_needs_drop(bcx.tcx(), ty) => {
                bcx = trans_into(bcx, &*base.expr, SaveIn(addr));
            },
            ty::RvalueStmtExpr => bcx.tcx().sess.bug("unexpected expr kind for struct base expr"),
            _ => {
                let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, &*base.expr, "base"));
                for &(i, t) in &base.fields {
                    let datum = base_datum.get_element(
                            bcx, t, |srcval| adt::trans_field_ptr(bcx, &*repr, srcval, discr, i));
                    assert!(type_is_sized(bcx.tcx(), datum.ty));
                    let dest = adt::trans_field_ptr(bcx, &*repr, addr, discr, i);
                    bcx = datum.store_to(bcx, dest);
                }
            }
        }
    }

    debug_location.apply(bcx.fcx);

    if ty::type_is_simd(bcx.tcx(), ty) {
        // This is the constructor of a SIMD type, such types are
        // always primitive machine types and so do not have a
        // destructor or require any clean-up.
        let llty = type_of::type_of(bcx.ccx(), ty);

        // keep a vector as a register, and running through the field
        // `insertelement`ing them directly into that register
        // (i.e. avoid GEPi and `store`s to an alloca) .
        let mut vec_val = C_undef(llty);

        for &(i, ref e) in fields {
            let block_datum = trans(bcx, &**e);
            bcx = block_datum.bcx;
            let position = C_uint(bcx.ccx(), i);
            let value = block_datum.datum.to_llscalarish(bcx);
            vec_val = InsertElement(bcx, vec_val, value, position);
        }
        Store(bcx, vec_val, addr);
    } else {
        // Now, we just overwrite the fields we've explicitly specified
        for &(i, ref e) in fields {
            let dest = adt::trans_field_ptr(bcx, &*repr, addr, discr, i);
            let e_ty = expr_ty_adjusted(bcx, &**e);
            bcx = trans_into(bcx, &**e, SaveIn(dest));
            let scope = cleanup::CustomScope(custom_cleanup_scope);
            fcx.schedule_lifetime_end(scope, dest);
            fcx.schedule_drop_mem(scope, dest, e_ty);
        }
    }

    adt::trans_set_discr(bcx, &*repr, addr, discr);

    fcx.pop_custom_cleanup_scope(custom_cleanup_scope);

    // If we don't care about the result drop the temporary we made
    match dest {
        SaveIn(_) => bcx,
        Ignore => {
            bcx = glue::drop_ty(bcx, addr, ty, debug_location);
            base::call_lifetime_end(bcx, addr);
            bcx
        }
    }
}


fn trans_immediate_lit<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   expr: &ast::Expr,
                                   lit: &ast::Lit)
                                   -> DatumBlock<'blk, 'tcx, Expr> {
    // must not be a string constant, that is a RvalueDpsExpr
    let _icx = push_ctxt("trans_immediate_lit");
    let ty = expr_ty(bcx, expr);
    let v = consts::const_lit(bcx.ccx(), expr, lit);
    immediate_rvalue_bcx(bcx, v, ty).to_expr_datumblock()
}

fn trans_unary<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           expr: &ast::Expr,
                           op: ast::UnOp,
                           sub_expr: &ast::Expr)
                           -> DatumBlock<'blk, 'tcx, Expr> {
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_unary_datum");

    let method_call = MethodCall::expr(expr.id);

    // The only overloaded operator that is translated to a datum
    // is an overloaded deref, since it is always yields a `&T`.
    // Otherwise, we should be in the RvalueDpsExpr path.
    assert!(
        op == ast::UnDeref ||
        !ccx.tcx().method_map.borrow().contains_key(&method_call));

    let un_ty = expr_ty(bcx, expr);

    let debug_loc = expr.debug_loc();

    match op {
        ast::UnNot => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            let llresult = Not(bcx, datum.to_llscalarish(bcx), debug_loc);
            immediate_rvalue_bcx(bcx, llresult, un_ty).to_expr_datumblock()
        }
        ast::UnNeg => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            let val = datum.to_llscalarish(bcx);
            let llneg = {
                if ty::type_is_fp(un_ty) {
                    FNeg(bcx, val, debug_loc)
                } else {
                    Neg(bcx, val, debug_loc)
                }
            };
            immediate_rvalue_bcx(bcx, llneg, un_ty).to_expr_datumblock()
        }
        ast::UnUniq => {
            trans_uniq_expr(bcx, expr, un_ty, sub_expr, expr_ty(bcx, sub_expr))
        }
        ast::UnDeref => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            deref_once(bcx, expr, datum, method_call)
        }
    }
}

fn trans_uniq_expr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               box_expr: &ast::Expr,
                               box_ty: Ty<'tcx>,
                               contents: &ast::Expr,
                               contents_ty: Ty<'tcx>)
                               -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_uniq_expr");
    let fcx = bcx.fcx;
    assert!(type_is_sized(bcx.tcx(), contents_ty));
    let llty = type_of::type_of(bcx.ccx(), contents_ty);
    let size = llsize_of(bcx.ccx(), llty);
    let align = C_uint(bcx.ccx(), type_of::align_of(bcx.ccx(), contents_ty));
    let llty_ptr = llty.ptr_to();
    let Result { bcx, val } = malloc_raw_dyn(bcx,
                                             llty_ptr,
                                             box_ty,
                                             size,
                                             align,
                                             box_expr.debug_loc());
    // Unique boxes do not allocate for zero-size types. The standard library
    // may assume that `free` is never called on the pointer returned for
    // `Box<ZeroSizeType>`.
    let bcx = if llsize_of_alloc(bcx.ccx(), llty) == 0 {
        trans_into(bcx, contents, SaveIn(val))
    } else {
        let custom_cleanup_scope = fcx.push_custom_cleanup_scope();
        fcx.schedule_free_value(cleanup::CustomScope(custom_cleanup_scope),
                                val, cleanup::HeapExchange, contents_ty);
        let bcx = trans_into(bcx, contents, SaveIn(val));
        fcx.pop_custom_cleanup_scope(custom_cleanup_scope);
        bcx
    };
    immediate_rvalue_bcx(bcx, val, box_ty).to_expr_datumblock()
}

fn ref_fat_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           lval: Datum<'tcx, Lvalue>)
                           -> DatumBlock<'blk, 'tcx, Expr> {
    let dest_ty = ty::close_type(bcx.tcx(), lval.ty);
    let scratch = rvalue_scratch_datum(bcx, dest_ty, "__fat_ptr");
    memcpy_ty(bcx, scratch.val, lval.val, scratch.ty);

    DatumBlock::new(bcx, scratch.to_expr_datum())
}

fn trans_addr_of<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             expr: &ast::Expr,
                             subexpr: &ast::Expr)
                             -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_addr_of");
    let mut bcx = bcx;
    let sub_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, subexpr, "addr_of"));
    match sub_datum.ty.sty {
        ty::ty_open(_) => {
            // Opened DST value, close to a fat pointer
            ref_fat_ptr(bcx, sub_datum)
        }
        _ => {
            // Sized value, ref to a thin pointer
            let ty = expr_ty(bcx, expr);
            immediate_rvalue_bcx(bcx, sub_datum.val, ty).to_expr_datumblock()
        }
    }
}

// Important to get types for both lhs and rhs, because one might be _|_
// and the other not.
fn trans_eager_binop<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 binop_expr: &ast::Expr,
                                 binop_ty: Ty<'tcx>,
                                 op: ast::BinOp,
                                 lhs_t: Ty<'tcx>,
                                 lhs: ValueRef,
                                 rhs_t: Ty<'tcx>,
                                 rhs: ValueRef)
                                 -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_eager_binop");

    let tcx = bcx.tcx();
    let is_simd = ty::type_is_simd(tcx, lhs_t);
    let intype = if is_simd {
        ty::simd_type(tcx, lhs_t)
    } else {
        lhs_t
    };
    let is_float = ty::type_is_fp(intype);
    let is_signed = ty::type_is_signed(intype);

    let rhs = base::cast_shift_expr_rhs(bcx, op, lhs, rhs);

    let binop_debug_loc = binop_expr.debug_loc();

    let mut bcx = bcx;
    let val = match op.node {
      ast::BiAdd => {
        if is_float {
            FAdd(bcx, lhs, rhs, binop_debug_loc)
        } else {
            Add(bcx, lhs, rhs, binop_debug_loc)
        }
      }
      ast::BiSub => {
        if is_float {
            FSub(bcx, lhs, rhs, binop_debug_loc)
        } else {
            Sub(bcx, lhs, rhs, binop_debug_loc)
        }
      }
      ast::BiMul => {
        if is_float {
            FMul(bcx, lhs, rhs, binop_debug_loc)
        } else {
            Mul(bcx, lhs, rhs, binop_debug_loc)
        }
      }
      ast::BiDiv => {
        if is_float {
            FDiv(bcx, lhs, rhs, binop_debug_loc)
        } else {
            // Only zero-check integers; fp /0 is NaN
            bcx = base::fail_if_zero_or_overflows(bcx,
                                                  expr_info(binop_expr),
                                                  op,
                                                  lhs,
                                                  rhs,
                                                  rhs_t);
            if is_signed {
                SDiv(bcx, lhs, rhs, binop_debug_loc)
            } else {
                UDiv(bcx, lhs, rhs, binop_debug_loc)
            }
        }
      }
      ast::BiRem => {
        if is_float {
            FRem(bcx, lhs, rhs, binop_debug_loc)
        } else {
            // Only zero-check integers; fp %0 is NaN
            bcx = base::fail_if_zero_or_overflows(bcx,
                                                  expr_info(binop_expr),
                                                  op, lhs, rhs, rhs_t);
            if is_signed {
                SRem(bcx, lhs, rhs, binop_debug_loc)
            } else {
                URem(bcx, lhs, rhs, binop_debug_loc)
            }
        }
      }
      ast::BiBitOr => Or(bcx, lhs, rhs, binop_debug_loc),
      ast::BiBitAnd => And(bcx, lhs, rhs, binop_debug_loc),
      ast::BiBitXor => Xor(bcx, lhs, rhs, binop_debug_loc),
      ast::BiShl => Shl(bcx, lhs, rhs, binop_debug_loc),
      ast::BiShr => {
        if is_signed {
            AShr(bcx, lhs, rhs, binop_debug_loc)
        } else {
            LShr(bcx, lhs, rhs, binop_debug_loc)
        }
      }
      ast::BiEq | ast::BiNe | ast::BiLt | ast::BiGe | ast::BiLe | ast::BiGt => {
        if is_simd {
            base::compare_simd_types(bcx, lhs, rhs, intype, op.node, binop_debug_loc)
        } else {
            base::compare_scalar_types(bcx, lhs, rhs, intype, op.node, binop_debug_loc)
        }
      }
      _ => {
        bcx.tcx().sess.span_bug(binop_expr.span, "unexpected binop");
      }
    };

    immediate_rvalue_bcx(bcx, val, binop_ty).to_expr_datumblock()
}

// refinement types would obviate the need for this
enum lazy_binop_ty {
    lazy_and,
    lazy_or,
}

fn trans_lazy_binop<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                binop_expr: &ast::Expr,
                                op: lazy_binop_ty,
                                a: &ast::Expr,
                                b: &ast::Expr)
                                -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_lazy_binop");
    let binop_ty = expr_ty(bcx, binop_expr);
    let fcx = bcx.fcx;

    let DatumBlock {bcx: past_lhs, datum: lhs} = trans(bcx, a);
    let lhs = lhs.to_llscalarish(past_lhs);

    if past_lhs.unreachable.get() {
        return immediate_rvalue_bcx(past_lhs, lhs, binop_ty).to_expr_datumblock();
    }

    let join = fcx.new_id_block("join", binop_expr.id);
    let before_rhs = fcx.new_id_block("before_rhs", b.id);

    match op {
      lazy_and => CondBr(past_lhs, lhs, before_rhs.llbb, join.llbb, DebugLoc::None),
      lazy_or => CondBr(past_lhs, lhs, join.llbb, before_rhs.llbb, DebugLoc::None)
    }

    let DatumBlock {bcx: past_rhs, datum: rhs} = trans(before_rhs, b);
    let rhs = rhs.to_llscalarish(past_rhs);

    if past_rhs.unreachable.get() {
        return immediate_rvalue_bcx(join, lhs, binop_ty).to_expr_datumblock();
    }

    Br(past_rhs, join.llbb, DebugLoc::None);
    let phi = Phi(join, Type::i1(bcx.ccx()), &[lhs, rhs],
                  &[past_lhs.llbb, past_rhs.llbb]);

    return immediate_rvalue_bcx(join, phi, binop_ty).to_expr_datumblock();
}

fn trans_binary<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                            expr: &ast::Expr,
                            op: ast::BinOp,
                            lhs: &ast::Expr,
                            rhs: &ast::Expr)
                            -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_binary");
    let ccx = bcx.ccx();

    // if overloaded, would be RvalueDpsExpr
    assert!(!ccx.tcx().method_map.borrow().contains_key(&MethodCall::expr(expr.id)));

    match op.node {
        ast::BiAnd => {
            trans_lazy_binop(bcx, expr, lazy_and, lhs, rhs)
        }
        ast::BiOr => {
            trans_lazy_binop(bcx, expr, lazy_or, lhs, rhs)
        }
        _ => {
            let mut bcx = bcx;
            let lhs_datum = unpack_datum!(bcx, trans(bcx, lhs));
            let rhs_datum = unpack_datum!(bcx, trans(bcx, rhs));
            let binop_ty = expr_ty(bcx, expr);

            debug!("trans_binary (expr {}): lhs_datum={}",
                   expr.id,
                   lhs_datum.to_string(ccx));
            let lhs_ty = lhs_datum.ty;
            let lhs = lhs_datum.to_llscalarish(bcx);

            debug!("trans_binary (expr {}): rhs_datum={}",
                   expr.id,
                   rhs_datum.to_string(ccx));
            let rhs_ty = rhs_datum.ty;
            let rhs = rhs_datum.to_llscalarish(bcx);
            trans_eager_binop(bcx, expr, binop_ty, op,
                              lhs_ty, lhs, rhs_ty, rhs)
        }
    }
}

fn trans_overloaded_op<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   expr: &ast::Expr,
                                   method_call: MethodCall,
                                   lhs: Datum<'tcx, Expr>,
                                   rhs: Vec<(Datum<'tcx, Expr>, ast::NodeId)>,
                                   dest: Option<Dest>,
                                   autoref: bool)
                                   -> Result<'blk, 'tcx> {
    let method_ty = (*bcx.tcx().method_map.borrow())[method_call].ty;
    callee::trans_call_inner(bcx,
                             expr.debug_loc(),
                             monomorphize_type(bcx, method_ty),
                             |bcx, arg_cleanup_scope| {
                                meth::trans_method_callee(bcx,
                                                          method_call,
                                                          None,
                                                          arg_cleanup_scope)
                             },
                             callee::ArgOverloadedOp(lhs, rhs, autoref),
                             dest)
}

fn trans_overloaded_call<'a, 'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                         expr: &ast::Expr,
                                         callee: &'a ast::Expr,
                                         args: &'a [P<ast::Expr>],
                                         dest: Option<Dest>)
                                         -> Block<'blk, 'tcx> {
    let method_call = MethodCall::expr(expr.id);
    let method_type = (*bcx.tcx()
                           .method_map
                           .borrow())[method_call]
                           .ty;
    let mut all_args = vec!(callee);
    all_args.extend(args.iter().map(|e| &**e));
    unpack_result!(bcx,
                   callee::trans_call_inner(bcx,
                                            expr.debug_loc(),
                                            monomorphize_type(bcx,
                                                              method_type),
                                            |bcx, arg_cleanup_scope| {
                                                meth::trans_method_callee(
                                                    bcx,
                                                    method_call,
                                                    None,
                                                    arg_cleanup_scope)
                                            },
                                            callee::ArgOverloadedCall(all_args),
                                            dest));
    bcx
}

fn int_cast(bcx: Block,
            lldsttype: Type,
            llsrctype: Type,
            llsrc: ValueRef,
            signed: bool)
            -> ValueRef {
    let _icx = push_ctxt("int_cast");
    let srcsz = llsrctype.int_width();
    let dstsz = lldsttype.int_width();
    return if dstsz == srcsz {
        BitCast(bcx, llsrc, lldsttype)
    } else if srcsz > dstsz {
        TruncOrBitCast(bcx, llsrc, lldsttype)
    } else if signed {
        SExtOrBitCast(bcx, llsrc, lldsttype)
    } else {
        ZExtOrBitCast(bcx, llsrc, lldsttype)
    }
}

fn float_cast(bcx: Block,
              lldsttype: Type,
              llsrctype: Type,
              llsrc: ValueRef)
              -> ValueRef {
    let _icx = push_ctxt("float_cast");
    let srcsz = llsrctype.float_width();
    let dstsz = lldsttype.float_width();
    return if dstsz > srcsz {
        FPExt(bcx, llsrc, lldsttype)
    } else if srcsz > dstsz {
        FPTrunc(bcx, llsrc, lldsttype)
    } else { llsrc };
}

#[derive(Copy, PartialEq, Debug)]
pub enum cast_kind {
    cast_pointer,
    cast_integral,
    cast_float,
    cast_enum,
    cast_other,
}

pub fn cast_type_kind<'tcx>(tcx: &ty::ctxt<'tcx>, t: Ty<'tcx>) -> cast_kind {
    match t.sty {
        ty::ty_char        => cast_integral,
        ty::ty_float(..)   => cast_float,
        ty::ty_rptr(_, mt) | ty::ty_ptr(mt) => {
            if type_is_sized(tcx, mt.ty) {
                cast_pointer
            } else {
                cast_other
            }
        }
        ty::ty_bare_fn(..) => cast_pointer,
        ty::ty_int(..)     => cast_integral,
        ty::ty_uint(..)    => cast_integral,
        ty::ty_bool        => cast_integral,
        ty::ty_enum(..)    => cast_enum,
        _                  => cast_other
    }
}

pub fn cast_is_noop<'tcx>(t_in: Ty<'tcx>, t_out: Ty<'tcx>) -> bool {
    match (ty::deref(t_in, true), ty::deref(t_out, true)) {
        (Some(ty::mt{ ty: t_in, .. }), Some(ty::mt{ ty: t_out, .. })) => {
            t_in == t_out
        }
        _ => false
    }
}

fn trans_imm_cast<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &ast::Expr,
                              id: ast::NodeId)
                              -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_cast");
    let mut bcx = bcx;
    let ccx = bcx.ccx();

    let t_in = expr_ty(bcx, expr);
    let t_out = node_id_type(bcx, id);
    let k_in = cast_type_kind(bcx.tcx(), t_in);
    let k_out = cast_type_kind(bcx.tcx(), t_out);
    let s_in = k_in == cast_integral && ty::type_is_signed(t_in);
    let ll_t_in = type_of::arg_type_of(ccx, t_in);
    let ll_t_out = type_of::arg_type_of(ccx, t_out);

    // Convert the value to be cast into a ValueRef, either by-ref or
    // by-value as appropriate given its type:
    let mut datum = unpack_datum!(bcx, trans(bcx, expr));

    if cast_is_noop(datum.ty, t_out) {
        datum.ty = t_out;
        return DatumBlock::new(bcx, datum);
    }

    let newval = match (k_in, k_out) {
        (cast_integral, cast_integral) => {
            let llexpr = datum.to_llscalarish(bcx);
            int_cast(bcx, ll_t_out, ll_t_in, llexpr, s_in)
        }
        (cast_float, cast_float) => {
            let llexpr = datum.to_llscalarish(bcx);
            float_cast(bcx, ll_t_out, ll_t_in, llexpr)
        }
        (cast_integral, cast_float) => {
            let llexpr = datum.to_llscalarish(bcx);
            if s_in {
                SIToFP(bcx, llexpr, ll_t_out)
            } else { UIToFP(bcx, llexpr, ll_t_out) }
        }
        (cast_float, cast_integral) => {
            let llexpr = datum.to_llscalarish(bcx);
            if ty::type_is_signed(t_out) {
                FPToSI(bcx, llexpr, ll_t_out)
            } else { FPToUI(bcx, llexpr, ll_t_out) }
        }
        (cast_integral, cast_pointer) => {
            let llexpr = datum.to_llscalarish(bcx);
            IntToPtr(bcx, llexpr, ll_t_out)
        }
        (cast_pointer, cast_integral) => {
            let llexpr = datum.to_llscalarish(bcx);
            PtrToInt(bcx, llexpr, ll_t_out)
        }
        (cast_pointer, cast_pointer) => {
            let llexpr = datum.to_llscalarish(bcx);
            PointerCast(bcx, llexpr, ll_t_out)
        }
        (cast_enum, cast_integral) |
        (cast_enum, cast_float) => {
            let mut bcx = bcx;
            let repr = adt::represent_type(ccx, t_in);
            let datum = unpack_datum!(
                bcx, datum.to_lvalue_datum(bcx, "trans_imm_cast", expr.id));
            let llexpr_ptr = datum.to_llref();
            let lldiscrim_a =
                adt::trans_get_discr(bcx, &*repr, llexpr_ptr, Some(Type::i64(ccx)));
            match k_out {
                cast_integral => int_cast(bcx, ll_t_out,
                                          val_ty(lldiscrim_a),
                                          lldiscrim_a, true),
                cast_float => SIToFP(bcx, lldiscrim_a, ll_t_out),
                _ => {
                    ccx.sess().bug(&format!("translating unsupported cast: \
                                            {} ({:?}) -> {} ({:?})",
                                            t_in.repr(bcx.tcx()),
                                            k_in,
                                            t_out.repr(bcx.tcx()),
                                            k_out)[])
                }
            }
        }
        _ => ccx.sess().bug(&format!("translating unsupported cast: \
                                    {} ({:?}) -> {} ({:?})",
                                    t_in.repr(bcx.tcx()),
                                    k_in,
                                    t_out.repr(bcx.tcx()),
                                    k_out)[])
    };
    return immediate_rvalue_bcx(bcx, newval, t_out).to_expr_datumblock();
}

fn trans_assign_op<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               expr: &ast::Expr,
                               op: ast::BinOp,
                               dst: &ast::Expr,
                               src: &ast::Expr)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_assign_op");
    let mut bcx = bcx;

    debug!("trans_assign_op(expr={})", bcx.expr_to_string(expr));

    // User-defined operator methods cannot be used with `+=` etc right now
    assert!(!bcx.tcx().method_map.borrow().contains_key(&MethodCall::expr(expr.id)));

    // Evaluate LHS (destination), which should be an lvalue
    let dst_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, dst, "assign_op"));
    assert!(!type_needs_drop(bcx.tcx(), dst_datum.ty));
    let dst_ty = dst_datum.ty;
    let dst = load_ty(bcx, dst_datum.val, dst_datum.ty);

    // Evaluate RHS
    let rhs_datum = unpack_datum!(bcx, trans(bcx, &*src));
    let rhs_ty = rhs_datum.ty;
    let rhs = rhs_datum.to_llscalarish(bcx);

    // Perform computation and store the result
    let result_datum = unpack_datum!(
        bcx, trans_eager_binop(bcx, expr, dst_datum.ty, op,
                               dst_ty, dst, rhs_ty, rhs));
    return result_datum.store_to(bcx, dst_datum.val);
}

fn auto_ref<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                        datum: Datum<'tcx, Expr>,
                        expr: &ast::Expr)
                        -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;

    // Ensure cleanup of `datum` if not already scheduled and obtain
    // a "by ref" pointer.
    let lv_datum = unpack_datum!(bcx, datum.to_lvalue_datum(bcx, "autoref", expr.id));

    // Compute final type. Note that we are loose with the region and
    // mutability, since those things don't matter in trans.
    let referent_ty = lv_datum.ty;
    let ptr_ty = ty::mk_imm_rptr(bcx.tcx(), bcx.tcx().mk_region(ty::ReStatic), referent_ty);

    // Get the pointer.
    let llref = lv_datum.to_llref();

    // Construct the resulting datum, using what was the "by ref"
    // ValueRef of type `referent_ty` to be the "by value" ValueRef
    // of type `&referent_ty`.
    DatumBlock::new(bcx, Datum::new(llref, ptr_ty, RvalueExpr(Rvalue::new(ByValue))))
}

fn deref_multiple<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &ast::Expr,
                              datum: Datum<'tcx, Expr>,
                              times: uint)
                              -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;
    let mut datum = datum;
    for i in 0..times {
        let method_call = MethodCall::autoderef(expr.id, i);
        datum = unpack_datum!(bcx, deref_once(bcx, expr, datum, method_call));
    }
    DatumBlock { bcx: bcx, datum: datum }
}

fn deref_once<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                          expr: &ast::Expr,
                          datum: Datum<'tcx, Expr>,
                          method_call: MethodCall)
                          -> DatumBlock<'blk, 'tcx, Expr> {
    let ccx = bcx.ccx();

    debug!("deref_once(expr={}, datum={}, method_call={:?})",
           expr.repr(bcx.tcx()),
           datum.to_string(ccx),
           method_call);

    let mut bcx = bcx;

    // Check for overloaded deref.
    let method_ty = ccx.tcx().method_map.borrow()
                       .get(&method_call).map(|method| method.ty);
    let datum = match method_ty {
        Some(method_ty) => {
            let method_ty = monomorphize_type(bcx, method_ty);

            // Overloaded. Evaluate `trans_overloaded_op`, which will
            // invoke the user's deref() method, which basically
            // converts from the `Smaht<T>` pointer that we have into
            // a `&T` pointer.  We can then proceed down the normal
            // path (below) to dereference that `&T`.
            let datum = match method_call.adjustment {
                // Always perform an AutoPtr when applying an overloaded auto-deref
                ty::AutoDeref(_) => unpack_datum!(bcx, auto_ref(bcx, datum, expr)),
                _ => datum
            };

            let ref_ty = // invoked methods have their LB regions instantiated
                ty::no_late_bound_regions(
                    ccx.tcx(), &ty::ty_fn_ret(method_ty)).unwrap().unwrap();
            let scratch = rvalue_scratch_datum(bcx, ref_ty, "overloaded_deref");

            unpack_result!(bcx, trans_overloaded_op(bcx, expr, method_call,
                                                    datum, Vec::new(), Some(SaveIn(scratch.val)),
                                                    false));
            scratch.to_expr_datum()
        }
        None => {
            // Not overloaded. We already have a pointer we know how to deref.
            datum
        }
    };

    let r = match datum.ty.sty {
        ty::ty_uniq(content_ty) => {
            if type_is_sized(bcx.tcx(), content_ty) {
                deref_owned_pointer(bcx, expr, datum, content_ty)
            } else {
                // A fat pointer and an opened DST value have the same
                // representation just different types. Since there is no
                // temporary for `*e` here (because it is unsized), we cannot
                // emulate the sized object code path for running drop glue and
                // free. Instead, we schedule cleanup for `e`, turning it into
                // an lvalue.
                let datum = unpack_datum!(
                    bcx, datum.to_lvalue_datum(bcx, "deref", expr.id));

                let datum = Datum::new(datum.val, ty::mk_open(bcx.tcx(), content_ty), LvalueExpr);
                DatumBlock::new(bcx, datum)
            }
        }

        ty::ty_ptr(ty::mt { ty: content_ty, .. }) |
        ty::ty_rptr(_, ty::mt { ty: content_ty, .. }) => {
            if type_is_sized(bcx.tcx(), content_ty) {
                let ptr = datum.to_llscalarish(bcx);

                // Always generate an lvalue datum, even if datum.mode is
                // an rvalue.  This is because datum.mode is only an
                // rvalue for non-owning pointers like &T or *T, in which
                // case cleanup *is* scheduled elsewhere, by the true
                // owner (or, in the case of *T, by the user).
                DatumBlock::new(bcx, Datum::new(ptr, content_ty, LvalueExpr))
            } else {
                // A fat pointer and an opened DST value have the same representation
                // just different types.
                DatumBlock::new(bcx, Datum::new(datum.val,
                                                ty::mk_open(bcx.tcx(), content_ty),
                                                LvalueExpr))
            }
        }

        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                &format!("deref invoked on expr of illegal type {}",
                        datum.ty.repr(bcx.tcx()))[]);
        }
    };

    debug!("deref_once(expr={}, method_call={:?}, result={})",
           expr.id, method_call, r.datum.to_string(ccx));

    return r;

    /// We microoptimize derefs of owned pointers a bit here. Basically, the idea is to make the
    /// deref of an rvalue result in an rvalue. This helps to avoid intermediate stack slots in the
    /// resulting LLVM. The idea here is that, if the `Box<T>` pointer is an rvalue, then we can
    /// schedule a *shallow* free of the `Box<T>` pointer, and then return a ByRef rvalue into the
    /// pointer. Because the free is shallow, it is legit to return an rvalue, because we know that
    /// the contents are not yet scheduled to be freed. The language rules ensure that the contents
    /// will be used (or moved) before the free occurs.
    fn deref_owned_pointer<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                       expr: &ast::Expr,
                                       datum: Datum<'tcx, Expr>,
                                       content_ty: Ty<'tcx>)
                                       -> DatumBlock<'blk, 'tcx, Expr> {
        match datum.kind {
            RvalueExpr(Rvalue { mode: ByRef }) => {
                let scope = cleanup::temporary_scope(bcx.tcx(), expr.id);
                let ptr = Load(bcx, datum.val);
                if !type_is_zero_size(bcx.ccx(), content_ty) {
                    bcx.fcx.schedule_free_value(scope, ptr, cleanup::HeapExchange, content_ty);
                }
            }
            RvalueExpr(Rvalue { mode: ByValue }) => {
                let scope = cleanup::temporary_scope(bcx.tcx(), expr.id);
                if !type_is_zero_size(bcx.ccx(), content_ty) {
                    bcx.fcx.schedule_free_value(scope, datum.val, cleanup::HeapExchange,
                                                content_ty);
                }
            }
            LvalueExpr => { }
        }

        // If we had an rvalue in, we produce an rvalue out.
        let (llptr, kind) = match datum.kind {
            LvalueExpr => {
                (Load(bcx, datum.val), LvalueExpr)
            }
            RvalueExpr(Rvalue { mode: ByRef }) => {
                (Load(bcx, datum.val), RvalueExpr(Rvalue::new(ByRef)))
            }
            RvalueExpr(Rvalue { mode: ByValue }) => {
                (datum.val, RvalueExpr(Rvalue::new(ByRef)))
            }
        };

        let datum = Datum { ty: content_ty, val: llptr, kind: kind };
        DatumBlock { bcx: bcx, datum: datum }
    }
}
