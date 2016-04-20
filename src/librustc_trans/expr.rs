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
//! - `trans_var -> Datum`: looks up a local variable, upvar or static.

#![allow(non_camel_case_types)]

pub use self::Dest::*;
use self::lazy_binop_ty::*;

use llvm::{self, ValueRef, TypeKind};
use middle::const_qualif::ConstQualif;
use rustc::hir::def::Def;
use rustc::ty::subst::Substs;
use {_match, abi, adt, asm, base, closure, consts, controlflow};
use base::*;
use build::*;
use callee::{Callee, ArgExprs, ArgOverloadedCall, ArgOverloadedOp};
use cleanup::{self, CleanupMethods, DropHintMethods};
use common::*;
use datum::*;
use debuginfo::{self, DebugLoc, ToDebugLoc};
use declare;
use glue;
use machine;
use tvec;
use type_of;
use value::Value;
use Disr;
use rustc::ty::adjustment::{AdjustDerefRef, AdjustReifyFnPointer};
use rustc::ty::adjustment::{AdjustUnsafeFnPointer, AdjustMutToConstPointer};
use rustc::ty::adjustment::CustomCoerceUnsized;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::MethodCall;
use rustc::ty::cast::{CastKind, CastTy};
use util::common::indenter;
use machine::{llsize_of, llsize_of_alloc};
use type_::Type;

use rustc::hir;

use syntax::{ast, codemap};
use syntax::parse::token::InternedString;
use std::fmt;
use std::mem;

// Destinations

// These are passed around by the code generating functions to track the
// destination of a computation's value.

#[derive(Copy, Clone, PartialEq)]
pub enum Dest {
    SaveIn(ValueRef),
    Ignore,
}

impl fmt::Debug for Dest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SaveIn(v) => write!(f, "SaveIn({:?})", Value(v)),
            Ignore => f.write_str("Ignore")
        }
    }
}

/// This function is equivalent to `trans(bcx, expr).store_to_dest(dest)` but it may generate
/// better optimized LLVM code.
pub fn trans_into<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &hir::Expr,
                              dest: Dest)
                              -> Block<'blk, 'tcx> {
    let mut bcx = bcx;

    expr.debug_loc().apply(bcx.fcx);

    if adjustment_required(bcx, expr) {
        // use trans, which may be less efficient but
        // which will perform the adjustments:
        let datum = unpack_datum!(bcx, trans(bcx, expr));
        return datum.store_to_dest(bcx, dest, expr.id);
    }

    let qualif = *bcx.tcx().const_qualif_map.borrow().get(&expr.id).unwrap();
    if !qualif.intersects(ConstQualif::NOT_CONST | ConstQualif::NEEDS_DROP) {
        if !qualif.intersects(ConstQualif::PREFER_IN_PLACE) {
            if let SaveIn(lldest) = dest {
                match consts::get_const_expr_as_global(bcx.ccx(), expr, qualif,
                                                       bcx.fcx.param_substs,
                                                       consts::TrueConst::No) {
                    Ok(global) => {
                        // Cast pointer to destination, because constants
                        // have different types.
                        let lldest = PointerCast(bcx, lldest, val_ty(global));
                        memcpy_ty(bcx, lldest, global, expr_ty_adjusted(bcx, expr));
                        return bcx;
                    },
                    Err(consts::ConstEvalFailure::Runtime(_)) => {
                        // in case const evaluation errors, translate normally
                        // debug assertions catch the same errors
                        // see RFC 1229
                    },
                    Err(consts::ConstEvalFailure::Compiletime(_)) => {
                        return bcx;
                    },
                }
            }

            // If we see a const here, that's because it evaluates to a type with zero size. We
            // should be able to just discard it, since const expressions are guaranteed not to
            // have side effects. This seems to be reached through tuple struct constructors being
            // passed zero-size constants.
            if let hir::ExprPath(..) = expr.node {
                match bcx.def(expr.id) {
                    Def::Const(_) | Def::AssociatedConst(_) => {
                        assert!(type_is_zero_size(bcx.ccx(), bcx.tcx().node_id_to_type(expr.id)));
                        return bcx;
                    }
                    _ => {}
                }
            }

            // Even if we don't have a value to emit, and the expression
            // doesn't have any side-effects, we still have to translate the
            // body of any closures.
            // FIXME: Find a better way of handling this case.
        } else {
            // The only way we're going to see a `const` at this point is if
            // it prefers in-place instantiation, likely because it contains
            // `[x; N]` somewhere within.
            match expr.node {
                hir::ExprPath(..) => {
                    match bcx.def(expr.id) {
                        Def::Const(did) | Def::AssociatedConst(did) => {
                            let empty_substs = bcx.tcx().mk_substs(Substs::empty());
                            let const_expr = consts::get_const_expr(bcx.ccx(), did, expr,
                                                                    empty_substs);
                            // Temporarily get cleanup scopes out of the way,
                            // as they require sub-expressions to be contained
                            // inside the current AST scope.
                            // These should record no cleanups anyways, `const`
                            // can't have destructors.
                            let scopes = mem::replace(&mut *bcx.fcx.scopes.borrow_mut(),
                                                      vec![]);
                            // Lock emitted debug locations to the location of
                            // the constant reference expression.
                            debuginfo::with_source_location_override(bcx.fcx,
                                                                     expr.debug_loc(),
                                                                     || {
                                bcx = trans_into(bcx, const_expr, dest)
                            });
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

    debug!("trans_into() expr={:?}", expr);

    let cleanup_debug_loc = debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(),
                                                                          expr.id,
                                                                          expr.span,
                                                                          false);
    bcx.fcx.push_ast_cleanup_scope(cleanup_debug_loc);

    let kind = expr_kind(bcx.tcx(), expr);
    bcx = match kind {
        ExprKind::Lvalue | ExprKind::RvalueDatum => {
            trans_unadjusted(bcx, expr).store_to_dest(dest, expr.id)
        }
        ExprKind::RvalueDps => {
            trans_rvalue_dps_unadjusted(bcx, expr, dest)
        }
        ExprKind::RvalueStmt => {
            trans_rvalue_stmt_unadjusted(bcx, expr)
        }
    };

    bcx.fcx.pop_and_trans_ast_cleanup_scope(bcx, expr.id)
}

/// Translates an expression, returning a datum (and new block) encapsulating the result. When
/// possible, it is preferred to use `trans_into`, as that may avoid creating a temporary on the
/// stack.
pub fn trans<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                         expr: &hir::Expr)
                         -> DatumBlock<'blk, 'tcx, Expr> {
    debug!("trans(expr={:?})", expr);

    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let qualif = *bcx.tcx().const_qualif_map.borrow().get(&expr.id).unwrap();
    let adjusted_global = !qualif.intersects(ConstQualif::NON_STATIC_BORROWS);
    let global = if !qualif.intersects(ConstQualif::NOT_CONST | ConstQualif::NEEDS_DROP) {
        match consts::get_const_expr_as_global(bcx.ccx(), expr, qualif,
                                                            bcx.fcx.param_substs,
                                                            consts::TrueConst::No) {
            Ok(global) => {
                if qualif.intersects(ConstQualif::HAS_STATIC_BORROWS) {
                    // Is borrowed as 'static, must return lvalue.

                    // Cast pointer to global, because constants have different types.
                    let const_ty = expr_ty_adjusted(bcx, expr);
                    let llty = type_of::type_of(bcx.ccx(), const_ty);
                    let global = PointerCast(bcx, global, llty.ptr_to());
                    let datum = Datum::new(global, const_ty, Lvalue::new("expr::trans"));
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
                    let scratch = alloc_ty(bcx, const_ty, "const");
                    call_lifetime_start(bcx, scratch);
                    let lldest = if !const_ty.is_structural() {
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
            },
            Err(consts::ConstEvalFailure::Runtime(_)) => {
                // in case const evaluation errors, translate normally
                // debug assertions catch the same errors
                // see RFC 1229
                None
            },
            Err(consts::ConstEvalFailure::Compiletime(_)) => {
                // generate a dummy llvm value
                let const_ty = expr_ty(bcx, expr);
                let llty = type_of::type_of(bcx.ccx(), const_ty);
                let dummy = C_undef(llty.ptr_to());
                Some(Datum::new(dummy, const_ty, Rvalue::new(ByRef)))
            },
        }
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

pub fn get_meta(bcx: Block, fat_ptr: ValueRef) -> ValueRef {
    StructGEP(bcx, fat_ptr, abi::FAT_PTR_EXTRA)
}

pub fn get_dataptr(bcx: Block, fat_ptr: ValueRef) -> ValueRef {
    StructGEP(bcx, fat_ptr, abi::FAT_PTR_ADDR)
}

pub fn copy_fat_ptr(bcx: Block, src_ptr: ValueRef, dst_ptr: ValueRef) {
    Store(bcx, Load(bcx, get_dataptr(bcx, src_ptr)), get_dataptr(bcx, dst_ptr));
    Store(bcx, Load(bcx, get_meta(bcx, src_ptr)), get_meta(bcx, dst_ptr));
}

fn adjustment_required<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   expr: &hir::Expr) -> bool {
    let adjustment = match bcx.tcx().tables.borrow().adjustments.get(&expr.id).cloned() {
        None => { return false; }
        Some(adj) => adj
    };

    // Don't skip a conversion from Box<T> to &T, etc.
    if bcx.tcx().is_overloaded_autoderef(expr.id, 0) {
        return true;
    }

    match adjustment {
        AdjustReifyFnPointer => true,
        AdjustUnsafeFnPointer | AdjustMutToConstPointer => {
            // purely a type-level thing
            false
        }
        AdjustDerefRef(ref adj) => {
            // We are a bit paranoid about adjustments and thus might have a re-
            // borrow here which merely derefs and then refs again (it might have
            // a different region or mutability, but we don't care here).
            !(adj.autoderefs == 1 && adj.autoref.is_some() && adj.unsize.is_none())
        }
    }
}

/// Helper for trans that apply adjustments from `expr` to `datum`, which should be the unadjusted
/// translation of `expr`.
fn apply_adjustments<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 expr: &hir::Expr,
                                 datum: Datum<'tcx, Expr>)
                                 -> DatumBlock<'blk, 'tcx, Expr>
{
    let mut bcx = bcx;
    let mut datum = datum;
    let adjustment = match bcx.tcx().tables.borrow().adjustments.get(&expr.id).cloned() {
        None => {
            return DatumBlock::new(bcx, datum);
        }
        Some(adj) => { adj }
    };
    debug!("unadjusted datum for expr {:?}: {:?} adjustment={:?}",
           expr, datum, adjustment);
    match adjustment {
        AdjustReifyFnPointer => {
            match datum.ty.sty {
                ty::TyFnDef(def_id, substs, _) => {
                    datum = Callee::def(bcx.ccx(), def_id, substs)
                        .reify(bcx.ccx()).to_expr_datum();
                }
                _ => {
                    bug!("{} cannot be reified to a fn ptr", datum.ty)
                }
            }
        }
        AdjustUnsafeFnPointer | AdjustMutToConstPointer => {
            // purely a type-level thing
        }
        AdjustDerefRef(ref adj) => {
            let skip_reborrows = if adj.autoderefs == 1 && adj.autoref.is_some() {
                // We are a bit paranoid about adjustments and thus might have a re-
                // borrow here which merely derefs and then refs again (it might have
                // a different region or mutability, but we don't care here).
                match datum.ty.sty {
                    // Don't skip a conversion from Box<T> to &T, etc.
                    ty::TyRef(..) => {
                        if bcx.tcx().is_overloaded_autoderef(expr.id, 0) {
                            // Don't skip an overloaded deref.
                            0
                        } else {
                            1
                        }
                    }
                    _ => 0
                }
            } else {
                0
            };

            if adj.autoderefs > skip_reborrows {
                // Schedule cleanup.
                let lval = unpack_datum!(bcx, datum.to_lvalue_datum(bcx, "auto_deref", expr.id));
                datum = unpack_datum!(bcx, deref_multiple(bcx, expr,
                                                          lval.to_expr_datum(),
                                                          adj.autoderefs - skip_reborrows));
            }

            // (You might think there is a more elegant way to do this than a
            // skip_reborrows bool, but then you remember that the borrow checker exists).
            if skip_reborrows == 0 && adj.autoref.is_some() {
                datum = unpack_datum!(bcx, auto_ref(bcx, datum, expr));
            }

            if let Some(target) = adj.unsize {
                // We do not arrange cleanup ourselves; if we already are an
                // L-value, then cleanup will have already been scheduled (and
                // the `datum.to_rvalue_datum` call below will emit code to zero
                // the drop flag when moving out of the L-value). If we are an
                // R-value, then we do not need to schedule cleanup.
                let source_datum = unpack_datum!(bcx,
                    datum.to_rvalue_datum(bcx, "__coerce_source"));

                let target = bcx.monomorphize(&target);

                let scratch = alloc_ty(bcx, target, "__coerce_target");
                call_lifetime_start(bcx, scratch);
                let target_datum = Datum::new(scratch, target,
                                              Rvalue::new(ByRef));
                bcx = coerce_unsized(bcx, expr.span, source_datum, target_datum);
                datum = Datum::new(scratch, target,
                                   RvalueExpr(Rvalue::new(ByRef)));
            }
        }
    }
    debug!("after adjustments, datum={:?}", datum);
    DatumBlock::new(bcx, datum)
}

fn coerce_unsized<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              span: codemap::Span,
                              source: Datum<'tcx, Rvalue>,
                              target: Datum<'tcx, Rvalue>)
                              -> Block<'blk, 'tcx> {
    let mut bcx = bcx;
    debug!("coerce_unsized({:?} -> {:?})", source, target);

    match (&source.ty.sty, &target.ty.sty) {
        (&ty::TyBox(a), &ty::TyBox(b)) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRef(_, ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRef(_, ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) |
        (&ty::TyRawPtr(ty::TypeAndMut { ty: a, .. }),
         &ty::TyRawPtr(ty::TypeAndMut { ty: b, .. })) => {
            let (inner_source, inner_target) = (a, b);

            let (base, old_info) = if !type_is_sized(bcx.tcx(), inner_source) {
                // Normally, the source is a thin pointer and we are
                // adding extra info to make a fat pointer. The exception
                // is when we are upcasting an existing object fat pointer
                // to use a different vtable. In that case, we want to
                // load out the original data pointer so we can repackage
                // it.
                (Load(bcx, get_dataptr(bcx, source.val)),
                Some(Load(bcx, get_meta(bcx, source.val))))
            } else {
                let val = if source.kind.is_by_ref() {
                    load_ty(bcx, source.val, source.ty)
                } else {
                    source.val
                };
                (val, None)
            };

            let info = unsized_info(bcx.ccx(), inner_source, inner_target, old_info);

            // Compute the base pointer. This doesn't change the pointer value,
            // but merely its type.
            let ptr_ty = type_of::in_memory_type_of(bcx.ccx(), inner_target).ptr_to();
            let base = PointerCast(bcx, base, ptr_ty);

            Store(bcx, base, get_dataptr(bcx, target.val));
            Store(bcx, info, get_meta(bcx, target.val));
        }

        // This can be extended to enums and tuples in the future.
        // (&ty::TyEnum(def_id_a, _), &ty::TyEnum(def_id_b, _)) |
        (&ty::TyStruct(def_id_a, _), &ty::TyStruct(def_id_b, _)) => {
            assert_eq!(def_id_a, def_id_b);

            // The target is already by-ref because it's to be written to.
            let source = unpack_datum!(bcx, source.to_ref_datum(bcx));
            assert!(target.kind.is_by_ref());

            let kind = custom_coerce_unsize_info(bcx.ccx(), source.ty, target.ty);

            let repr_source = adt::represent_type(bcx.ccx(), source.ty);
            let src_fields = match &*repr_source {
                &adt::Repr::Univariant(ref s, _) => &s.fields,
                _ => span_bug!(span,
                               "Non univariant struct? (repr_source: {:?})",
                               repr_source),
            };
            let repr_target = adt::represent_type(bcx.ccx(), target.ty);
            let target_fields = match &*repr_target {
                &adt::Repr::Univariant(ref s, _) => &s.fields,
                _ => span_bug!(span,
                               "Non univariant struct? (repr_target: {:?})",
                               repr_target),
            };

            let coerce_index = match kind {
                CustomCoerceUnsized::Struct(i) => i
            };
            assert!(coerce_index < src_fields.len() && src_fields.len() == target_fields.len());

            let source_val = adt::MaybeSizedValue::sized(source.val);
            let target_val = adt::MaybeSizedValue::sized(target.val);

            let iter = src_fields.iter().zip(target_fields).enumerate();
            for (i, (src_ty, target_ty)) in iter {
                let ll_source = adt::trans_field_ptr(bcx, &repr_source, source_val, Disr(0), i);
                let ll_target = adt::trans_field_ptr(bcx, &repr_target, target_val, Disr(0), i);

                // If this is the field we need to coerce, recurse on it.
                if i == coerce_index {
                    coerce_unsized(bcx, span,
                                   Datum::new(ll_source, src_ty,
                                              Rvalue::new(ByRef)),
                                   Datum::new(ll_target, target_ty,
                                              Rvalue::new(ByRef)));
                } else {
                    // Otherwise, simply copy the data from the source.
                    assert!(src_ty.is_phantom_data() || src_ty == target_ty);
                    memcpy_ty(bcx, ll_target, ll_source, src_ty);
                }
            }
        }
        _ => bug!("coerce_unsized: invalid coercion {:?} -> {:?}",
                  source.ty,
                  target.ty)
    }
    bcx
}

/// Translates an expression in "lvalue" mode -- meaning that it returns a reference to the memory
/// that the expr represents.
///
/// If this expression is an rvalue, this implies introducing a temporary.  In other words,
/// something like `x().f` is translated into roughly the equivalent of
///
///   { tmp = x(); tmp.f }
pub fn trans_to_lvalue<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   expr: &hir::Expr,
                                   name: &str)
                                   -> DatumBlock<'blk, 'tcx, Lvalue> {
    let mut bcx = bcx;
    let datum = unpack_datum!(bcx, trans(bcx, expr));
    return datum.to_lvalue_datum(bcx, name, expr.id);
}

/// A version of `trans` that ignores adjustments. You almost certainly do not want to call this
/// directly.
fn trans_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                expr: &hir::Expr)
                                -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;

    debug!("trans_unadjusted(expr={:?})", expr);
    let _indenter = indenter();

    expr.debug_loc().apply(bcx.fcx);

    return match expr_kind(bcx.tcx(), expr) {
        ExprKind::Lvalue | ExprKind::RvalueDatum => {
            let datum = unpack_datum!(bcx, {
                trans_datum_unadjusted(bcx, expr)
            });

            DatumBlock {bcx: bcx, datum: datum}
        }

        ExprKind::RvalueStmt => {
            bcx = trans_rvalue_stmt_unadjusted(bcx, expr);
            nil(bcx, expr_ty(bcx, expr))
        }

        ExprKind::RvalueDps => {
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
                                      expr: &hir::Expr)
                                      -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let _icx = push_ctxt("trans_datum_unadjusted");

    match expr.node {
        hir::ExprType(ref e, _) => {
            trans(bcx, &e)
        }
        hir::ExprPath(..) => {
            let var = trans_var(bcx, bcx.def(expr.id));
            DatumBlock::new(bcx, var.to_expr_datum())
        }
        hir::ExprField(ref base, name) => {
            trans_rec_field(bcx, &base, name.node)
        }
        hir::ExprTupField(ref base, idx) => {
            trans_rec_tup_field(bcx, &base, idx.node)
        }
        hir::ExprIndex(ref base, ref idx) => {
            trans_index(bcx, expr, &base, &idx, MethodCall::expr(expr.id))
        }
        hir::ExprBox(ref contents) => {
            // Special case for `Box<T>`
            let box_ty = expr_ty(bcx, expr);
            let contents_ty = expr_ty(bcx, &contents);
            match box_ty.sty {
                ty::TyBox(..) => {
                    trans_uniq_expr(bcx, expr, box_ty, &contents, contents_ty)
                }
                _ => span_bug!(expr.span,
                               "expected unique box")
            }

        }
        hir::ExprLit(ref lit) => trans_immediate_lit(bcx, expr, &lit),
        hir::ExprBinary(op, ref lhs, ref rhs) => {
            trans_binary(bcx, expr, op, &lhs, &rhs)
        }
        hir::ExprUnary(op, ref x) => {
            trans_unary(bcx, expr, op, &x)
        }
        hir::ExprAddrOf(_, ref x) => {
            match x.node {
                hir::ExprRepeat(..) | hir::ExprVec(..) => {
                    // Special case for slices.
                    let cleanup_debug_loc =
                        debuginfo::get_cleanup_debug_loc_for_ast_node(bcx.ccx(),
                                                                      x.id,
                                                                      x.span,
                                                                      false);
                    fcx.push_ast_cleanup_scope(cleanup_debug_loc);
                    let datum = unpack_datum!(
                        bcx, tvec::trans_slice_vec(bcx, expr, &x));
                    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, x.id);
                    DatumBlock::new(bcx, datum)
                }
                _ => {
                    trans_addr_of(bcx, expr, &x)
                }
            }
        }
        hir::ExprCast(ref val, _) => {
            // Datum output mode means this is a scalar cast:
            trans_imm_cast(bcx, &val, expr.id)
        }
        _ => {
            span_bug!(
                expr.span,
                "trans_rvalue_datum_unadjusted reached \
                 fall-through case: {:?}",
                expr.node);
        }
    }
}

fn trans_field<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                              base: &hir::Expr,
                              get_idx: F)
                              -> DatumBlock<'blk, 'tcx, Expr> where
    F: FnOnce(&'blk TyCtxt<'tcx>, &VariantInfo<'tcx>) -> usize,
{
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rec_field");

    let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, base, "field"));
    let bare_ty = base_datum.ty;
    let repr = adt::represent_type(bcx.ccx(), bare_ty);
    let vinfo = VariantInfo::from_ty(bcx.tcx(), bare_ty, None);

    let ix = get_idx(bcx.tcx(), &vinfo);
    let d = base_datum.get_element(
        bcx,
        vinfo.fields[ix].1,
        |srcval| {
            adt::trans_field_ptr(bcx, &repr, srcval, vinfo.discr, ix)
        });

    if type_is_sized(bcx.tcx(), d.ty) {
        DatumBlock { datum: d.to_expr_datum(), bcx: bcx }
    } else {
        let scratch = rvalue_scratch_datum(bcx, d.ty, "");
        Store(bcx, d.val, get_dataptr(bcx, scratch.val));
        let info = Load(bcx, get_meta(bcx, base_datum.val));
        Store(bcx, info, get_meta(bcx, scratch.val));

        // Always generate an lvalue datum, because this pointer doesn't own
        // the data and cleanup is scheduled elsewhere.
        DatumBlock::new(bcx, Datum::new(scratch.val, scratch.ty, LvalueExpr(d.kind)))
    }
}

/// Translates `base.field`.
fn trans_rec_field<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               base: &hir::Expr,
                               field: ast::Name)
                               -> DatumBlock<'blk, 'tcx, Expr> {
    trans_field(bcx, base, |_, vinfo| vinfo.field_index(field))
}

/// Translates `base.<idx>`.
fn trans_rec_tup_field<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   base: &hir::Expr,
                                   idx: usize)
                                   -> DatumBlock<'blk, 'tcx, Expr> {
    trans_field(bcx, base, |_, _| idx)
}

fn trans_index<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           index_expr: &hir::Expr,
                           base: &hir::Expr,
                           idx: &hir::Expr,
                           method_call: MethodCall)
                           -> DatumBlock<'blk, 'tcx, Expr> {
    //! Translates `base[idx]`.

    let _icx = push_ctxt("trans_index");
    let ccx = bcx.ccx();
    let mut bcx = bcx;

    let index_expr_debug_loc = index_expr.debug_loc();

    // Check for overloaded index.
    let method = ccx.tcx().tables.borrow().method_map.get(&method_call).cloned();
    let elt_datum = match method {
        Some(method) => {
            let method_ty = monomorphize_type(bcx, method.ty);

            let base_datum = unpack_datum!(bcx, trans(bcx, base));

            // Translate index expression.
            let ix_datum = unpack_datum!(bcx, trans(bcx, idx));

            let ref_ty = // invoked methods have LB regions instantiated:
                bcx.tcx().no_late_bound_regions(&method_ty.fn_ret()).unwrap().unwrap();
            let elt_ty = match ref_ty.builtin_deref(true, ty::NoPreference) {
                None => {
                    span_bug!(index_expr.span,
                              "index method didn't return a \
                              dereferenceable type?!")
                }
                Some(elt_tm) => elt_tm.ty,
            };

            // Overloaded. Invoke the index() method, which basically
            // yields a `&T` pointer.  We can then proceed down the
            // normal path (below) to dereference that `&T`.
            let scratch = rvalue_scratch_datum(bcx, ref_ty, "overloaded_index_elt");

            bcx = Callee::method(bcx, method)
                .call(bcx, index_expr_debug_loc,
                      ArgOverloadedOp(base_datum, Some(ix_datum)),
                      Some(SaveIn(scratch.val))).bcx;

            let datum = scratch.to_expr_datum();
            let lval = Lvalue::new("expr::trans_index overload");
            if type_is_sized(bcx.tcx(), elt_ty) {
                Datum::new(datum.to_llscalarish(bcx), elt_ty, LvalueExpr(lval))
            } else {
                Datum::new(datum.val, elt_ty, LvalueExpr(lval))
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
                    if expr_ty(bcx, idx).is_signed() {
                        SExt(bcx, ix_val, ccx.int_type())
                    } else { ZExt(bcx, ix_val, ccx.int_type()) }
                } else if ix_size > int_size {
                    Trunc(bcx, ix_val, ccx.int_type())
                } else {
                    ix_val
                }
            };

            let unit_ty = base_datum.ty.sequence_element_type(bcx.tcx());

            let (base, len) = base_datum.get_vec_base_and_len(bcx);

            debug!("trans_index: base {:?}", Value(base));
            debug!("trans_index: len {:?}", Value(len));

            let bounds_check = ICmp(bcx,
                                    llvm::IntUGE,
                                    ix_val,
                                    len,
                                    index_expr_debug_loc);
            let expect = ccx.get_intrinsic(&("llvm.expect.i1"));
            let expected = Call(bcx,
                                expect,
                                &[bounds_check, C_bool(ccx, false)],
                                index_expr_debug_loc);
            bcx = with_cond(bcx, expected, |bcx| {
                controlflow::trans_fail_bounds_check(bcx,
                                                     expr_info(index_expr),
                                                     ix_val,
                                                     len)
            });
            let elt = InBoundsGEP(bcx, base, &[ix_val]);
            let elt = PointerCast(bcx, elt, type_of::type_of(ccx, unit_ty).ptr_to());
            let lval = Lvalue::new("expr::trans_index fallback");
            Datum::new(elt, unit_ty, LvalueExpr(lval))
        }
    };

    DatumBlock::new(bcx, elt_datum)
}

/// Translates a reference to a variable.
pub fn trans_var<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, def: Def)
                             -> Datum<'tcx, Lvalue> {

    match def {
        Def::Static(did, _) => consts::get_static(bcx.ccx(), did),
        Def::Upvar(_, nid, _, _) => {
            // Can't move upvars, so this is never a ZeroMemLastUse.
            let local_ty = node_id_type(bcx, nid);
            let lval = Lvalue::new_with_hint("expr::trans_var (upvar)",
                                             bcx, nid, HintKind::ZeroAndMaintain);
            match bcx.fcx.llupvars.borrow().get(&nid) {
                Some(&val) => Datum::new(val, local_ty, lval),
                None => {
                    bug!("trans_var: no llval for upvar {} found", nid);
                }
            }
        }
        Def::Local(_, nid) => {
            let datum = match bcx.fcx.lllocals.borrow().get(&nid) {
                Some(&v) => v,
                None => {
                    bug!("trans_var: no datum for local/arg {} found", nid);
                }
            };
            debug!("take_local(nid={}, v={:?}, ty={})",
                   nid, Value(datum.val), datum.ty);
            datum
        }
        _ => bug!("{:?} should not reach expr::trans_var", def)
    }
}

fn trans_rvalue_stmt_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                            expr: &hir::Expr)
                                            -> Block<'blk, 'tcx> {
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rvalue_stmt");

    if bcx.unreachable.get() {
        return bcx;
    }

    expr.debug_loc().apply(bcx.fcx);

    match expr.node {
        hir::ExprBreak(label_opt) => {
            controlflow::trans_break(bcx, expr, label_opt.map(|l| l.node.name))
        }
        hir::ExprType(ref e, _) => {
            trans_into(bcx, &e, Ignore)
        }
        hir::ExprAgain(label_opt) => {
            controlflow::trans_cont(bcx, expr, label_opt.map(|l| l.node.name))
        }
        hir::ExprRet(ref ex) => {
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
                    bcx = trans_into(bcx, &x, Ignore);
                }
                // Mark the end of the block as unreachable. Once we get to
                // a return expression, there's no more we should be doing
                // after this.
                Unreachable(bcx);
                bcx
            }
        }
        hir::ExprWhile(ref cond, ref body, _) => {
            controlflow::trans_while(bcx, expr, &cond, &body)
        }
        hir::ExprLoop(ref body, _) => {
            controlflow::trans_loop(bcx, expr, &body)
        }
        hir::ExprAssign(ref dst, ref src) => {
            let src_datum = unpack_datum!(bcx, trans(bcx, &src));
            let dst_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, &dst, "assign"));

            if bcx.fcx.type_needs_drop(dst_datum.ty) {
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
                expr.debug_loc().apply(bcx.fcx);
                let src_datum = unpack_datum!(
                    bcx, src_datum.to_rvalue_datum(bcx, "ExprAssign"));
                let opt_hint_datum = dst_datum.kind.drop_flag_info.hint_datum(bcx);
                let opt_hint_val = opt_hint_datum.map(|d|d.to_value());

                // 1. Drop the data at the destination, passing the
                //    drop-hint in case the lvalue has already been
                //    dropped or moved.
                bcx = glue::drop_ty_core(bcx,
                                         dst_datum.val,
                                         dst_datum.ty,
                                         expr.debug_loc(),
                                         false,
                                         opt_hint_val);

                // 2. We are overwriting the destination; ensure that
                //    its drop-hint (if any) says "initialized."
                if let Some(hint_val) = opt_hint_val {
                    let hint_llval = hint_val.value();
                    let drop_needed = C_u8(bcx.fcx.ccx, adt::DTOR_NEEDED_HINT);
                    Store(bcx, drop_needed, hint_llval);
                }
                src_datum.store_to(bcx, dst_datum.val)
            } else {
                src_datum.store_to(bcx, dst_datum.val)
            }
        }
        hir::ExprAssignOp(op, ref dst, ref src) => {
            let method = bcx.tcx().tables
                                  .borrow()
                                  .method_map
                                  .get(&MethodCall::expr(expr.id)).cloned();

            if let Some(method) = method {
                let dst = unpack_datum!(bcx, trans(bcx, &dst));
                let src_datum = unpack_datum!(bcx, trans(bcx, &src));

                Callee::method(bcx, method)
                    .call(bcx, expr.debug_loc(),
                          ArgOverloadedOp(dst, Some(src_datum)), None).bcx
            } else {
                trans_assign_op(bcx, expr, op, &dst, &src)
            }
        }
        hir::ExprInlineAsm(ref a, ref outputs, ref inputs) => {
            let outputs = outputs.iter().map(|output| {
                let out_datum = unpack_datum!(bcx, trans(bcx, output));
                unpack_datum!(bcx, out_datum.to_lvalue_datum(bcx, "out", expr.id))
            }).collect();
            let inputs = inputs.iter().map(|input| {
                let input = unpack_datum!(bcx, trans(bcx, input));
                let input = unpack_datum!(bcx, input.to_rvalue_datum(bcx, "in"));
                input.to_llscalarish(bcx)
            }).collect();
            asm::trans_inline_asm(bcx, a, outputs, inputs);
            bcx
        }
        _ => {
            span_bug!(
                expr.span,
                "trans_rvalue_stmt_unadjusted reached \
                 fall-through case: {:?}",
                expr.node);
        }
    }
}

fn trans_rvalue_dps_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                           expr: &hir::Expr,
                                           dest: Dest)
                                           -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_rvalue_dps_unadjusted");
    let mut bcx = bcx;

    expr.debug_loc().apply(bcx.fcx);

    // Entry into the method table if this is an overloaded call/op.
    let method_call = MethodCall::expr(expr.id);

    match expr.node {
        hir::ExprType(ref e, _) => {
            trans_into(bcx, &e, dest)
        }
        hir::ExprPath(..) => {
            trans_def_dps_unadjusted(bcx, expr, bcx.def(expr.id), dest)
        }
        hir::ExprIf(ref cond, ref thn, ref els) => {
            controlflow::trans_if(bcx, expr.id, &cond, &thn, els.as_ref().map(|e| &**e), dest)
        }
        hir::ExprMatch(ref discr, ref arms, _) => {
            _match::trans_match(bcx, expr, &discr, &arms[..], dest)
        }
        hir::ExprBlock(ref blk) => {
            controlflow::trans_block(bcx, &blk, dest)
        }
        hir::ExprStruct(_, ref fields, ref base) => {
            trans_struct(bcx,
                         &fields[..],
                         base.as_ref().map(|e| &**e),
                         expr.span,
                         expr.id,
                         node_id_type(bcx, expr.id),
                         dest)
        }
        hir::ExprTup(ref args) => {
            let numbered_fields: Vec<(usize, &hir::Expr)> =
                args.iter().enumerate().map(|(i, arg)| (i, &**arg)).collect();
            trans_adt(bcx,
                      expr_ty(bcx, expr),
                      Disr(0),
                      &numbered_fields[..],
                      None,
                      dest,
                      expr.debug_loc())
        }
        hir::ExprLit(ref lit) => {
            match lit.node {
                ast::LitKind::Str(ref s, _) => {
                    tvec::trans_lit_str(bcx, expr, (*s).clone(), dest)
                }
                _ => {
                    span_bug!(expr.span,
                              "trans_rvalue_dps_unadjusted shouldn't be \
                              translating this type of literal")
                }
            }
        }
        hir::ExprVec(..) | hir::ExprRepeat(..) => {
            tvec::trans_fixed_vstore(bcx, expr, dest)
        }
        hir::ExprClosure(_, ref decl, ref body, _) => {
            let dest = match dest {
                SaveIn(lldest) => closure::Dest::SaveIn(bcx, lldest),
                Ignore => closure::Dest::Ignore(bcx.ccx())
            };

            // NB. To get the id of the closure, we don't use
            // `local_def_id(id)`, but rather we extract the closure
            // def-id from the expr's type. This is because this may
            // be an inlined expression from another crate, and we
            // want to get the ORIGINAL closure def-id, since that is
            // the key we need to find the closure-kind and
            // closure-type etc.
            let (def_id, substs) = match expr_ty(bcx, expr).sty {
                ty::TyClosure(def_id, ref substs) => (def_id, substs),
                ref t =>
                    span_bug!(
                        expr.span,
                        "closure expr without closure type: {:?}", t),
            };

            closure::trans_closure_expr(dest,
                                        decl,
                                        body,
                                        expr.id,
                                        def_id,
                                        substs).unwrap_or(bcx)
        }
        hir::ExprCall(ref f, ref args) => {
            let method = bcx.tcx().tables.borrow().method_map.get(&method_call).cloned();
            let (callee, args) = if let Some(method) = method {
                let mut all_args = vec![&**f];
                all_args.extend(args.iter().map(|e| &**e));

                (Callee::method(bcx, method), ArgOverloadedCall(all_args))
            } else {
                let f = unpack_datum!(bcx, trans(bcx, f));
                (match f.ty.sty {
                    ty::TyFnDef(def_id, substs, _) => {
                        Callee::def(bcx.ccx(), def_id, substs)
                    }
                    ty::TyFnPtr(_) => {
                        let f = unpack_datum!(bcx,
                            f.to_rvalue_datum(bcx, "callee"));
                        Callee::ptr(f)
                    }
                    _ => {
                        span_bug!(expr.span,
                            "type of callee is not a fn: {}", f.ty);
                    }
                }, ArgExprs(&args))
            };
            callee.call(bcx, expr.debug_loc(), args, Some(dest)).bcx
        }
        hir::ExprMethodCall(_, _, ref args) => {
            Callee::method_call(bcx, method_call)
                .call(bcx, expr.debug_loc(), ArgExprs(&args), Some(dest)).bcx
        }
        hir::ExprBinary(op, ref lhs, ref rhs_expr) => {
            // if not overloaded, would be RvalueDatumExpr
            let lhs = unpack_datum!(bcx, trans(bcx, &lhs));
            let mut rhs = unpack_datum!(bcx, trans(bcx, &rhs_expr));
            if !op.node.is_by_value() {
                rhs = unpack_datum!(bcx, auto_ref(bcx, rhs, rhs_expr));
            }

            Callee::method_call(bcx, method_call)
                .call(bcx, expr.debug_loc(),
                      ArgOverloadedOp(lhs, Some(rhs)), Some(dest)).bcx
        }
        hir::ExprUnary(_, ref subexpr) => {
            // if not overloaded, would be RvalueDatumExpr
            let arg = unpack_datum!(bcx, trans(bcx, &subexpr));

            Callee::method_call(bcx, method_call)
                .call(bcx, expr.debug_loc(),
                      ArgOverloadedOp(arg, None), Some(dest)).bcx
        }
        hir::ExprCast(..) => {
            // Trait casts used to come this way, now they should be coercions.
            span_bug!(expr.span, "DPS expr_cast (residual trait cast?)")
        }
        hir::ExprAssignOp(op, _, _) => {
            span_bug!(
                expr.span,
                "augmented assignment `{}=` should always be a rvalue_stmt",
                op.node.as_str())
        }
        _ => {
            span_bug!(
                expr.span,
                "trans_rvalue_dps_unadjusted reached fall-through \
                 case: {:?}",
                expr.node);
        }
    }
}

fn trans_def_dps_unadjusted<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        ref_expr: &hir::Expr,
                                        def: Def,
                                        dest: Dest)
                                        -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_def_dps_unadjusted");

    let lldest = match dest {
        SaveIn(lldest) => lldest,
        Ignore => { return bcx; }
    };

    let ty = expr_ty(bcx, ref_expr);
    if let ty::TyFnDef(..) = ty.sty {
        // Zero-sized function or ctor.
        return bcx;
    }

    match def {
        Def::Variant(tid, vid) => {
            let variant = bcx.tcx().lookup_adt_def(tid).variant_with_id(vid);
            // Nullary variant.
            let ty = expr_ty(bcx, ref_expr);
            let repr = adt::represent_type(bcx.ccx(), ty);
            adt::trans_set_discr(bcx, &repr, lldest, Disr::from(variant.disr_val));
            bcx
        }
        Def::Struct(..) => {
            match ty.sty {
                ty::TyStruct(def, _) if def.has_dtor() => {
                    let repr = adt::represent_type(bcx.ccx(), ty);
                    adt::trans_set_discr(bcx, &repr, lldest, Disr(0));
                }
                _ => {}
            }
            bcx
        }
        _ => {
            span_bug!(ref_expr.span,
                      "Non-DPS def {:?} referened by {}",
                      def, bcx.node_id_to_string(ref_expr.id));
        }
    }
}

fn trans_struct<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                            fields: &[hir::Field],
                            base: Option<&hir::Expr>,
                            expr_span: codemap::Span,
                            expr_id: ast::NodeId,
                            ty: Ty<'tcx>,
                            dest: Dest) -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_rec");

    let tcx = bcx.tcx();
    let vinfo = VariantInfo::of_node(tcx, ty, expr_id);

    let mut need_base = vec![true; vinfo.fields.len()];

    let numbered_fields = fields.iter().map(|field| {
        let pos = vinfo.field_index(field.name.node);
        need_base[pos] = false;
        (pos, &*field.expr)
    }).collect::<Vec<_>>();

    let optbase = match base {
        Some(base_expr) => {
            let mut leftovers = Vec::new();
            for (i, b) in need_base.iter().enumerate() {
                if *b {
                    leftovers.push((i, vinfo.fields[i].1));
                }
            }
            Some(StructBaseInfo {expr: base_expr,
                                 fields: leftovers })
        }
        None => {
            if need_base.iter().any(|b| *b) {
                span_bug!(expr_span, "missing fields and no base expr")
            }
            None
        }
    };

    trans_adt(bcx,
              ty,
              vinfo.discr,
              &numbered_fields,
              optbase,
              dest,
              DebugLoc::At(expr_id, expr_span))
}

/// Information that `trans_adt` needs in order to fill in the fields
/// of a struct copied from a base struct (e.g., from an expression
/// like `Foo { a: b, ..base }`.
///
/// Note that `fields` may be empty; the base expression must always be
/// evaluated for side-effects.
pub struct StructBaseInfo<'a, 'tcx> {
    /// The base expression; will be evaluated after all explicit fields.
    expr: &'a hir::Expr,
    /// The indices of fields to copy paired with their types.
    fields: Vec<(usize, Ty<'tcx>)>
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
                                 discr: Disr,
                                 fields: &[(usize, &hir::Expr)],
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
        Ignore => {
            let llresult = alloc_ty(bcx, ty, "temp");
            call_lifetime_start(bcx, llresult);
            llresult
        }
    };

    debug!("trans_adt");

    // This scope holds intermediates that must be cleaned should
    // panic occur before the ADT as a whole is ready.
    let custom_cleanup_scope = fcx.push_custom_cleanup_scope();

    if ty.is_simd() {
        // Issue 23112: The original logic appeared vulnerable to same
        // order-of-eval bug. But, SIMD values are tuple-structs;
        // i.e. functional record update (FRU) syntax is unavailable.
        //
        // To be safe, double-check that we did not get here via FRU.
        assert!(optbase.is_none());

        // This is the constructor of a SIMD type, such types are
        // always primitive machine types and so do not have a
        // destructor or require any clean-up.
        let llty = type_of::type_of(bcx.ccx(), ty);

        // keep a vector as a register, and running through the field
        // `insertelement`ing them directly into that register
        // (i.e. avoid GEPi and `store`s to an alloca) .
        let mut vec_val = C_undef(llty);

        for &(i, ref e) in fields {
            let block_datum = trans(bcx, &e);
            bcx = block_datum.bcx;
            let position = C_uint(bcx.ccx(), i);
            let value = block_datum.datum.to_llscalarish(bcx);
            vec_val = InsertElement(bcx, vec_val, value, position);
        }
        Store(bcx, vec_val, addr);
    } else if let Some(base) = optbase {
        // Issue 23112: If there is a base, then order-of-eval
        // requires field expressions eval'ed before base expression.

        // First, trans field expressions to temporary scratch values.
        let scratch_vals: Vec<_> = fields.iter().map(|&(i, ref e)| {
            let datum = unpack_datum!(bcx, trans(bcx, &e));
            (i, datum)
        }).collect();

        debug_location.apply(bcx.fcx);

        // Second, trans the base to the dest.
        assert_eq!(discr, Disr(0));

        let addr = adt::MaybeSizedValue::sized(addr);
        match expr_kind(bcx.tcx(), &base.expr) {
            ExprKind::RvalueDps | ExprKind::RvalueDatum if !bcx.fcx.type_needs_drop(ty) => {
                bcx = trans_into(bcx, &base.expr, SaveIn(addr.value));
            },
            ExprKind::RvalueStmt => {
                bug!("unexpected expr kind for struct base expr")
            }
            _ => {
                let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, &base.expr, "base"));
                for &(i, t) in &base.fields {
                    let datum = base_datum.get_element(
                            bcx, t, |srcval| adt::trans_field_ptr(bcx, &repr, srcval, discr, i));
                    assert!(type_is_sized(bcx.tcx(), datum.ty));
                    let dest = adt::trans_field_ptr(bcx, &repr, addr, discr, i);
                    bcx = datum.store_to(bcx, dest);
                }
            }
        }

        // Finally, move scratch field values into actual field locations
        for (i, datum) in scratch_vals {
            let dest = adt::trans_field_ptr(bcx, &repr, addr, discr, i);
            bcx = datum.store_to(bcx, dest);
        }
    } else {
        // No base means we can write all fields directly in place.
        let addr = adt::MaybeSizedValue::sized(addr);
        for &(i, ref e) in fields {
            let dest = adt::trans_field_ptr(bcx, &repr, addr, discr, i);
            let e_ty = expr_ty_adjusted(bcx, &e);
            bcx = trans_into(bcx, &e, SaveIn(dest));
            let scope = cleanup::CustomScope(custom_cleanup_scope);
            fcx.schedule_lifetime_end(scope, dest);
            // FIXME: nonzeroing move should generalize to fields
            fcx.schedule_drop_mem(scope, dest, e_ty, None);
        }
    }

    adt::trans_set_discr(bcx, &repr, addr, discr);

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
                                   expr: &hir::Expr,
                                   lit: &ast::Lit)
                                   -> DatumBlock<'blk, 'tcx, Expr> {
    // must not be a string constant, that is a RvalueDpsExpr
    let _icx = push_ctxt("trans_immediate_lit");
    let ty = expr_ty(bcx, expr);
    let v = consts::const_lit(bcx.ccx(), expr, lit);
    immediate_rvalue_bcx(bcx, v, ty).to_expr_datumblock()
}

fn trans_unary<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                           expr: &hir::Expr,
                           op: hir::UnOp,
                           sub_expr: &hir::Expr)
                           -> DatumBlock<'blk, 'tcx, Expr> {
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_unary_datum");

    let method_call = MethodCall::expr(expr.id);

    // The only overloaded operator that is translated to a datum
    // is an overloaded deref, since it is always yields a `&T`.
    // Otherwise, we should be in the RvalueDpsExpr path.
    assert!(op == hir::UnDeref || !ccx.tcx().is_method_call(expr.id));

    let un_ty = expr_ty(bcx, expr);

    let debug_loc = expr.debug_loc();

    match op {
        hir::UnNot => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            let llresult = Not(bcx, datum.to_llscalarish(bcx), debug_loc);
            immediate_rvalue_bcx(bcx, llresult, un_ty).to_expr_datumblock()
        }
        hir::UnNeg => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            let val = datum.to_llscalarish(bcx);
            let (bcx, llneg) = {
                if un_ty.is_fp() {
                    let result = FNeg(bcx, val, debug_loc);
                    (bcx, result)
                } else {
                    let is_signed = un_ty.is_signed();
                    let result = Neg(bcx, val, debug_loc);
                    let bcx = if bcx.ccx().check_overflow() && is_signed {
                        let (llty, min) = base::llty_and_min_for_signed_ty(bcx, un_ty);
                        let is_min = ICmp(bcx, llvm::IntEQ, val,
                                          C_integral(llty, min, true), debug_loc);
                        with_cond(bcx, is_min, |bcx| {
                            let msg = InternedString::new(
                                "attempted to negate with overflow");
                            controlflow::trans_fail(bcx, expr_info(expr), msg)
                        })
                    } else {
                        bcx
                    };
                    (bcx, result)
                }
            };
            immediate_rvalue_bcx(bcx, llneg, un_ty).to_expr_datumblock()
        }
        hir::UnDeref => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            deref_once(bcx, expr, datum, method_call)
        }
    }
}

fn trans_uniq_expr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               box_expr: &hir::Expr,
                               box_ty: Ty<'tcx>,
                               contents: &hir::Expr,
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

fn trans_addr_of<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             expr: &hir::Expr,
                             subexpr: &hir::Expr)
                             -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_addr_of");
    let mut bcx = bcx;
    let sub_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, subexpr, "addr_of"));
    let ty = expr_ty(bcx, expr);
    if !type_is_sized(bcx.tcx(), sub_datum.ty) {
        // Always generate an lvalue datum, because this pointer doesn't own
        // the data and cleanup is scheduled elsewhere.
        DatumBlock::new(bcx, Datum::new(sub_datum.val, ty, LvalueExpr(sub_datum.kind)))
    } else {
        // Sized value, ref to a thin pointer
        immediate_rvalue_bcx(bcx, sub_datum.val, ty).to_expr_datumblock()
    }
}

fn trans_scalar_binop<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  binop_expr: &hir::Expr,
                                  binop_ty: Ty<'tcx>,
                                  op: hir::BinOp,
                                  lhs: Datum<'tcx, Rvalue>,
                                  rhs: Datum<'tcx, Rvalue>)
                                  -> DatumBlock<'blk, 'tcx, Expr>
{
    let _icx = push_ctxt("trans_scalar_binop");

    let tcx = bcx.tcx();
    let lhs_t = lhs.ty;
    assert!(!lhs_t.is_simd());
    let is_float = lhs_t.is_fp();
    let is_signed = lhs_t.is_signed();
    let info = expr_info(binop_expr);

    let binop_debug_loc = binop_expr.debug_loc();

    let mut bcx = bcx;
    let lhs = lhs.to_llscalarish(bcx);
    let rhs = rhs.to_llscalarish(bcx);
    let val = match op.node {
      hir::BiAdd => {
        if is_float {
            FAdd(bcx, lhs, rhs, binop_debug_loc)
        } else {
            let (newbcx, res) = with_overflow_check(
                bcx, OverflowOp::Add, info, lhs_t, lhs, rhs, binop_debug_loc);
            bcx = newbcx;
            res
        }
      }
      hir::BiSub => {
        if is_float {
            FSub(bcx, lhs, rhs, binop_debug_loc)
        } else {
            let (newbcx, res) = with_overflow_check(
                bcx, OverflowOp::Sub, info, lhs_t, lhs, rhs, binop_debug_loc);
            bcx = newbcx;
            res
        }
      }
      hir::BiMul => {
        if is_float {
            FMul(bcx, lhs, rhs, binop_debug_loc)
        } else {
            let (newbcx, res) = with_overflow_check(
                bcx, OverflowOp::Mul, info, lhs_t, lhs, rhs, binop_debug_loc);
            bcx = newbcx;
            res
        }
      }
      hir::BiDiv => {
        if is_float {
            FDiv(bcx, lhs, rhs, binop_debug_loc)
        } else {
            // Only zero-check integers; fp /0 is NaN
            bcx = base::fail_if_zero_or_overflows(bcx,
                                                  expr_info(binop_expr),
                                                  op,
                                                  lhs,
                                                  rhs,
                                                  lhs_t);
            if is_signed {
                SDiv(bcx, lhs, rhs, binop_debug_loc)
            } else {
                UDiv(bcx, lhs, rhs, binop_debug_loc)
            }
        }
      }
      hir::BiRem => {
        if is_float {
            // LLVM currently always lowers the `frem` instructions appropriate
            // library calls typically found in libm. Notably f64 gets wired up
            // to `fmod` and f32 gets wired up to `fmodf`. Inconveniently for
            // us, 32-bit MSVC does not actually have a `fmodf` symbol, it's
            // instead just an inline function in a header that goes up to a
            // f64, uses `fmod`, and then comes back down to a f32.
            //
            // Although LLVM knows that `fmodf` doesn't exist on MSVC, it will
            // still unconditionally lower frem instructions over 32-bit floats
            // to a call to `fmodf`. To work around this we special case MSVC
            // 32-bit float rem instructions and instead do the call out to
            // `fmod` ourselves.
            //
            // Note that this is currently duplicated with src/libcore/ops.rs
            // which does the same thing, and it would be nice to perhaps unify
            // these two implementations on day! Also note that we call `fmod`
            // for both 32 and 64-bit floats because if we emit any FRem
            // instruction at all then LLVM is capable of optimizing it into a
            // 32-bit FRem (which we're trying to avoid).
            let use_fmod = tcx.sess.target.target.options.is_like_msvc &&
                           tcx.sess.target.target.arch == "x86";
            if use_fmod {
                let f64t = Type::f64(bcx.ccx());
                let fty = Type::func(&[f64t, f64t], &f64t);
                let llfn = declare::declare_cfn(bcx.ccx(), "fmod", fty);
                if lhs_t == tcx.types.f32 {
                    let lhs = FPExt(bcx, lhs, f64t);
                    let rhs = FPExt(bcx, rhs, f64t);
                    let res = Call(bcx, llfn, &[lhs, rhs], binop_debug_loc);
                    FPTrunc(bcx, res, Type::f32(bcx.ccx()))
                } else {
                    Call(bcx, llfn, &[lhs, rhs], binop_debug_loc)
                }
            } else {
                FRem(bcx, lhs, rhs, binop_debug_loc)
            }
        } else {
            // Only zero-check integers; fp %0 is NaN
            bcx = base::fail_if_zero_or_overflows(bcx,
                                                  expr_info(binop_expr),
                                                  op, lhs, rhs, lhs_t);
            if is_signed {
                SRem(bcx, lhs, rhs, binop_debug_loc)
            } else {
                URem(bcx, lhs, rhs, binop_debug_loc)
            }
        }
      }
      hir::BiBitOr => Or(bcx, lhs, rhs, binop_debug_loc),
      hir::BiBitAnd => And(bcx, lhs, rhs, binop_debug_loc),
      hir::BiBitXor => Xor(bcx, lhs, rhs, binop_debug_loc),
      hir::BiShl => {
          let (newbcx, res) = with_overflow_check(
              bcx, OverflowOp::Shl, info, lhs_t, lhs, rhs, binop_debug_loc);
          bcx = newbcx;
          res
      }
      hir::BiShr => {
          let (newbcx, res) = with_overflow_check(
              bcx, OverflowOp::Shr, info, lhs_t, lhs, rhs, binop_debug_loc);
          bcx = newbcx;
          res
      }
      hir::BiEq | hir::BiNe | hir::BiLt | hir::BiGe | hir::BiLe | hir::BiGt => {
          base::compare_scalar_types(bcx, lhs, rhs, lhs_t, op.node, binop_debug_loc)
      }
      _ => {
        span_bug!(binop_expr.span, "unexpected binop");
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
                                binop_expr: &hir::Expr,
                                op: lazy_binop_ty,
                                a: &hir::Expr,
                                b: &hir::Expr)
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
                            expr: &hir::Expr,
                            op: hir::BinOp,
                            lhs: &hir::Expr,
                            rhs: &hir::Expr)
                            -> DatumBlock<'blk, 'tcx, Expr> {
    let _icx = push_ctxt("trans_binary");
    let ccx = bcx.ccx();

    // if overloaded, would be RvalueDpsExpr
    assert!(!ccx.tcx().is_method_call(expr.id));

    match op.node {
        hir::BiAnd => {
            trans_lazy_binop(bcx, expr, lazy_and, lhs, rhs)
        }
        hir::BiOr => {
            trans_lazy_binop(bcx, expr, lazy_or, lhs, rhs)
        }
        _ => {
            let mut bcx = bcx;
            let binop_ty = expr_ty(bcx, expr);

            let lhs = unpack_datum!(bcx, trans(bcx, lhs));
            let lhs = unpack_datum!(bcx, lhs.to_rvalue_datum(bcx, "binop_lhs"));
            debug!("trans_binary (expr {}): lhs={:?}", expr.id, lhs);
            let rhs = unpack_datum!(bcx, trans(bcx, rhs));
            let rhs = unpack_datum!(bcx, rhs.to_rvalue_datum(bcx, "binop_rhs"));
            debug!("trans_binary (expr {}): rhs={:?}", expr.id, rhs);

            if type_is_fat_ptr(ccx.tcx(), lhs.ty) {
                assert!(type_is_fat_ptr(ccx.tcx(), rhs.ty),
                        "built-in binary operators on fat pointers are homogeneous");
                assert_eq!(binop_ty, bcx.tcx().types.bool);
                let val = base::compare_scalar_types(
                    bcx,
                    lhs.val,
                    rhs.val,
                    lhs.ty,
                    op.node,
                    expr.debug_loc());
                immediate_rvalue_bcx(bcx, val, binop_ty).to_expr_datumblock()
            } else {
                assert!(!type_is_fat_ptr(ccx.tcx(), rhs.ty),
                        "built-in binary operators on fat pointers are homogeneous");
                trans_scalar_binop(bcx, expr, binop_ty, op, lhs, rhs)
            }
        }
    }
}

pub fn cast_is_noop<'tcx>(tcx: &TyCtxt<'tcx>,
                          expr: &hir::Expr,
                          t_in: Ty<'tcx>,
                          t_out: Ty<'tcx>)
                          -> bool {
    if let Some(&CastKind::CoercionCast) = tcx.cast_kinds.borrow().get(&expr.id) {
        return true;
    }

    match (t_in.builtin_deref(true, ty::NoPreference),
           t_out.builtin_deref(true, ty::NoPreference)) {
        (Some(ty::TypeAndMut{ ty: t_in, .. }), Some(ty::TypeAndMut{ ty: t_out, .. })) => {
            t_in == t_out
        }
        _ => {
            // This condition isn't redundant with the check for CoercionCast:
            // different types can be substituted into the same type, and
            // == equality can be overconservative if there are regions.
            t_in == t_out
        }
    }
}

fn trans_imm_cast<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &hir::Expr,
                              id: ast::NodeId)
                              -> DatumBlock<'blk, 'tcx, Expr>
{
    use rustc::ty::cast::CastTy::*;
    use rustc::ty::cast::IntTy::*;

    fn int_cast(bcx: Block,
                lldsttype: Type,
                llsrctype: Type,
                llsrc: ValueRef,
                signed: bool)
                -> ValueRef
    {
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
                  -> ValueRef
    {
        let _icx = push_ctxt("float_cast");
        let srcsz = llsrctype.float_width();
        let dstsz = lldsttype.float_width();
        return if dstsz > srcsz {
            FPExt(bcx, llsrc, lldsttype)
        } else if srcsz > dstsz {
            FPTrunc(bcx, llsrc, lldsttype)
        } else { llsrc };
    }

    let _icx = push_ctxt("trans_cast");
    let mut bcx = bcx;
    let ccx = bcx.ccx();

    let t_in = expr_ty_adjusted(bcx, expr);
    let t_out = node_id_type(bcx, id);

    debug!("trans_cast({:?} as {:?})", t_in, t_out);
    let mut ll_t_in = type_of::immediate_type_of(ccx, t_in);
    let ll_t_out = type_of::immediate_type_of(ccx, t_out);
    // Convert the value to be cast into a ValueRef, either by-ref or
    // by-value as appropriate given its type:
    let mut datum = unpack_datum!(bcx, trans(bcx, expr));

    let datum_ty = monomorphize_type(bcx, datum.ty);

    if cast_is_noop(bcx.tcx(), expr, datum_ty, t_out) {
        datum.ty = t_out;
        return DatumBlock::new(bcx, datum);
    }

    if type_is_fat_ptr(bcx.tcx(), t_in) {
        assert!(datum.kind.is_by_ref());
        if type_is_fat_ptr(bcx.tcx(), t_out) {
            return DatumBlock::new(bcx, Datum::new(
                PointerCast(bcx, datum.val, ll_t_out.ptr_to()),
                t_out,
                Rvalue::new(ByRef)
            )).to_expr_datumblock();
        } else {
            // Return the address
            return immediate_rvalue_bcx(bcx,
                                        PointerCast(bcx,
                                                    Load(bcx, get_dataptr(bcx, datum.val)),
                                                    ll_t_out),
                                        t_out).to_expr_datumblock();
        }
    }

    let r_t_in = CastTy::from_ty(t_in).expect("bad input type for cast");
    let r_t_out = CastTy::from_ty(t_out).expect("bad output type for cast");

    let (llexpr, signed) = if let Int(CEnum) = r_t_in {
        let repr = adt::represent_type(ccx, t_in);
        let datum = unpack_datum!(
            bcx, datum.to_lvalue_datum(bcx, "trans_imm_cast", expr.id));
        let llexpr_ptr = datum.to_llref();
        let discr = adt::trans_get_discr(bcx, &repr, llexpr_ptr,
                                         Some(Type::i64(ccx)), true);
        ll_t_in = val_ty(discr);
        (discr, adt::is_discr_signed(&repr))
    } else {
        (datum.to_llscalarish(bcx), t_in.is_signed())
    };

    let newval = match (r_t_in, r_t_out) {
        (Ptr(_), Ptr(_)) | (FnPtr, Ptr(_)) | (RPtr(_), Ptr(_)) => {
            PointerCast(bcx, llexpr, ll_t_out)
        }
        (Ptr(_), Int(_)) | (FnPtr, Int(_)) => PtrToInt(bcx, llexpr, ll_t_out),
        (Int(_), Ptr(_)) => IntToPtr(bcx, llexpr, ll_t_out),

        (Int(_), Int(_)) => int_cast(bcx, ll_t_out, ll_t_in, llexpr, signed),
        (Float, Float) => float_cast(bcx, ll_t_out, ll_t_in, llexpr),
        (Int(_), Float) if signed => SIToFP(bcx, llexpr, ll_t_out),
        (Int(_), Float) => UIToFP(bcx, llexpr, ll_t_out),
        (Float, Int(I)) => FPToSI(bcx, llexpr, ll_t_out),
        (Float, Int(_)) => FPToUI(bcx, llexpr, ll_t_out),

        _ => span_bug!(expr.span,
                       "translating unsupported cast: \
                        {:?} -> {:?}",
                       t_in,
                       t_out)
    };
    return immediate_rvalue_bcx(bcx, newval, t_out).to_expr_datumblock();
}

fn trans_assign_op<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                               expr: &hir::Expr,
                               op: hir::BinOp,
                               dst: &hir::Expr,
                               src: &hir::Expr)
                               -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("trans_assign_op");
    let mut bcx = bcx;

    debug!("trans_assign_op(expr={:?})", expr);

    // User-defined operator methods cannot be used with `+=` etc right now
    assert!(!bcx.tcx().is_method_call(expr.id));

    // Evaluate LHS (destination), which should be an lvalue
    let dst = unpack_datum!(bcx, trans_to_lvalue(bcx, dst, "assign_op"));
    assert!(!bcx.fcx.type_needs_drop(dst.ty));
    let lhs = load_ty(bcx, dst.val, dst.ty);
    let lhs = immediate_rvalue(lhs, dst.ty);

    // Evaluate RHS - FIXME(#28160) this sucks
    let rhs = unpack_datum!(bcx, trans(bcx, &src));
    let rhs = unpack_datum!(bcx, rhs.to_rvalue_datum(bcx, "assign_op_rhs"));

    // Perform computation and store the result
    let result_datum = unpack_datum!(
        bcx, trans_scalar_binop(bcx, expr, dst.ty, op, lhs, rhs));
    return result_datum.store_to(bcx, dst.val);
}

fn auto_ref<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                        datum: Datum<'tcx, Expr>,
                        expr: &hir::Expr)
                        -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;

    // Ensure cleanup of `datum` if not already scheduled and obtain
    // a "by ref" pointer.
    let lv_datum = unpack_datum!(bcx, datum.to_lvalue_datum(bcx, "autoref", expr.id));

    // Compute final type. Note that we are loose with the region and
    // mutability, since those things don't matter in trans.
    let referent_ty = lv_datum.ty;
    let ptr_ty = bcx.tcx().mk_imm_ref(bcx.tcx().mk_region(ty::ReStatic), referent_ty);

    // Construct the resulting datum. The right datum to return here would be an Lvalue datum,
    // because there is cleanup scheduled and the datum doesn't own the data, but for thin pointers
    // we microoptimize it to be an Rvalue datum to avoid the extra alloca and level of
    // indirection and for thin pointers, this has no ill effects.
    let kind  = if type_is_sized(bcx.tcx(), referent_ty) {
        RvalueExpr(Rvalue::new(ByValue))
    } else {
        LvalueExpr(lv_datum.kind)
    };

    // Get the pointer.
    let llref = lv_datum.to_llref();
    DatumBlock::new(bcx, Datum::new(llref, ptr_ty, kind))
}

fn deref_multiple<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &hir::Expr,
                              datum: Datum<'tcx, Expr>,
                              times: usize)
                              -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;
    let mut datum = datum;
    for i in 0..times {
        let method_call = MethodCall::autoderef(expr.id, i as u32);
        datum = unpack_datum!(bcx, deref_once(bcx, expr, datum, method_call));
    }
    DatumBlock { bcx: bcx, datum: datum }
}

fn deref_once<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                          expr: &hir::Expr,
                          datum: Datum<'tcx, Expr>,
                          method_call: MethodCall)
                          -> DatumBlock<'blk, 'tcx, Expr> {
    let ccx = bcx.ccx();

    debug!("deref_once(expr={:?}, datum={:?}, method_call={:?})",
           expr, datum, method_call);

    let mut bcx = bcx;

    // Check for overloaded deref.
    let method = ccx.tcx().tables.borrow().method_map.get(&method_call).cloned();
    let datum = match method {
        Some(method) => {
            let method_ty = monomorphize_type(bcx, method.ty);

            // Overloaded. Invoke the deref() method, which basically
            // converts from the `Smaht<T>` pointer that we have into
            // a `&T` pointer.  We can then proceed down the normal
            // path (below) to dereference that `&T`.
            let datum = if method_call.autoderef == 0 {
                datum
            } else {
                // Always perform an AutoPtr when applying an overloaded auto-deref
                unpack_datum!(bcx, auto_ref(bcx, datum, expr))
            };

            let ref_ty = // invoked methods have their LB regions instantiated
                ccx.tcx().no_late_bound_regions(&method_ty.fn_ret()).unwrap().unwrap();
            let scratch = rvalue_scratch_datum(bcx, ref_ty, "overloaded_deref");

            bcx = Callee::method(bcx, method)
                .call(bcx, expr.debug_loc(),
                      ArgOverloadedOp(datum, None),
                      Some(SaveIn(scratch.val))).bcx;
            scratch.to_expr_datum()
        }
        None => {
            // Not overloaded. We already have a pointer we know how to deref.
            datum
        }
    };

    let r = match datum.ty.sty {
        ty::TyBox(content_ty) => {
            // Make sure we have an lvalue datum here to get the
            // proper cleanups scheduled
            let datum = unpack_datum!(
                bcx, datum.to_lvalue_datum(bcx, "deref", expr.id));

            if type_is_sized(bcx.tcx(), content_ty) {
                let ptr = load_ty(bcx, datum.val, datum.ty);
                DatumBlock::new(bcx, Datum::new(ptr, content_ty, LvalueExpr(datum.kind)))
            } else {
                // A fat pointer and a DST lvalue have the same representation
                // just different types. Since there is no temporary for `*e`
                // here (because it is unsized), we cannot emulate the sized
                // object code path for running drop glue and free. Instead,
                // we schedule cleanup for `e`, turning it into an lvalue.

                let lval = Lvalue::new("expr::deref_once ty_uniq");
                let datum = Datum::new(datum.val, content_ty, LvalueExpr(lval));
                DatumBlock::new(bcx, datum)
            }
        }

        ty::TyRawPtr(ty::TypeAndMut { ty: content_ty, .. }) |
        ty::TyRef(_, ty::TypeAndMut { ty: content_ty, .. }) => {
            let lval = Lvalue::new("expr::deref_once ptr");
            if type_is_sized(bcx.tcx(), content_ty) {
                let ptr = datum.to_llscalarish(bcx);

                // Always generate an lvalue datum, even if datum.mode is
                // an rvalue.  This is because datum.mode is only an
                // rvalue for non-owning pointers like &T or *T, in which
                // case cleanup *is* scheduled elsewhere, by the true
                // owner (or, in the case of *T, by the user).
                DatumBlock::new(bcx, Datum::new(ptr, content_ty, LvalueExpr(lval)))
            } else {
                // A fat pointer and a DST lvalue have the same representation
                // just different types.
                DatumBlock::new(bcx, Datum::new(datum.val, content_ty, LvalueExpr(lval)))
            }
        }

        _ => {
            span_bug!(
                expr.span,
                "deref invoked on expr of invalid type {:?}",
                datum.ty);
        }
    };

    debug!("deref_once(expr={}, method_call={:?}, result={:?})",
           expr.id, method_call, r.datum);

    return r;
}

#[derive(Debug)]
enum OverflowOp {
    Add,
    Sub,
    Mul,
    Shl,
    Shr,
}

impl OverflowOp {
    fn codegen_strategy(&self) -> OverflowCodegen {
        use self::OverflowCodegen::{ViaIntrinsic, ViaInputCheck};
        match *self {
            OverflowOp::Add => ViaIntrinsic(OverflowOpViaIntrinsic::Add),
            OverflowOp::Sub => ViaIntrinsic(OverflowOpViaIntrinsic::Sub),
            OverflowOp::Mul => ViaIntrinsic(OverflowOpViaIntrinsic::Mul),

            OverflowOp::Shl => ViaInputCheck(OverflowOpViaInputCheck::Shl),
            OverflowOp::Shr => ViaInputCheck(OverflowOpViaInputCheck::Shr),
        }
    }
}

enum OverflowCodegen {
    ViaIntrinsic(OverflowOpViaIntrinsic),
    ViaInputCheck(OverflowOpViaInputCheck),
}

enum OverflowOpViaInputCheck { Shl, Shr, }

#[derive(Debug)]
enum OverflowOpViaIntrinsic { Add, Sub, Mul, }

impl OverflowOpViaIntrinsic {
    fn to_intrinsic<'blk, 'tcx>(&self, bcx: Block<'blk, 'tcx>, lhs_ty: Ty) -> ValueRef {
        let name = self.to_intrinsic_name(bcx.tcx(), lhs_ty);
        bcx.ccx().get_intrinsic(&name)
    }
    fn to_intrinsic_name(&self, tcx: &TyCtxt, ty: Ty) -> &'static str {
        use syntax::ast::IntTy::*;
        use syntax::ast::UintTy::*;
        use rustc::ty::{TyInt, TyUint};

        let new_sty = match ty.sty {
            TyInt(Is) => match &tcx.sess.target.target.target_pointer_width[..] {
                "32" => TyInt(I32),
                "64" => TyInt(I64),
                _ => bug!("unsupported target word size")
            },
            TyUint(Us) => match &tcx.sess.target.target.target_pointer_width[..] {
                "32" => TyUint(U32),
                "64" => TyUint(U64),
                _ => bug!("unsupported target word size")
            },
            ref t @ TyUint(_) | ref t @ TyInt(_) => t.clone(),
            _ => bug!("tried to get overflow intrinsic for {:?} applied to non-int type",
                      *self)
        };

        match *self {
            OverflowOpViaIntrinsic::Add => match new_sty {
                TyInt(I8) => "llvm.sadd.with.overflow.i8",
                TyInt(I16) => "llvm.sadd.with.overflow.i16",
                TyInt(I32) => "llvm.sadd.with.overflow.i32",
                TyInt(I64) => "llvm.sadd.with.overflow.i64",

                TyUint(U8) => "llvm.uadd.with.overflow.i8",
                TyUint(U16) => "llvm.uadd.with.overflow.i16",
                TyUint(U32) => "llvm.uadd.with.overflow.i32",
                TyUint(U64) => "llvm.uadd.with.overflow.i64",

                _ => bug!(),
            },
            OverflowOpViaIntrinsic::Sub => match new_sty {
                TyInt(I8) => "llvm.ssub.with.overflow.i8",
                TyInt(I16) => "llvm.ssub.with.overflow.i16",
                TyInt(I32) => "llvm.ssub.with.overflow.i32",
                TyInt(I64) => "llvm.ssub.with.overflow.i64",

                TyUint(U8) => "llvm.usub.with.overflow.i8",
                TyUint(U16) => "llvm.usub.with.overflow.i16",
                TyUint(U32) => "llvm.usub.with.overflow.i32",
                TyUint(U64) => "llvm.usub.with.overflow.i64",

                _ => bug!(),
            },
            OverflowOpViaIntrinsic::Mul => match new_sty {
                TyInt(I8) => "llvm.smul.with.overflow.i8",
                TyInt(I16) => "llvm.smul.with.overflow.i16",
                TyInt(I32) => "llvm.smul.with.overflow.i32",
                TyInt(I64) => "llvm.smul.with.overflow.i64",

                TyUint(U8) => "llvm.umul.with.overflow.i8",
                TyUint(U16) => "llvm.umul.with.overflow.i16",
                TyUint(U32) => "llvm.umul.with.overflow.i32",
                TyUint(U64) => "llvm.umul.with.overflow.i64",

                _ => bug!(),
            },
        }
    }

    fn build_intrinsic_call<'blk, 'tcx>(&self, bcx: Block<'blk, 'tcx>,
                                        info: NodeIdAndSpan,
                                        lhs_t: Ty<'tcx>, lhs: ValueRef,
                                        rhs: ValueRef,
                                        binop_debug_loc: DebugLoc)
                                        -> (Block<'blk, 'tcx>, ValueRef) {
        let llfn = self.to_intrinsic(bcx, lhs_t);

        let val = Call(bcx, llfn, &[lhs, rhs], binop_debug_loc);
        let result = ExtractValue(bcx, val, 0); // iN operation result
        let overflow = ExtractValue(bcx, val, 1); // i1 "did it overflow?"

        let cond = ICmp(bcx, llvm::IntEQ, overflow, C_integral(Type::i1(bcx.ccx()), 1, false),
                        binop_debug_loc);

        let expect = bcx.ccx().get_intrinsic(&"llvm.expect.i1");
        Call(bcx, expect, &[cond, C_integral(Type::i1(bcx.ccx()), 0, false)],
             binop_debug_loc);

        let bcx =
            base::with_cond(bcx, cond, |bcx|
                controlflow::trans_fail(bcx, info,
                    InternedString::new("arithmetic operation overflowed")));

        (bcx, result)
    }
}

impl OverflowOpViaInputCheck {
    fn build_with_input_check<'blk, 'tcx>(&self,
                                          bcx: Block<'blk, 'tcx>,
                                          info: NodeIdAndSpan,
                                          lhs_t: Ty<'tcx>,
                                          lhs: ValueRef,
                                          rhs: ValueRef,
                                          binop_debug_loc: DebugLoc)
                                          -> (Block<'blk, 'tcx>, ValueRef)
    {
        let lhs_llty = val_ty(lhs);
        let rhs_llty = val_ty(rhs);

        // Panic if any bits are set outside of bits that we always
        // mask in.
        //
        // Note that the mask's value is derived from the LHS type
        // (since that is where the 32/64 distinction is relevant) but
        // the mask's type must match the RHS type (since they will
        // both be fed into an and-binop)
        let invert_mask = shift_mask_val(bcx, lhs_llty, rhs_llty, true);

        let outer_bits = And(bcx, rhs, invert_mask, binop_debug_loc);
        let cond = build_nonzero_check(bcx, outer_bits, binop_debug_loc);
        let result = match *self {
            OverflowOpViaInputCheck::Shl =>
                build_unchecked_lshift(bcx, lhs, rhs, binop_debug_loc),
            OverflowOpViaInputCheck::Shr =>
                build_unchecked_rshift(bcx, lhs_t, lhs, rhs, binop_debug_loc),
        };
        let bcx =
            base::with_cond(bcx, cond, |bcx|
                controlflow::trans_fail(bcx, info,
                    InternedString::new("shift operation overflowed")));

        (bcx, result)
    }
}

// Check if an integer or vector contains a nonzero element.
fn build_nonzero_check<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   value: ValueRef,
                                   binop_debug_loc: DebugLoc) -> ValueRef {
    let llty = val_ty(value);
    let kind = llty.kind();
    match kind {
        TypeKind::Integer => ICmp(bcx, llvm::IntNE, value, C_null(llty), binop_debug_loc),
        TypeKind::Vector => {
            // Check if any elements of the vector are nonzero by treating
            // it as a wide integer and checking if the integer is nonzero.
            let width = llty.vector_length() as u64 * llty.element_type().int_width();
            let int_value = BitCast(bcx, value, Type::ix(bcx.ccx(), width));
            build_nonzero_check(bcx, int_value, binop_debug_loc)
        },
        _ => bug!("build_nonzero_check: expected Integer or Vector, found {:?}", kind),
    }
}

fn with_overflow_check<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, oop: OverflowOp, info: NodeIdAndSpan,
                                   lhs_t: Ty<'tcx>, lhs: ValueRef,
                                   rhs: ValueRef,
                                   binop_debug_loc: DebugLoc)
                                   -> (Block<'blk, 'tcx>, ValueRef) {
    if bcx.unreachable.get() { return (bcx, _Undef(lhs)); }
    if bcx.ccx().check_overflow() {

        match oop.codegen_strategy() {
            OverflowCodegen::ViaIntrinsic(oop) =>
                oop.build_intrinsic_call(bcx, info, lhs_t, lhs, rhs, binop_debug_loc),
            OverflowCodegen::ViaInputCheck(oop) =>
                oop.build_with_input_check(bcx, info, lhs_t, lhs, rhs, binop_debug_loc),
        }
    } else {
        let res = match oop {
            OverflowOp::Add => Add(bcx, lhs, rhs, binop_debug_loc),
            OverflowOp::Sub => Sub(bcx, lhs, rhs, binop_debug_loc),
            OverflowOp::Mul => Mul(bcx, lhs, rhs, binop_debug_loc),

            OverflowOp::Shl =>
                build_unchecked_lshift(bcx, lhs, rhs, binop_debug_loc),
            OverflowOp::Shr =>
                build_unchecked_rshift(bcx, lhs_t, lhs, rhs, binop_debug_loc),
        };
        (bcx, res)
    }
}

/// We categorize expressions into three kinds.  The distinction between
/// lvalue/rvalue is fundamental to the language.  The distinction between the
/// two kinds of rvalues is an artifact of trans which reflects how we will
/// generate code for that kind of expression.  See trans/expr.rs for more
/// information.
#[derive(Copy, Clone)]
enum ExprKind {
    Lvalue,
    RvalueDps,
    RvalueDatum,
    RvalueStmt
}

fn expr_kind(tcx: &TyCtxt, expr: &hir::Expr) -> ExprKind {
    if tcx.is_method_call(expr.id) {
        // Overloaded operations are generally calls, and hence they are
        // generated via DPS, but there are a few exceptions:
        return match expr.node {
            // `a += b` has a unit result.
            hir::ExprAssignOp(..) => ExprKind::RvalueStmt,

            // the deref method invoked for `*a` always yields an `&T`
            hir::ExprUnary(hir::UnDeref, _) => ExprKind::Lvalue,

            // the index method invoked for `a[i]` always yields an `&T`
            hir::ExprIndex(..) => ExprKind::Lvalue,

            // in the general case, result could be any type, use DPS
            _ => ExprKind::RvalueDps
        };
    }

    match expr.node {
        hir::ExprPath(..) => {
            match tcx.resolve_expr(expr) {
                // Put functions and ctors with the ADTs, as they
                // are zero-sized, so DPS is the cheapest option.
                Def::Struct(..) | Def::Variant(..) |
                Def::Fn(..) | Def::Method(..) => {
                    ExprKind::RvalueDps
                }

                // Note: there is actually a good case to be made that
                // DefArg's, particularly those of immediate type, ought to
                // considered rvalues.
                Def::Static(..) |
                Def::Upvar(..) |
                Def::Local(..) => ExprKind::Lvalue,

                Def::Const(..) |
                Def::AssociatedConst(..) => ExprKind::RvalueDatum,

                def => {
                    span_bug!(
                        expr.span,
                        "uncategorized def for expr {}: {:?}",
                        expr.id,
                        def);
                }
            }
        }

        hir::ExprType(ref expr, _) => {
            expr_kind(tcx, expr)
        }

        hir::ExprUnary(hir::UnDeref, _) |
        hir::ExprField(..) |
        hir::ExprTupField(..) |
        hir::ExprIndex(..) => {
            ExprKind::Lvalue
        }

        hir::ExprCall(..) |
        hir::ExprMethodCall(..) |
        hir::ExprStruct(..) |
        hir::ExprTup(..) |
        hir::ExprIf(..) |
        hir::ExprMatch(..) |
        hir::ExprClosure(..) |
        hir::ExprBlock(..) |
        hir::ExprRepeat(..) |
        hir::ExprVec(..) => {
            ExprKind::RvalueDps
        }

        hir::ExprLit(ref lit) if lit.node.is_str() => {
            ExprKind::RvalueDps
        }

        hir::ExprBreak(..) |
        hir::ExprAgain(..) |
        hir::ExprRet(..) |
        hir::ExprWhile(..) |
        hir::ExprLoop(..) |
        hir::ExprAssign(..) |
        hir::ExprInlineAsm(..) |
        hir::ExprAssignOp(..) => {
            ExprKind::RvalueStmt
        }

        hir::ExprLit(_) | // Note: LitStr is carved out above
        hir::ExprUnary(..) |
        hir::ExprBox(_) |
        hir::ExprAddrOf(..) |
        hir::ExprBinary(..) |
        hir::ExprCast(..) => {
            ExprKind::RvalueDatum
        }
    }
}
