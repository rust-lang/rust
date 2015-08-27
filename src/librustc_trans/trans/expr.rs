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

pub use self::Dest::*;
use self::lazy_binop_ty::*;

use back::abi;
use llvm::{self, ValueRef, TypeKind};
use middle::check_const;
use middle::def;
use middle::lang_items::CoerceUnsizedTraitLangItem;
use middle::subst::{Substs, VecPerParamSpace};
use middle::traits;
use trans::{_match, adt, asm, base, callee, closure, consts, controlflow};
use trans::base::*;
use trans::build::*;
use trans::cleanup::{self, CleanupMethods, DropHintMethods};
use trans::common::*;
use trans::datum::*;
use trans::debuginfo::{self, DebugLoc, ToDebugLoc};
use trans::declare;
use trans::glue;
use trans::machine;
use trans::meth;
use trans::tvec;
use trans::type_of;
use middle::cast::{CastKind, CastTy};
use middle::ty::{AdjustDerefRef, AdjustReifyFnPointer, AdjustUnsafeFnPointer};
use middle::ty::{self, Ty};
use middle::ty::MethodCall;
use util::common::indenter;
use trans::machine::{llsize_of, llsize_of_alloc};
use trans::type_::Type;

use syntax::{ast, ast_util, codemap};
use syntax::parse::token::InternedString;
use syntax::ptr::P;
use syntax::parse::token;
use std::mem;

// Destinations

// These are passed around by the code generating functions to track the
// destination of a computation's value.

#[derive(Copy, Clone, PartialEq)]
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

    if bcx.tcx().tables.borrow().adjustments.contains_key(&expr.id) {
        // use trans, which may be less efficient but
        // which will perform the adjustments:
        let datum = unpack_datum!(bcx, trans(bcx, expr));
        return datum.store_to_dest(bcx, dest, expr.id);
    }

    let qualif = *bcx.tcx().const_qualif_map.borrow().get(&expr.id).unwrap();
    if !qualif.intersects(
        check_const::ConstQualif::NOT_CONST |
        check_const::ConstQualif::NEEDS_DROP
    ) {
        if !qualif.intersects(check_const::ConstQualif::PREFER_IN_PLACE) {
            if let SaveIn(lldest) = dest {
                let global = consts::get_const_expr_as_global(bcx.ccx(), expr, qualif,
                                                            bcx.fcx.param_substs);
                // Cast pointer to destination, because constants
                // have different types.
                let lldest = PointerCast(bcx, lldest, val_ty(global));
                memcpy_ty(bcx, lldest, global, expr_ty_adjusted(bcx, expr));
                return bcx;
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
                ast::ExprPath(..) => {
                    match bcx.def(expr.id) {
                        def::DefConst(did) => {
                            let const_expr = consts::get_const_expr(bcx.ccx(), did, expr);
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
                         expr: &ast::Expr)
                         -> DatumBlock<'blk, 'tcx, Expr> {
    debug!("trans(expr={:?})", expr);

    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let qualif = *bcx.tcx().const_qualif_map.borrow().get(&expr.id).unwrap();
    let adjusted_global = !qualif.intersects(check_const::ConstQualif::NON_STATIC_BORROWS);
    let global = if !qualif.intersects(
        check_const::ConstQualif::NOT_CONST |
        check_const::ConstQualif::NEEDS_DROP
    ) {
        let global = consts::get_const_expr_as_global(bcx.ccx(), expr, qualif,
                                                      bcx.fcx.param_substs);

        if qualif.intersects(check_const::ConstQualif::HAS_STATIC_BORROWS) {
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
            let llty = type_of::type_of(bcx.ccx(), const_ty);
            // HACK(eddyb) get around issues with lifetime intrinsics.
            let scratch = alloca_no_lifetime(bcx, llty, "const");
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

/// Retrieve the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit funny. It is intended for use
/// in an upcast, where the new vtable for an object will be drived
/// from the old one.
pub fn unsized_info<'ccx, 'tcx>(ccx: &CrateContext<'ccx, 'tcx>,
                                source: Ty<'tcx>,
                                target: Ty<'tcx>,
                                old_info: Option<ValueRef>,
                                param_substs: &'tcx Substs<'tcx>)
                                -> ValueRef {
    let (source, target) = ccx.tcx().struct_lockstep_tails(source, target);
    match (&source.sty, &target.sty) {
        (&ty::TyArray(_, len), &ty::TySlice(_)) => C_uint(ccx, len),
        (&ty::TyTrait(_), &ty::TyTrait(_)) => {
            // For now, upcasts are limited to changes in marker
            // traits, and hence never actually require an actual
            // change to the vtable.
            old_info.expect("unsized_info: missing old info for trait upcast")
        }
        (_, &ty::TyTrait(box ty::TraitTy { ref principal, .. })) => {
            // Note that we preserve binding levels here:
            let substs = principal.0.substs.with_self_ty(source).erase_regions();
            let substs = ccx.tcx().mk_substs(substs);
            let trait_ref = ty::Binder(ty::TraitRef { def_id: principal.def_id(),
                                                      substs: substs });
            consts::ptrcast(meth::get_vtable(ccx, trait_ref, param_substs),
                            Type::vtable_ptr(ccx))
        }
        _ => ccx.sess().bug(&format!("unsized_info: invalid unsizing {:?} -> {:?}",
                                     source,
                                     target))
    }
}

/// Helper for trans that apply adjustments from `expr` to `datum`, which should be the unadjusted
/// translation of `expr`.
fn apply_adjustments<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 expr: &ast::Expr,
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
    debug!("unadjusted datum for expr {:?}: {} adjustment={:?}",
           expr,
           datum.to_string(bcx.ccx()),
           adjustment);
    match adjustment {
        AdjustReifyFnPointer => {
            // FIXME(#19925) once fn item types are
            // zero-sized, we'll need to do something here
        }
        AdjustUnsafeFnPointer => {
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
                let llty = type_of::type_of(bcx.ccx(), target);

                // HACK(eddyb) get around issues with lifetime intrinsics.
                let scratch = alloca_no_lifetime(bcx, llty, "__coerce_target");
                let target_datum = Datum::new(scratch, target,
                                              Rvalue::new(ByRef));
                bcx = coerce_unsized(bcx, expr.span, source_datum, target_datum);
                datum = Datum::new(scratch, target,
                                   RvalueExpr(Rvalue::new(ByRef)));
            }
        }
    }
    debug!("after adjustments, datum={}", datum.to_string(bcx.ccx()));
    DatumBlock::new(bcx, datum)
}

fn coerce_unsized<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              span: codemap::Span,
                              source: Datum<'tcx, Rvalue>,
                              target: Datum<'tcx, Rvalue>)
                              -> Block<'blk, 'tcx> {
    let mut bcx = bcx;
    debug!("coerce_unsized({} -> {})",
           source.to_string(bcx.ccx()),
           target.to_string(bcx.ccx()));

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

            let info = unsized_info(bcx.ccx(), inner_source, inner_target,
                                    old_info, bcx.fcx.param_substs);

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

            let trait_substs = Substs::erased(VecPerParamSpace::new(vec![target.ty],
                                                                    vec![source.ty],
                                                                    Vec::new()));
            let trait_ref = ty::Binder(ty::TraitRef {
                def_id: langcall(bcx, Some(span), "coercion",
                                 CoerceUnsizedTraitLangItem),
                substs: bcx.tcx().mk_substs(trait_substs)
            });

            let kind = match fulfill_obligation(bcx.ccx(), span, trait_ref) {
                traits::VtableImpl(traits::VtableImplData { impl_def_id, .. }) => {
                    bcx.tcx().custom_coerce_unsized_kind(impl_def_id)
                }
                vtable => {
                    bcx.sess().span_bug(span, &format!("invalid CoerceUnsized vtable: {:?}",
                                                       vtable));
                }
            };

            let repr_source = adt::represent_type(bcx.ccx(), source.ty);
            let src_fields = match &*repr_source {
                &adt::Repr::Univariant(ref s, _) => &s.fields,
                _ => bcx.sess().span_bug(span,
                                         &format!("Non univariant struct? (repr_source: {:?})",
                                                  repr_source)),
            };
            let repr_target = adt::represent_type(bcx.ccx(), target.ty);
            let target_fields = match &*repr_target {
                &adt::Repr::Univariant(ref s, _) => &s.fields,
                _ => bcx.sess().span_bug(span,
                                         &format!("Non univariant struct? (repr_target: {:?})",
                                                  repr_target)),
            };

            let coerce_index = match kind {
                ty::CustomCoerceUnsized::Struct(i) => i
            };
            assert!(coerce_index < src_fields.len() && src_fields.len() == target_fields.len());

            let iter = src_fields.iter().zip(target_fields).enumerate();
            for (i, (src_ty, target_ty)) in iter {
                let ll_source = adt::trans_field_ptr(bcx, &repr_source, source.val, 0, i);
                let ll_target = adt::trans_field_ptr(bcx, &repr_target, target.val, 0, i);

                // If this is the field we need to coerce, recurse on it.
                if i == coerce_index {
                    coerce_unsized(bcx, span,
                                   Datum::new(ll_source, src_ty,
                                              Rvalue::new(ByRef)),
                                   Datum::new(ll_target, target_ty,
                                              Rvalue::new(ByRef)));
                } else {
                    // Otherwise, simply copy the data from the source.
                    assert_eq!(src_ty, target_ty);
                    memcpy_ty(bcx, ll_target, ll_source, src_ty);
                }
            }
        }
        _ => bcx.sess().bug(&format!("coerce_unsized: invalid coercion {:?} -> {:?}",
                                     source.ty,
                                     target.ty))
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

    debug!("trans_unadjusted(expr={:?})", expr);
    let _indenter = indenter();

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

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
                                      expr: &ast::Expr)
                                      -> DatumBlock<'blk, 'tcx, Expr> {
    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let _icx = push_ctxt("trans_datum_unadjusted");

    match expr.node {
        ast::ExprParen(ref e) => {
            trans(bcx, &**e)
        }
        ast::ExprPath(..) => {
            trans_def(bcx, expr, bcx.def(expr.id))
        }
        ast::ExprField(ref base, ident) => {
            trans_rec_field(bcx, &**base, ident.node.name)
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
                ty::TyBox(..) => {
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
                        expr.node));
        }
    }
}

fn trans_field<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                              base: &ast::Expr,
                              get_idx: F)
                              -> DatumBlock<'blk, 'tcx, Expr> where
    F: FnOnce(&'blk ty::ctxt<'tcx>, &VariantInfo<'tcx>) -> usize,
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
        |srcval| adt::trans_field_ptr(bcx, &*repr, srcval, vinfo.discr, ix));

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
                               base: &ast::Expr,
                               field: ast::Name)
                               -> DatumBlock<'blk, 'tcx, Expr> {
    trans_field(bcx, base, |_, vinfo| vinfo.field_index(field))
}

/// Translates `base.<idx>`.
fn trans_rec_tup_field<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   base: &ast::Expr,
                                   idx: usize)
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
                       .tables
                       .borrow()
                       .method_map
                       .get(&method_call)
                       .map(|method| method.ty);
    let elt_datum = match method_ty {
        Some(method_ty) => {
            let method_ty = monomorphize_type(bcx, method_ty);

            let base_datum = unpack_datum!(bcx, trans(bcx, base));

            // Translate index expression.
            let ix_datum = unpack_datum!(bcx, trans(bcx, idx));

            let ref_ty = // invoked methods have LB regions instantiated:
                bcx.tcx().no_late_bound_regions(&method_ty.fn_ret()).unwrap().unwrap();
            let elt_ty = match ref_ty.builtin_deref(true) {
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
                                               Some((ix_datum, idx.id)),
                                               Some(SaveIn(scratch.val)),
                                               false));
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
            let elt = PointerCast(bcx, elt, type_of::type_of(ccx, unit_ty).ptr_to());
            let lval = Lvalue::new("expr::trans_index fallback");
            Datum::new(elt, unit_ty, LvalueExpr(lval))
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
        def::DefFn(..) | def::DefMethod(..) |
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
            let val = if did.is_local() {
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
            let lval = Lvalue::new("expr::trans_def");
            DatumBlock::new(bcx, Datum::new(val, const_ty, LvalueExpr(lval)))
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
                debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);
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
                        expr.node));
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
        ast::ExprPath(..) => {
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
                             tcx.mk_struct(tcx.lookup_adt_def(did),
                                           tcx.mk_substs(substs)),
                             dest)
            } else {
                tcx.sess.span_bug(expr.span,
                                  "No lang item for ranges (how did we get this far?)")
            }
        }
        ast::ExprTup(ref args) => {
            let numbered_fields: Vec<(usize, &ast::Expr)> =
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
            let substs = match expr_ty(bcx, expr).sty {
                ty::TyClosure(_, ref substs) => substs,
                ref t =>
                    bcx.tcx().sess.span_bug(
                        expr.span,
                        &format!("closure expr without closure type: {:?}", t)),
            };
            closure::trans_closure_expr(dest, decl, body, expr.id, substs).unwrap_or(bcx)
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
                                Some((rhs_datum, rhs.id)), Some(dest),
                                !ast_util::is_by_value_binop(op.node)).bcx
        }
        ast::ExprUnary(op, ref subexpr) => {
            // if not overloaded, would be RvalueDatumExpr
            let arg = unpack_datum!(bcx, trans(bcx, &**subexpr));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id),
                                arg, None, Some(dest), !ast_util::is_by_value_unop(op)).bcx
        }
        ast::ExprIndex(ref base, ref idx) => {
            // if not overloaded, would be RvalueDatumExpr
            let base = unpack_datum!(bcx, trans(bcx, &**base));
            let idx_datum = unpack_datum!(bcx, trans(bcx, &**idx));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id), base,
                                Some((idx_datum, idx.id)), Some(dest), true).bcx
        }
        ast::ExprCast(..) => {
            // Trait casts used to come this way, now they should be coercions.
            bcx.tcx().sess.span_bug(expr.span, "DPS expr_cast (residual trait cast?)")
        }
        ast::ExprAssignOp(op, ref dst, ref src) => {
            trans_assign_op(bcx, expr, op, &**dst, &**src)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                &format!("trans_rvalue_dps_unadjusted reached fall-through \
                         case: {:?}",
                        expr.node));
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
            let variant = bcx.tcx().lookup_adt_def(tid).variant_with_id(vid);
            if let ty::VariantKind::Tuple = variant.kind() {
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
                adt::trans_set_discr(bcx, &*repr, lldest, variant.disr_val);
                return bcx;
            }
        }
        def::DefStruct(_) => {
            let ty = expr_ty(bcx, ref_expr);
            match ty.sty {
                ty::TyStruct(def, _) if def.has_dtor() => {
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
                def, bcx.node_id_to_string(ref_expr.id)));
        }
    }
}

pub fn trans_def_fn_unadjusted<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                         ref_expr: &ast::Expr,
                                         def: def::Def,
                                         param_substs: &'tcx Substs<'tcx>)
                                         -> Datum<'tcx, Rvalue> {
    let _icx = push_ctxt("trans_def_datum_unadjusted");

    match def {
        def::DefFn(did, _) |
        def::DefStruct(did) | def::DefVariant(_, did, _) => {
            callee::trans_fn_ref(ccx, did, ExprId(ref_expr.id), param_substs)
        }
        def::DefMethod(method_did) => {
            match ccx.tcx().impl_or_trait_item(method_did).container() {
                ty::ImplContainer(_) => {
                    callee::trans_fn_ref(ccx, method_did,
                                         ExprId(ref_expr.id),
                                         param_substs)
                }
                ty::TraitContainer(trait_did) => {
                    meth::trans_static_method_callee(ccx, method_did,
                                                     trait_did, ref_expr.id,
                                                     param_substs)
                }
            }
        }
        _ => {
            ccx.tcx().sess.span_bug(ref_expr.span, &format!(
                    "trans_def_fn_unadjusted invoked on: {:?} for {:?}",
                    def,
                    ref_expr));
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
            let lval = Lvalue::new_with_hint("expr::trans_local_var (upvar)",
                                             bcx, nid, HintKind::ZeroAndMaintain);
            match bcx.fcx.llupvars.borrow().get(&nid) {
                Some(&val) => Datum::new(val, local_ty, lval),
                None => {
                    bcx.sess().bug(&format!(
                        "trans_local_var: no llval for upvar {} found",
                        nid));
                }
            }
        }
        def::DefLocal(nid) => {
            let datum = match bcx.fcx.lllocals.borrow().get(&nid) {
                Some(&v) => v,
                None => {
                    bcx.sess().bug(&format!(
                        "trans_local_var: no datum for local/arg {} found",
                        nid));
                }
            };
            debug!("take_local(nid={}, v={}, ty={})",
                   nid, bcx.val_to_string(datum.val), datum.ty);
            datum
        }
        _ => {
            bcx.sess().unimpl(&format!(
                "unsupported def type in trans_local_var: {:?}",
                def));
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
    let vinfo = VariantInfo::of_node(tcx, ty, expr_id);

    let mut need_base = vec![true; vinfo.fields.len()];

    let numbered_fields = fields.iter().map(|field| {
        let pos = vinfo.field_index(field.ident.node.name);
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
                tcx.sess.span_bug(expr_span, "missing fields and no base expr")
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
    expr: &'a ast::Expr,
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
                                 discr: ty::Disr,
                                 fields: &[(usize, &ast::Expr)],
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
            let block_datum = trans(bcx, &**e);
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
            let datum = unpack_datum!(bcx, trans(bcx, &**e));
            (i, datum)
        }).collect();

        debug_location.apply(bcx.fcx);

        // Second, trans the base to the dest.
        assert_eq!(discr, 0);

        match expr_kind(bcx.tcx(), &*base.expr) {
            ExprKind::RvalueDps | ExprKind::RvalueDatum if !bcx.fcx.type_needs_drop(ty) => {
                bcx = trans_into(bcx, &*base.expr, SaveIn(addr));
            },
            ExprKind::RvalueStmt => {
                bcx.tcx().sess.bug("unexpected expr kind for struct base expr")
            }
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

        // Finally, move scratch field values into actual field locations
        for (i, datum) in scratch_vals {
            let dest = adt::trans_field_ptr(bcx, &*repr, addr, discr, i);
            bcx = datum.store_to(bcx, dest);
        }
    } else {
        // No base means we can write all fields directly in place.
        for &(i, ref e) in fields {
            let dest = adt::trans_field_ptr(bcx, &*repr, addr, discr, i);
            let e_ty = expr_ty_adjusted(bcx, &**e);
            bcx = trans_into(bcx, &**e, SaveIn(dest));
            let scope = cleanup::CustomScope(custom_cleanup_scope);
            fcx.schedule_lifetime_end(scope, dest);
            // FIXME: nonzeroing move should generalize to fields
            fcx.schedule_drop_mem(scope, dest, e_ty, None);
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
    assert!(op == ast::UnDeref || !ccx.tcx().is_method_call(expr.id));

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
    let dest_ty = bcx.tcx().mk_imm_ref(bcx.tcx().mk_region(ty::ReStatic), lval.ty);
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
    if !type_is_sized(bcx.tcx(), sub_datum.ty) {
        // DST lvalue, close to a fat pointer
        ref_fat_ptr(bcx, sub_datum)
    } else {
        // Sized value, ref to a thin pointer
        let ty = expr_ty(bcx, expr);
        immediate_rvalue_bcx(bcx, sub_datum.val, ty).to_expr_datumblock()
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
    let is_simd = lhs_t.is_simd();
    let intype = if is_simd {
        lhs_t.simd_type(tcx)
    } else {
        lhs_t
    };
    let is_float = intype.is_fp();
    let is_signed = intype.is_signed();
    let info = expr_info(binop_expr);

    let binop_debug_loc = binop_expr.debug_loc();

    let mut bcx = bcx;
    let val = match op.node {
      ast::BiAdd => {
        if is_float {
            FAdd(bcx, lhs, rhs, binop_debug_loc)
        } else if is_simd {
            Add(bcx, lhs, rhs, binop_debug_loc)
        } else {
            let (newbcx, res) = with_overflow_check(
                bcx, OverflowOp::Add, info, lhs_t, lhs, rhs, binop_debug_loc);
            bcx = newbcx;
            res
        }
      }
      ast::BiSub => {
        if is_float {
            FSub(bcx, lhs, rhs, binop_debug_loc)
        } else if is_simd {
            Sub(bcx, lhs, rhs, binop_debug_loc)
        } else {
            let (newbcx, res) = with_overflow_check(
                bcx, OverflowOp::Sub, info, lhs_t, lhs, rhs, binop_debug_loc);
            bcx = newbcx;
            res
        }
      }
      ast::BiMul => {
        if is_float {
            FMul(bcx, lhs, rhs, binop_debug_loc)
        } else if is_simd {
            Mul(bcx, lhs, rhs, binop_debug_loc)
        } else {
            let (newbcx, res) = with_overflow_check(
                bcx, OverflowOp::Mul, info, lhs_t, lhs, rhs, binop_debug_loc);
            bcx = newbcx;
            res
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
                let llfn = declare::declare_cfn(bcx.ccx(), "fmod", fty,
                                                tcx.types.f64);
                if lhs_t == tcx.types.f32 {
                    let lhs = FPExt(bcx, lhs, f64t);
                    let rhs = FPExt(bcx, rhs, f64t);
                    let res = Call(bcx, llfn, &[lhs, rhs], None, binop_debug_loc);
                    FPTrunc(bcx, res, Type::f32(bcx.ccx()))
                } else {
                    Call(bcx, llfn, &[lhs, rhs], None, binop_debug_loc)
                }
            } else {
                FRem(bcx, lhs, rhs, binop_debug_loc)
            }
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
      ast::BiShl => {
          let (newbcx, res) = with_overflow_check(
              bcx, OverflowOp::Shl, info, lhs_t, lhs, rhs, binop_debug_loc);
          bcx = newbcx;
          res
      }
      ast::BiShr => {
          let (newbcx, res) = with_overflow_check(
              bcx, OverflowOp::Shr, info, lhs_t, lhs, rhs, binop_debug_loc);
          bcx = newbcx;
          res
      }
      ast::BiEq | ast::BiNe | ast::BiLt | ast::BiGe | ast::BiLe | ast::BiGt => {
        if is_simd {
            base::compare_simd_types(bcx, lhs, rhs, intype, val_ty(lhs), op.node, binop_debug_loc)
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
    assert!(!ccx.tcx().is_method_call(expr.id));

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
                                   rhs: Option<(Datum<'tcx, Expr>, ast::NodeId)>,
                                   dest: Option<Dest>,
                                   autoref: bool)
                                   -> Result<'blk, 'tcx> {
    callee::trans_call_inner(bcx,
                             expr.debug_loc(),
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
    debug!("trans_overloaded_call {}", expr.id);
    let method_call = MethodCall::expr(expr.id);
    let mut all_args = vec!(callee);
    all_args.extend(args.iter().map(|e| &**e));
    unpack_result!(bcx,
                   callee::trans_call_inner(bcx,
                                            expr.debug_loc(),
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

pub fn cast_is_noop<'tcx>(tcx: &ty::ctxt<'tcx>,
                          expr: &ast::Expr,
                          t_in: Ty<'tcx>,
                          t_out: Ty<'tcx>)
                          -> bool {
    if let Some(&CastKind::CoercionCast) = tcx.cast_kinds.borrow().get(&expr.id) {
        return true;
    }

    match (t_in.builtin_deref(true), t_out.builtin_deref(true)) {
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
                              expr: &ast::Expr,
                              id: ast::NodeId)
                              -> DatumBlock<'blk, 'tcx, Expr>
{
    use middle::cast::CastTy::*;
    use middle::cast::IntTy::*;

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
    let mut ll_t_in = type_of::arg_type_of(ccx, t_in);
    let ll_t_out = type_of::arg_type_of(ccx, t_out);
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
        let discr = adt::trans_get_discr(bcx, &*repr, llexpr_ptr, Some(Type::i64(ccx)));
        ll_t_in = val_ty(discr);
        (discr, adt::is_discr_signed(&*repr))
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

        _ => ccx.sess().span_bug(expr.span,
                                  &format!("translating unsupported cast: \
                                            {:?} -> {:?}",
                                           t_in,
                                           t_out)
                                 )
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

    debug!("trans_assign_op(expr={:?})", expr);

    // User-defined operator methods cannot be used with `+=` etc right now
    assert!(!bcx.tcx().is_method_call(expr.id));

    // Evaluate LHS (destination), which should be an lvalue
    let dst_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, dst, "assign_op"));
    assert!(!bcx.fcx.type_needs_drop(dst_datum.ty));
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
    let ptr_ty = bcx.tcx().mk_imm_ref(bcx.tcx().mk_region(ty::ReStatic), referent_ty);

    // Get the pointer.
    let llref = lv_datum.to_llref();

    // Construct the resulting datum, using what was the "by ref"
    // ValueRef of type `referent_ty` to be the "by value" ValueRef
    // of type `&referent_ty`.
    // Pointers to DST types are non-immediate, and therefore still use ByRef.
    let kind  = if type_is_sized(bcx.tcx(), referent_ty) { ByValue } else { ByRef };
    DatumBlock::new(bcx, Datum::new(llref, ptr_ty, RvalueExpr(Rvalue::new(kind))))
}

fn deref_multiple<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              expr: &ast::Expr,
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
                          expr: &ast::Expr,
                          datum: Datum<'tcx, Expr>,
                          method_call: MethodCall)
                          -> DatumBlock<'blk, 'tcx, Expr> {
    let ccx = bcx.ccx();

    debug!("deref_once(expr={:?}, datum={}, method_call={:?})",
           expr,
           datum.to_string(ccx),
           method_call);

    let mut bcx = bcx;

    // Check for overloaded deref.
    let method_ty = ccx.tcx()
                       .tables
                       .borrow()
                       .method_map
                       .get(&method_call).map(|method| method.ty);

    let datum = match method_ty {
        Some(method_ty) => {
            let method_ty = monomorphize_type(bcx, method_ty);

            // Overloaded. Evaluate `trans_overloaded_op`, which will
            // invoke the user's deref() method, which basically
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

            unpack_result!(bcx, trans_overloaded_op(bcx, expr, method_call,
                                                    datum, None, Some(SaveIn(scratch.val)),
                                                    false));
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
            bcx.tcx().sess.span_bug(
                expr.span,
                &format!("deref invoked on expr of invalid type {:?}",
                        datum.ty));
        }
    };

    debug!("deref_once(expr={}, method_call={:?}, result={})",
           expr.id, method_call, r.datum.to_string(ccx));

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
    fn to_intrinsic_name(&self, tcx: &ty::ctxt, ty: Ty) -> &'static str {
        use syntax::ast::IntTy::*;
        use syntax::ast::UintTy::*;
        use middle::ty::{TyInt, TyUint};

        let new_sty = match ty.sty {
            TyInt(TyIs) => match &tcx.sess.target.target.target_pointer_width[..] {
                "32" => TyInt(TyI32),
                "64" => TyInt(TyI64),
                _ => panic!("unsupported target word size")
            },
            TyUint(TyUs) => match &tcx.sess.target.target.target_pointer_width[..] {
                "32" => TyUint(TyU32),
                "64" => TyUint(TyU64),
                _ => panic!("unsupported target word size")
            },
            ref t @ TyUint(_) | ref t @ TyInt(_) => t.clone(),
            _ => panic!("tried to get overflow intrinsic for {:?} applied to non-int type",
                        *self)
        };

        match *self {
            OverflowOpViaIntrinsic::Add => match new_sty {
                TyInt(TyI8) => "llvm.sadd.with.overflow.i8",
                TyInt(TyI16) => "llvm.sadd.with.overflow.i16",
                TyInt(TyI32) => "llvm.sadd.with.overflow.i32",
                TyInt(TyI64) => "llvm.sadd.with.overflow.i64",

                TyUint(TyU8) => "llvm.uadd.with.overflow.i8",
                TyUint(TyU16) => "llvm.uadd.with.overflow.i16",
                TyUint(TyU32) => "llvm.uadd.with.overflow.i32",
                TyUint(TyU64) => "llvm.uadd.with.overflow.i64",

                _ => unreachable!(),
            },
            OverflowOpViaIntrinsic::Sub => match new_sty {
                TyInt(TyI8) => "llvm.ssub.with.overflow.i8",
                TyInt(TyI16) => "llvm.ssub.with.overflow.i16",
                TyInt(TyI32) => "llvm.ssub.with.overflow.i32",
                TyInt(TyI64) => "llvm.ssub.with.overflow.i64",

                TyUint(TyU8) => "llvm.usub.with.overflow.i8",
                TyUint(TyU16) => "llvm.usub.with.overflow.i16",
                TyUint(TyU32) => "llvm.usub.with.overflow.i32",
                TyUint(TyU64) => "llvm.usub.with.overflow.i64",

                _ => unreachable!(),
            },
            OverflowOpViaIntrinsic::Mul => match new_sty {
                TyInt(TyI8) => "llvm.smul.with.overflow.i8",
                TyInt(TyI16) => "llvm.smul.with.overflow.i16",
                TyInt(TyI32) => "llvm.smul.with.overflow.i32",
                TyInt(TyI64) => "llvm.smul.with.overflow.i64",

                TyUint(TyU8) => "llvm.umul.with.overflow.i8",
                TyUint(TyU16) => "llvm.umul.with.overflow.i16",
                TyUint(TyU32) => "llvm.umul.with.overflow.i32",
                TyUint(TyU64) => "llvm.umul.with.overflow.i64",

                _ => unreachable!(),
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

        let val = Call(bcx, llfn, &[lhs, rhs], None, binop_debug_loc);
        let result = ExtractValue(bcx, val, 0); // iN operation result
        let overflow = ExtractValue(bcx, val, 1); // i1 "did it overflow?"

        let cond = ICmp(bcx, llvm::IntEQ, overflow, C_integral(Type::i1(bcx.ccx()), 1, false),
                        binop_debug_loc);

        let expect = bcx.ccx().get_intrinsic(&"llvm.expect.i1");
        Call(bcx, expect, &[cond, C_integral(Type::i1(bcx.ccx()), 0, false)],
             None, binop_debug_loc);

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
        // both be fed into a and-binop)
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

fn shift_mask_val<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              llty: Type,
                              mask_llty: Type,
                              invert: bool) -> ValueRef {
    let kind = llty.kind();
    match kind {
        TypeKind::Integer => {
            // i8/u8 can shift by at most 7, i16/u16 by at most 15, etc.
            let val = llty.int_width() - 1;
            if invert {
                C_integral(mask_llty, !val, true)
            } else {
                C_integral(mask_llty, val, false)
            }
        },
        TypeKind::Vector => {
            let mask = shift_mask_val(bcx, llty.element_type(), mask_llty.element_type(), invert);
            VectorSplat(bcx, mask_llty.vector_length(), mask)
        },
        _ => panic!("shift_mask_val: expected Integer or Vector, found {:?}", kind),
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
        _ => panic!("build_nonzero_check: expected Integer or Vector, found {:?}", kind),
    }
}

// To avoid UB from LLVM, these two functions mask RHS with an
// appropriate mask unconditionally (i.e. the fallback behavior for
// all shifts). For 32- and 64-bit types, this matches the semantics
// of Java. (See related discussion on #1877 and #10183.)

fn build_unchecked_lshift<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      lhs: ValueRef,
                                      rhs: ValueRef,
                                      binop_debug_loc: DebugLoc) -> ValueRef {
    let rhs = base::cast_shift_expr_rhs(bcx, ast::BinOp_::BiShl, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bcx, rhs, binop_debug_loc);
    Shl(bcx, lhs, rhs, binop_debug_loc)
}

fn build_unchecked_rshift<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      lhs_t: Ty<'tcx>,
                                      lhs: ValueRef,
                                      rhs: ValueRef,
                                      binop_debug_loc: DebugLoc) -> ValueRef {
    let rhs = base::cast_shift_expr_rhs(bcx, ast::BinOp_::BiShr, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bcx, rhs, binop_debug_loc);
    let tcx = bcx.tcx();
    let is_simd = lhs_t.is_simd();
    let intype = if is_simd {
        lhs_t.simd_type(tcx)
    } else {
        lhs_t
    };
    let is_signed = intype.is_signed();
    if is_signed {
        AShr(bcx, lhs, rhs, binop_debug_loc)
    } else {
        LShr(bcx, lhs, rhs, binop_debug_loc)
    }
}

fn shift_mask_rhs<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              rhs: ValueRef,
                              debug_loc: DebugLoc) -> ValueRef {
    let rhs_llty = val_ty(rhs);
    And(bcx, rhs, shift_mask_val(bcx, rhs_llty, rhs_llty, false), debug_loc)
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

fn expr_kind(tcx: &ty::ctxt, expr: &ast::Expr) -> ExprKind {
    if tcx.is_method_call(expr.id) {
        // Overloaded operations are generally calls, and hence they are
        // generated via DPS, but there are a few exceptions:
        return match expr.node {
            // `a += b` has a unit result.
            ast::ExprAssignOp(..) => ExprKind::RvalueStmt,

            // the deref method invoked for `*a` always yields an `&T`
            ast::ExprUnary(ast::UnDeref, _) => ExprKind::Lvalue,

            // the index method invoked for `a[i]` always yields an `&T`
            ast::ExprIndex(..) => ExprKind::Lvalue,

            // in the general case, result could be any type, use DPS
            _ => ExprKind::RvalueDps
        };
    }

    match expr.node {
        ast::ExprPath(..) => {
            match tcx.resolve_expr(expr) {
                def::DefStruct(_) | def::DefVariant(..) => {
                    if let ty::TyBareFn(..) = tcx.node_id_to_type(expr.id).sty {
                        // ctor function
                        ExprKind::RvalueDatum
                    } else {
                        ExprKind::RvalueDps
                    }
                }

                // Special case: A unit like struct's constructor must be called without () at the
                // end (like `UnitStruct`) which means this is an ExprPath to a DefFn. But in case
                // of unit structs this is should not be interpreted as function pointer but as
                // call to the constructor.
                def::DefFn(_, true) => ExprKind::RvalueDps,

                // Fn pointers are just scalar values.
                def::DefFn(..) | def::DefMethod(..) => ExprKind::RvalueDatum,

                // Note: there is actually a good case to be made that
                // DefArg's, particularly those of immediate type, ought to
                // considered rvalues.
                def::DefStatic(..) |
                def::DefUpvar(..) |
                def::DefLocal(..) => ExprKind::Lvalue,

                def::DefConst(..) |
                def::DefAssociatedConst(..) => ExprKind::RvalueDatum,

                def => {
                    tcx.sess.span_bug(
                        expr.span,
                        &format!("uncategorized def for expr {}: {:?}",
                                expr.id,
                                def));
                }
            }
        }

        ast::ExprUnary(ast::UnDeref, _) |
        ast::ExprField(..) |
        ast::ExprTupField(..) |
        ast::ExprIndex(..) => {
            ExprKind::Lvalue
        }

        ast::ExprCall(..) |
        ast::ExprMethodCall(..) |
        ast::ExprStruct(..) |
        ast::ExprRange(..) |
        ast::ExprTup(..) |
        ast::ExprIf(..) |
        ast::ExprMatch(..) |
        ast::ExprClosure(..) |
        ast::ExprBlock(..) |
        ast::ExprRepeat(..) |
        ast::ExprVec(..) => {
            ExprKind::RvalueDps
        }

        ast::ExprIfLet(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprIfLet");
        }
        ast::ExprWhileLet(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprWhileLet");
        }

        ast::ExprForLoop(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprForLoop");
        }

        ast::ExprLit(ref lit) if ast_util::lit_is_str(&**lit) => {
            ExprKind::RvalueDps
        }

        ast::ExprBreak(..) |
        ast::ExprAgain(..) |
        ast::ExprRet(..) |
        ast::ExprWhile(..) |
        ast::ExprLoop(..) |
        ast::ExprAssign(..) |
        ast::ExprInlineAsm(..) |
        ast::ExprAssignOp(..) => {
            ExprKind::RvalueStmt
        }

        ast::ExprLit(_) | // Note: LitStr is carved out above
        ast::ExprUnary(..) |
        ast::ExprBox(None, _) |
        ast::ExprAddrOf(..) |
        ast::ExprBinary(..) |
        ast::ExprCast(..) => {
            ExprKind::RvalueDatum
        }

        ast::ExprBox(Some(ref place), _) => {
            // Special case `Box<T>` for now:
            let def_id = match tcx.def_map.borrow().get(&place.id) {
                Some(def) => def.def_id(),
                None => panic!("no def for place"),
            };
            if tcx.lang_items.exchange_heap() == Some(def_id) {
                ExprKind::RvalueDatum
            } else {
                ExprKind::RvalueDps
            }
        }

        ast::ExprParen(ref e) => expr_kind(tcx, &**e),

        ast::ExprMac(..) => {
            tcx.sess.span_bug(
                expr.span,
                "macro expression remains after expansion");
        }
    }
}
