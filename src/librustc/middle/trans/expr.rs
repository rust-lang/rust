// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * # Translation of Expressions
 *
 * Public entry points:
 *
 * - `trans_into(bcx, expr, dest) -> bcx`: evaluates an expression,
 *   storing the result into `dest`. This is the preferred form, if you
 *   can manage it.
 *
 * - `trans(bcx, expr) -> DatumBlock`: evaluates an expression, yielding
 *   `Datum` with the result. You can then store the datum, inspect
 *   the value, etc. This may introduce temporaries if the datum is a
 *   structural type.
 *
 * - `trans_to_lvalue(bcx, expr, "...") -> DatumBlock`: evaluates an
 *   expression and ensures that the result has a cleanup associated with it,
 *   creating a temporary stack slot if necessary.
 *
 * - `trans_local_var -> Datum`: looks up a local variable or upvar.
 *
 * See doc.rs for more comments.
 */

#![allow(non_camel_case_types)]

use back::abi;
use lib::llvm::{ValueRef, llvm};
use lib;
use metadata::csearch;
use middle::trans::_match;
use middle::trans::adt;
use middle::trans::asm;
use middle::trans::base::*;
use middle::trans::base;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::closure;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::controlflow;
use middle::trans::datum::*;
use middle::trans::debuginfo;
use middle::trans::glue;
use middle::trans::machine;
use middle::trans::meth;
use middle::trans::inline;
use middle::trans::tvec;
use middle::trans::type_of;
use middle::trans::write_guard;
use middle::ty::struct_fields;
use middle::ty::{AutoBorrowObj, AutoDerefRef, AutoAddEnv, AutoObject, AutoUnsafe};
use middle::ty::{AutoPtr, AutoBorrowVec, AutoBorrowVecRef, AutoBorrowFn};
use middle::ty;
use middle::typeck::MethodCall;
use util::common::indenter;
use util::ppaux::Repr;
use util::nodemap::NodeMap;
use middle::trans::machine::{llsize_of, llsize_of_alloc};
use middle::trans::type_::Type;

use std::slice;
use syntax::ast;
use syntax::codemap;
use syntax::print::pprust::{expr_to_str};

// Destinations

// These are passed around by the code generating functions to track the
// destination of a computation's value.

#[deriving(Eq)]
pub enum Dest {
    SaveIn(ValueRef),
    Ignore,
}

impl Dest {
    pub fn to_str(&self, ccx: &CrateContext) -> ~str {
        match *self {
            SaveIn(v) => format!("SaveIn({})", ccx.tn.val_to_str(v)),
            Ignore => ~"Ignore"
        }
    }
}

pub fn trans_into<'a>(bcx: &'a Block<'a>,
                      expr: &ast::Expr,
                      dest: Dest)
                      -> &'a Block<'a> {
    /*!
     * This function is equivalent to `trans(bcx, expr).store_to_dest(dest)`
     * but it may generate better optimized LLVM code.
     */

    let mut bcx = bcx;

    if bcx.tcx().adjustments.borrow().contains_key(&expr.id) {
        // use trans, which may be less efficient but
        // which will perform the adjustments:
        let datum = unpack_datum!(bcx, trans(bcx, expr));
        return datum.store_to_dest(bcx, dest, expr.id)
    }

    debug!("trans_into() expr={}", expr.repr(bcx.tcx()));
    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    bcx.fcx.push_ast_cleanup_scope(expr.id);

    let kind = ty::expr_kind(bcx.tcx(), bcx.ccx().maps.method_map, expr);
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

pub fn trans<'a>(bcx: &'a Block<'a>,
                 expr: &ast::Expr)
                 -> DatumBlock<'a, Expr> {
    /*!
     * Translates an expression, returning a datum (and new block)
     * encapsulating the result. When possible, it is preferred to
     * use `trans_into`, as that may avoid creating a temporary on
     * the stack.
     */

    debug!("trans(expr={})", bcx.expr_to_str(expr));

    let mut bcx = bcx;
    let fcx = bcx.fcx;

    fcx.push_ast_cleanup_scope(expr.id);
    let datum = unpack_datum!(bcx, trans_unadjusted(bcx, expr));
    let datum = unpack_datum!(bcx, apply_adjustments(bcx, expr, datum));
    bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, expr.id);
    return DatumBlock(bcx, datum);
}

fn apply_adjustments<'a>(bcx: &'a Block<'a>,
                         expr: &ast::Expr,
                         datum: Datum<Expr>)
                         -> DatumBlock<'a, Expr> {
    /*!
     * Helper for trans that apply adjustments from `expr` to `datum`,
     * which should be the unadjusted translation of `expr`.
     */

    let mut bcx = bcx;
    let mut datum = datum;
    let adjustment = match bcx.tcx().adjustments.borrow().find_copy(&expr.id) {
        None => {
            return DatumBlock(bcx, datum);
        }
        Some(adj) => { adj }
    };
    debug!("unadjusted datum for expr {}: {}",
           expr.id, datum.to_str(bcx.ccx()));
    match *adjustment {
        AutoAddEnv(..) => {
            datum = unpack_datum!(bcx, add_env(bcx, expr, datum));
        }
        AutoDerefRef(ref adj) => {
            if adj.autoderefs > 0 {
                datum = unpack_datum!(
                    bcx, deref_multiple(bcx, expr, datum, adj.autoderefs));
            }

            datum = match adj.autoref {
                None => {
                    datum
                }
                Some(AutoUnsafe(..)) | // region + unsafe ptrs have same repr
                Some(AutoPtr(..)) => {
                    unpack_datum!(bcx, auto_ref(bcx, datum, expr))
                }
                Some(AutoBorrowVec(..)) => {
                    unpack_datum!(bcx, auto_slice(bcx, expr, datum))
                }
                Some(AutoBorrowVecRef(..)) => {
                    unpack_datum!(bcx, auto_slice_and_ref(bcx, expr, datum))
                }
                Some(AutoBorrowFn(..)) => {
                    let adjusted_ty = ty::adjust_ty(bcx.tcx(), expr.span, expr.id, datum.ty,
                                                    Some(adjustment), |method_call| {
                        bcx.ccx().maps.method_map.borrow()
                           .find(&method_call).map(|method| method.ty)
                    });
                    unpack_datum!(bcx, auto_borrow_fn(bcx, adjusted_ty, datum))
                }
                Some(AutoBorrowObj(..)) => {
                    unpack_datum!(bcx, auto_borrow_obj(bcx, expr, datum))
                }
            };
        }
        AutoObject(..) => {
            let adjusted_ty = ty::expr_ty_adjusted(bcx.tcx(), expr,
                                                   &*bcx.ccx().maps.method_map.borrow());
            let scratch = rvalue_scratch_datum(bcx, adjusted_ty, "__adjust");
            bcx = meth::trans_trait_cast(
                bcx, datum, expr.id, SaveIn(scratch.val));
            datum = scratch.to_expr_datum();
        }
    }
    debug!("after adjustments, datum={}", datum.to_str(bcx.ccx()));
    return DatumBlock {bcx: bcx, datum: datum};

    fn auto_borrow_fn<'a>(
                      bcx: &'a Block<'a>,
                      adjusted_ty: ty::t,
                      datum: Datum<Expr>)
                      -> DatumBlock<'a, Expr> {
        // Currently, all closure types are represented precisely the
        // same, so no runtime adjustment is required, but we still
        // must patchup the type.
        DatumBlock {bcx: bcx,
                    datum: Datum {val: datum.val,
                                  ty: adjusted_ty,
                                  kind: datum.kind}}
    }

    fn auto_slice<'a>(
                  bcx: &'a Block<'a>,
                  expr: &ast::Expr,
                  datum: Datum<Expr>)
                  -> DatumBlock<'a, Expr> {
        // This is not the most efficient thing possible; since slices
        // are two words it'd be better if this were compiled in
        // 'dest' mode, but I can't find a nice way to structure the
        // code and keep it DRY that accommodates that use case at the
        // moment.

        let mut bcx = bcx;
        let tcx = bcx.tcx();
        let unit_ty = ty::sequence_element_type(tcx, datum.ty);

        // Arrange cleanup, if not already done. This is needed in
        // case we are auto-slicing an owned vector or some such.
        let datum = unpack_datum!(
            bcx, datum.to_lvalue_datum(bcx, "auto_slice", expr.id));

        let (base, len) = datum.get_vec_base_and_len(bcx);

        // this type may have a different region/mutability than the
        // real one, but it will have the same runtime representation
        let slice_ty = ty::mk_vec(tcx,
                                  ty::mt { ty: unit_ty, mutbl: ast::MutImmutable },
                                  ty::vstore_slice(ty::ReStatic));

        let scratch = rvalue_scratch_datum(bcx, slice_ty, "__adjust");
        Store(bcx, base, GEPi(bcx, scratch.val, [0u, abi::slice_elt_base]));
        Store(bcx, len, GEPi(bcx, scratch.val, [0u, abi::slice_elt_len]));
        DatumBlock(bcx, scratch.to_expr_datum())
    }

    fn add_env<'a>(bcx: &'a Block<'a>,
                   expr: &ast::Expr,
                   datum: Datum<Expr>)
                   -> DatumBlock<'a, Expr> {
        // This is not the most efficient thing possible; since closures
        // are two words it'd be better if this were compiled in
        // 'dest' mode, but I can't find a nice way to structure the
        // code and keep it DRY that accommodates that use case at the
        // moment.

        let closure_ty = expr_ty_adjusted(bcx, expr);
        let fn_ptr = datum.to_llscalarish(bcx);
        let def = ty::resolve_expr(bcx.tcx(), expr);
        closure::make_closure_from_bare_fn(bcx, closure_ty, def, fn_ptr)
    }

    fn auto_slice_and_ref<'a>(
                          bcx: &'a Block<'a>,
                          expr: &ast::Expr,
                          datum: Datum<Expr>)
                          -> DatumBlock<'a, Expr> {
        let DatumBlock { bcx, datum } = auto_slice(bcx, expr, datum);
        auto_ref(bcx, datum, expr)
    }

    fn auto_borrow_obj<'a>(bcx: &'a Block<'a>,
                           expr: &ast::Expr,
                           source_datum: Datum<Expr>)
                           -> DatumBlock<'a, Expr> {
        let tcx = bcx.tcx();
        let target_obj_ty = expr_ty_adjusted(bcx, expr);
        debug!("auto_borrow_obj(target={})", target_obj_ty.repr(tcx));

        let mut datum = source_datum.to_expr_datum();
        datum.ty = target_obj_ty;
        DatumBlock(bcx, datum)
    }
}

pub fn trans_to_lvalue<'a>(bcx: &'a Block<'a>,
                           expr: &ast::Expr,
                           name: &str)
                           -> DatumBlock<'a, Lvalue> {
    /*!
     * Translates an expression in "lvalue" mode -- meaning that it
     * returns a reference to the memory that the expr represents.
     *
     * If this expression is an rvalue, this implies introducing a
     * temporary.  In other words, something like `x().f` is
     * translated into roughly the equivalent of
     *
     *   { tmp = x(); tmp.f }
     */

    let mut bcx = bcx;
    let datum = unpack_datum!(bcx, trans(bcx, expr));
    return datum.to_lvalue_datum(bcx, name, expr.id);
}

fn trans_unadjusted<'a>(bcx: &'a Block<'a>,
                        expr: &ast::Expr)
                        -> DatumBlock<'a, Expr> {
    /*!
     * A version of `trans` that ignores adjustments. You almost
     * certainly do not want to call this directly.
     */

    let mut bcx = bcx;

    debug!("trans_unadjusted(expr={})", bcx.expr_to_str(expr));
    let _indenter = indenter();

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    return match ty::expr_kind(bcx.tcx(), bcx.ccx().maps.method_map, expr) {
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

                DatumBlock(bcx, scratch.to_expr_datum())
            }
        }
    };

    fn nil<'a>(bcx: &'a Block<'a>, ty: ty::t) -> DatumBlock<'a, Expr> {
        let llval = C_undef(type_of::type_of(bcx.ccx(), ty));
        let datum = immediate_rvalue(llval, ty);
        DatumBlock(bcx, datum.to_expr_datum())
    }
}

fn trans_datum_unadjusted<'a>(bcx: &'a Block<'a>,
                              expr: &ast::Expr)
                              -> DatumBlock<'a, Expr> {
    let mut bcx = bcx;
    let fcx = bcx.fcx;
    let _icx = push_ctxt("trans_datum_unadjusted");

    match expr.node {
        ast::ExprParen(e) => {
            trans(bcx, e)
        }
        ast::ExprPath(_) => {
            trans_def(bcx, expr, bcx.def(expr.id))
        }
        ast::ExprField(base, ident, _) => {
            trans_rec_field(bcx, base, ident)
        }
        ast::ExprIndex(base, idx) => {
            trans_index(bcx, expr, base, idx)
        }
        ast::ExprVstore(contents, ast::ExprVstoreUniq) => {
            fcx.push_ast_cleanup_scope(contents.id);
            let datum = unpack_datum!(
                bcx, tvec::trans_uniq_vstore(bcx, expr, contents));
            bcx = fcx.pop_and_trans_ast_cleanup_scope(bcx, contents.id);
            DatumBlock(bcx, datum)
        }
        ast::ExprBox(_, contents) => {
            // Special case for `~T`. (The other case, for GC, is handled in
            // `trans_rvalue_dps_unadjusted`.)
            let box_ty = expr_ty(bcx, expr);
            let contents_ty = expr_ty(bcx, contents);
            let heap = heap_exchange;
            return trans_boxed_expr(bcx, box_ty, contents, contents_ty, heap)
        }
        ast::ExprLit(lit) => trans_immediate_lit(bcx, expr, (*lit).clone()),
        ast::ExprBinary(op, lhs, rhs) => {
            trans_binary(bcx, expr, op, lhs, rhs)
        }
        ast::ExprUnary(op, x) => {
            trans_unary(bcx, expr, op, x)
        }
        ast::ExprAddrOf(_, x) => {
            trans_addr_of(bcx, expr, x)
        }
        ast::ExprCast(val, _) => {
            // Datum output mode means this is a scalar cast:
            trans_imm_cast(bcx, val, expr.id)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                format!("trans_rvalue_datum_unadjusted reached \
                      fall-through case: {:?}",
                     expr.node));
        }
    }
}

fn trans_rec_field<'a>(bcx: &'a Block<'a>,
                       base: &ast::Expr,
                       field: ast::Ident)
                       -> DatumBlock<'a, Expr> {
    //! Translates `base.field`.

    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rec_field");

    let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, base, "field"));
    let repr = adt::represent_type(bcx.ccx(), base_datum.ty);
    with_field_tys(bcx.tcx(), base_datum.ty, None, |discr, field_tys| {
            let ix = ty::field_idx_strict(bcx.tcx(), field.name, field_tys);
            let d = base_datum.get_element(
                field_tys[ix].mt.ty,
                |srcval| adt::trans_field_ptr(bcx, repr, srcval, discr, ix));
            DatumBlock { datum: d.to_expr_datum(), bcx: bcx }
        })
}

fn trans_index<'a>(bcx: &'a Block<'a>,
                   index_expr: &ast::Expr,
                   base: &ast::Expr,
                   idx: &ast::Expr)
                   -> DatumBlock<'a, Expr> {
    //! Translates `base[idx]`.

    let _icx = push_ctxt("trans_index");
    let ccx = bcx.ccx();
    let mut bcx = bcx;

    let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, base, "index"));

    // Translate index expression and cast to a suitable LLVM integer.
    // Rust is less strict than LLVM in this regard.
    let ix_datum = unpack_datum!(bcx, trans(bcx, idx));
    let ix_val = ix_datum.to_llscalarish(bcx);
    let ix_size = machine::llbitsize_of_real(bcx.ccx(), val_ty(ix_val));
    let int_size = machine::llbitsize_of_real(bcx.ccx(), ccx.int_type);
    let ix_val = {
        if ix_size < int_size {
            if ty::type_is_signed(expr_ty(bcx, idx)) {
                SExt(bcx, ix_val, ccx.int_type)
            } else { ZExt(bcx, ix_val, ccx.int_type) }
        } else if ix_size > int_size {
            Trunc(bcx, ix_val, ccx.int_type)
        } else {
            ix_val
        }
    };

    let vt = tvec::vec_types(bcx, base_datum.ty);
    base::maybe_name_value(bcx.ccx(), vt.llunit_size, "unit_sz");

    let (base, len) = base_datum.get_vec_base_and_len(bcx);

    debug!("trans_index: base {}", bcx.val_to_str(base));
    debug!("trans_index: len {}", bcx.val_to_str(len));

    let bounds_check = ICmp(bcx, lib::llvm::IntUGE, ix_val, len);
    let expect = ccx.intrinsics.get_copy(&("llvm.expect.i1"));
    let expected = Call(bcx, expect, [bounds_check, C_i1(ccx, false)], []);
    let bcx = with_cond(bcx, expected, |bcx| {
            controlflow::trans_fail_bounds_check(bcx, index_expr.span, ix_val, len)
        });
    let elt = InBoundsGEP(bcx, base, [ix_val]);
    let elt = PointerCast(bcx, elt, vt.llunit_ty.ptr_to());
    DatumBlock(bcx, Datum(elt, vt.unit_ty, LvalueExpr))
}

fn trans_def<'a>(bcx: &'a Block<'a>,
                 ref_expr: &ast::Expr,
                 def: ast::Def)
                 -> DatumBlock<'a, Expr>
{
    //! Translates a reference to a path.

    let _icx = push_ctxt("trans_def_lvalue");
    match def {
        ast::DefFn(..) | ast::DefStaticMethod(..) |
        ast::DefStruct(_) | ast::DefVariant(..) => {
            trans_def_fn_unadjusted(bcx, ref_expr, def)
        }
        ast::DefStatic(did, _) => {
            let const_ty = expr_ty(bcx, ref_expr);

            fn get_did(ccx: &CrateContext, did: ast::DefId)
                       -> ast::DefId {
                if did.krate != ast::LOCAL_CRATE {
                    inline::maybe_instantiate_inline(ccx, did)
                } else {
                    did
                }
            }

            fn get_val<'a>(bcx: &'a Block<'a>, did: ast::DefId, const_ty: ty::t)
                       -> ValueRef {
                // For external constants, we don't inline.
                if did.krate == ast::LOCAL_CRATE {
                    // The LLVM global has the type of its initializer,
                    // which may not be equal to the enum's type for
                    // non-C-like enums.
                    let val = base::get_item_val(bcx.ccx(), did.node);
                    let pty = type_of::type_of(bcx.ccx(), const_ty).ptr_to();
                    PointerCast(bcx, val, pty)
                } else {
                    match bcx.ccx().extern_const_values.borrow().find(&did) {
                        None => {}  // Continue.
                        Some(llval) => {
                            return *llval;
                        }
                    }

                    unsafe {
                        let llty = type_of::type_of(bcx.ccx(), const_ty);
                        let symbol = csearch::get_symbol(
                            &bcx.ccx().sess().cstore,
                            did);
                        let llval = symbol.with_c_str(|buf| {
                                llvm::LLVMAddGlobal(bcx.ccx().llmod,
                                                    llty.to_ref(),
                                                    buf)
                            });
                        bcx.ccx().extern_const_values.borrow_mut()
                           .insert(did, llval);
                        llval
                    }
                }
            }

            let did = get_did(bcx.ccx(), did);
            let val = get_val(bcx, did, const_ty);
            DatumBlock(bcx, Datum(val, const_ty, LvalueExpr))
        }
        _ => {
            DatumBlock(bcx, trans_local_var(bcx, def).to_expr_datum())
        }
    }
}

fn trans_rvalue_stmt_unadjusted<'a>(bcx: &'a Block<'a>,
                                    expr: &ast::Expr)
                                    -> &'a Block<'a> {
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rvalue_stmt");

    if bcx.unreachable.get() {
        return bcx;
    }

    match expr.node {
        ast::ExprParen(e) => {
            trans_into(bcx, e, Ignore)
        }
        ast::ExprBreak(label_opt) => {
            controlflow::trans_break(bcx, expr.id, label_opt)
        }
        ast::ExprAgain(label_opt) => {
            controlflow::trans_cont(bcx, expr.id, label_opt)
        }
        ast::ExprRet(ex) => {
            controlflow::trans_ret(bcx, ex)
        }
        ast::ExprWhile(cond, body) => {
            controlflow::trans_while(bcx, expr.id, cond, body)
        }
        ast::ExprLoop(body, _) => {
            controlflow::trans_loop(bcx, expr.id, body)
        }
        ast::ExprAssign(dst, src) => {
            let src_datum = unpack_datum!(bcx, trans(bcx, src));
            let dst_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, dst, "assign"));

            if ty::type_needs_drop(bcx.tcx(), dst_datum.ty) {
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
                let src_datum = unpack_datum!(
                    bcx, src_datum.to_rvalue_datum(bcx, "ExprAssign"));
                bcx = glue::drop_ty(bcx, dst_datum.val, dst_datum.ty);
                src_datum.store_to(bcx, dst_datum.val)
            } else {
                src_datum.store_to(bcx, dst_datum.val)
            }
        }
        ast::ExprAssignOp(op, dst, src) => {
            trans_assign_op(bcx, expr, op, dst, src)
        }
        ast::ExprInlineAsm(ref a) => {
            asm::trans_inline_asm(bcx, a)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                format!("trans_rvalue_stmt_unadjusted reached \
                      fall-through case: {:?}",
                     expr.node));
        }
    }
}

fn trans_rvalue_dps_unadjusted<'a>(bcx: &'a Block<'a>,
                                   expr: &ast::Expr,
                                   dest: Dest)
                                   -> &'a Block<'a> {
    let _icx = push_ctxt("trans_rvalue_dps_unadjusted");
    let mut bcx = bcx;
    let tcx = bcx.tcx();
    let fcx = bcx.fcx;

    match expr.node {
        ast::ExprParen(e) => {
            trans_into(bcx, e, dest)
        }
        ast::ExprPath(_) => {
            trans_def_dps_unadjusted(bcx, expr, bcx.def(expr.id), dest)
        }
        ast::ExprIf(cond, thn, els) => {
            controlflow::trans_if(bcx, expr.id, cond, thn, els, dest)
        }
        ast::ExprMatch(discr, ref arms) => {
            _match::trans_match(bcx, expr, discr, arms.as_slice(), dest)
        }
        ast::ExprBlock(blk) => {
            controlflow::trans_block(bcx, blk, dest)
        }
        ast::ExprStruct(_, ref fields, base) => {
            trans_rec_or_struct(bcx,
                                fields.as_slice(),
                                base,
                                expr.span,
                                expr.id,
                                dest)
        }
        ast::ExprTup(ref args) => {
            let repr = adt::represent_type(bcx.ccx(), expr_ty(bcx, expr));
            let numbered_fields: Vec<(uint, @ast::Expr)> =
                args.iter().enumerate().map(|(i, arg)| (i, *arg)).collect();
            trans_adt(bcx, repr, 0, numbered_fields.as_slice(), None, dest)
        }
        ast::ExprLit(lit) => {
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
        ast::ExprVstore(contents, ast::ExprVstoreSlice) |
        ast::ExprVstore(contents, ast::ExprVstoreMutSlice) => {
            fcx.push_ast_cleanup_scope(contents.id);
            bcx = tvec::trans_slice_vstore(bcx, expr, contents, dest);
            fcx.pop_and_trans_ast_cleanup_scope(bcx, contents.id)
        }
        ast::ExprVec(..) | ast::ExprRepeat(..) => {
            tvec::trans_fixed_vstore(bcx, expr, expr, dest)
        }
        ast::ExprFnBlock(decl, body) |
        ast::ExprProc(decl, body) => {
            let expr_ty = expr_ty(bcx, expr);
            let sigil = ty::ty_closure_sigil(expr_ty);
            debug!("translating block function {} with type {}",
                   expr_to_str(expr), expr_ty.repr(tcx));
            closure::trans_expr_fn(bcx, sigil, decl, body, expr.id, dest)
        }
        ast::ExprCall(f, ref args) => {
            callee::trans_call(bcx, expr, f, callee::ArgExprs(args.as_slice()), dest)
        }
        ast::ExprMethodCall(_, _, ref args) => {
            callee::trans_method_call(bcx,
                                      expr,
                                      *args.get(0),
                                      callee::ArgExprs(args.as_slice()),
                                      dest)
        }
        ast::ExprBinary(_, lhs, rhs) => {
            // if not overloaded, would be RvalueDatumExpr
            let lhs = unpack_datum!(bcx, trans(bcx, lhs));
            let rhs_datum = unpack_datum!(bcx, trans(bcx, rhs));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id), lhs,
                                Some((rhs_datum, rhs.id)), Some(dest)).bcx
        }
        ast::ExprUnary(_, subexpr) => {
            // if not overloaded, would be RvalueDatumExpr
            let arg = unpack_datum!(bcx, trans(bcx, subexpr));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id),
                                arg, None, Some(dest)).bcx
        }
        ast::ExprIndex(base, idx) => {
            // if not overloaded, would be RvalueDatumExpr
            let base = unpack_datum!(bcx, trans(bcx, base));
            let idx_datum = unpack_datum!(bcx, trans(bcx, idx));
            trans_overloaded_op(bcx, expr, MethodCall::expr(expr.id), base,
                                Some((idx_datum, idx.id)), Some(dest)).bcx
        }
        ast::ExprCast(val, _) => {
            // DPS output mode means this is a trait cast:
            match ty::get(node_id_type(bcx, expr.id)).sty {
                ty::ty_trait(..) => {
                    let datum = unpack_datum!(bcx, trans(bcx, val));
                    meth::trans_trait_cast(bcx, datum, expr.id, dest)
                }
                _ => {
                    bcx.tcx().sess.span_bug(expr.span,
                                            "expr_cast of non-trait");
                }
            }
        }
        ast::ExprAssignOp(op, dst, src) => {
            trans_assign_op(bcx, expr, op, dst, src)
        }
        ast::ExprBox(_, contents) => {
            // Special case for `Gc<T>` for now. The other case, for unique
            // pointers, is handled in `trans_rvalue_datum_unadjusted`.
            trans_gc(bcx, expr, contents, dest)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                format!("trans_rvalue_dps_unadjusted reached fall-through case: {:?}",
                     expr.node));
        }
    }
}

fn trans_def_dps_unadjusted<'a>(
                            bcx: &'a Block<'a>,
                            ref_expr: &ast::Expr,
                            def: ast::Def,
                            dest: Dest)
                            -> &'a Block<'a> {
    let _icx = push_ctxt("trans_def_dps_unadjusted");

    let lldest = match dest {
        SaveIn(lldest) => lldest,
        Ignore => { return bcx; }
    };

    match def {
        ast::DefVariant(tid, vid, _) => {
            let variant_info = ty::enum_variant_with_id(bcx.tcx(), tid, vid);
            if variant_info.args.len() > 0u {
                // N-ary variant.
                let llfn = callee::trans_fn_ref(bcx, vid, ExprId(ref_expr.id));
                Store(bcx, llfn, lldest);
                return bcx;
            } else {
                // Nullary variant.
                let ty = expr_ty(bcx, ref_expr);
                let repr = adt::represent_type(bcx.ccx(), ty);
                adt::trans_start_init(bcx, repr, lldest,
                                      variant_info.disr_val);
                return bcx;
            }
        }
        ast::DefStruct(_) => {
            let ty = expr_ty(bcx, ref_expr);
            match ty::get(ty).sty {
                ty::ty_struct(did, _) if ty::has_dtor(bcx.tcx(), did) => {
                    let repr = adt::represent_type(bcx.ccx(), ty);
                    adt::trans_start_init(bcx, repr, lldest, 0);
                }
                _ => {}
            }
            bcx
        }
        _ => {
            bcx.tcx().sess.span_bug(ref_expr.span, format!(
                "Non-DPS def {:?} referened by {}",
                def, bcx.node_id_to_str(ref_expr.id)));
        }
    }
}

fn trans_def_fn_unadjusted<'a>(bcx: &'a Block<'a>,
                               ref_expr: &ast::Expr,
                               def: ast::Def) -> DatumBlock<'a, Expr> {
    let _icx = push_ctxt("trans_def_datum_unadjusted");

    let llfn = match def {
        ast::DefFn(did, _) |
        ast::DefStruct(did) | ast::DefVariant(_, did, _) |
        ast::DefStaticMethod(did, ast::FromImpl(_), _) => {
            callee::trans_fn_ref(bcx, did, ExprId(ref_expr.id))
        }
        ast::DefStaticMethod(impl_did, ast::FromTrait(trait_did), _) => {
            meth::trans_static_method_callee(bcx, impl_did,
                                             trait_did, ref_expr.id)
        }
        _ => {
            bcx.tcx().sess.span_bug(ref_expr.span, format!(
                    "trans_def_fn_unadjusted invoked on: {:?} for {}",
                    def,
                    ref_expr.repr(bcx.tcx())));
        }
    };

    let fn_ty = expr_ty(bcx, ref_expr);
    DatumBlock(bcx, Datum(llfn, fn_ty, RvalueExpr(Rvalue(ByValue))))
}

pub fn trans_local_var<'a>(bcx: &'a Block<'a>,
                           def: ast::Def)
                           -> Datum<Lvalue> {
    /*!
     * Translates a reference to a local variable or argument.
     * This always results in an lvalue datum.
     */

    let _icx = push_ctxt("trans_local_var");

    return match def {
        ast::DefUpvar(nid, _, _, _) => {
            // Can't move upvars, so this is never a ZeroMemLastUse.
            let local_ty = node_id_type(bcx, nid);
            match bcx.fcx.llupvars.borrow().find(&nid) {
                Some(&val) => Datum(val, local_ty, Lvalue),
                None => {
                    bcx.sess().bug(format!(
                        "trans_local_var: no llval for upvar {:?} found", nid));
                }
            }
        }
        ast::DefArg(nid, _) => {
            take_local(bcx, &*bcx.fcx.llargs.borrow(), nid)
        }
        ast::DefLocal(nid, _) | ast::DefBinding(nid, _) => {
            take_local(bcx, &*bcx.fcx.lllocals.borrow(), nid)
        }
        _ => {
            bcx.sess().unimpl(format!(
                "unsupported def type in trans_local_var: {:?}", def));
        }
    };

    fn take_local<'a>(bcx: &'a Block<'a>,
                      table: &NodeMap<Datum<Lvalue>>,
                      nid: ast::NodeId)
                      -> Datum<Lvalue> {
        let datum = match table.find(&nid) {
            Some(&v) => v,
            None => {
                bcx.sess().bug(format!(
                    "trans_local_var: no datum for local/arg {:?} found", nid));
            }
        };
        debug!("take_local(nid={:?}, v={}, ty={})",
               nid, bcx.val_to_str(datum.val), bcx.ty_to_str(datum.ty));
        datum
    }
}

pub fn with_field_tys<R>(tcx: &ty::ctxt,
                         ty: ty::t,
                         node_id_opt: Option<ast::NodeId>,
                         op: |ty::Disr, (&[ty::field])| -> R)
                         -> R {
    /*!
     * Helper for enumerating the field types of structs, enums, or records.
     * The optional node ID here is the node ID of the path identifying the enum
     * variant in use. If none, this cannot possibly an enum variant (so, if it
     * is and `node_id_opt` is none, this function fails).
     */

    match ty::get(ty).sty {
        ty::ty_struct(did, ref substs) => {
            op(0, struct_fields(tcx, did, substs).as_slice())
        }

        ty::ty_enum(_, ref substs) => {
            // We want the *variant* ID here, not the enum ID.
            match node_id_opt {
                None => {
                    tcx.sess.bug(format!(
                        "cannot get field types from the enum type {} \
                         without a node ID",
                        ty.repr(tcx)));
                }
                Some(node_id) => {
                    let def = tcx.def_map.borrow().get_copy(&node_id);
                    match def {
                        ast::DefVariant(enum_id, variant_id, _) => {
                            let variant_info = ty::enum_variant_with_id(
                                tcx, enum_id, variant_id);
                            op(variant_info.disr_val,
                               struct_fields(tcx,
                                             variant_id,
                                             substs).as_slice())
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
            tcx.sess.bug(format!(
                "cannot get field types from the type {}",
                ty.repr(tcx)));
        }
    }
}

fn trans_rec_or_struct<'a>(
                       bcx: &'a Block<'a>,
                       fields: &[ast::Field],
                       base: Option<@ast::Expr>,
                       expr_span: codemap::Span,
                       id: ast::NodeId,
                       dest: Dest)
                       -> &'a Block<'a> {
    let _icx = push_ctxt("trans_rec");
    let bcx = bcx;

    let ty = node_id_type(bcx, id);
    let tcx = bcx.tcx();
    with_field_tys(tcx, ty, Some(id), |discr, field_tys| {
        let mut need_base = slice::from_elem(field_tys.len(), true);

        let numbered_fields = fields.iter().map(|field| {
            let opt_pos =
                field_tys.iter().position(|field_ty|
                                          field_ty.ident.name == field.ident.node.name);
            match opt_pos {
                Some(i) => {
                    need_base[i] = false;
                    (i, field.expr)
                }
                None => {
                    tcx.sess.span_bug(field.span,
                                      "Couldn't find field in struct type")
                }
            }
        }).collect::<Vec<_>>();
        let optbase = match base {
            Some(base_expr) => {
                let mut leftovers = Vec::new();
                for (i, b) in need_base.iter().enumerate() {
                    if *b {
                        leftovers.push((i, field_tys[i].mt.ty))
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

        let repr = adt::represent_type(bcx.ccx(), ty);
        trans_adt(bcx, repr, discr, numbered_fields.as_slice(), optbase, dest)
    })
}

/**
 * Information that `trans_adt` needs in order to fill in the fields
 * of a struct copied from a base struct (e.g., from an expression
 * like `Foo { a: b, ..base }`.
 *
 * Note that `fields` may be empty; the base expression must always be
 * evaluated for side-effects.
 */
struct StructBaseInfo {
    /// The base expression; will be evaluated after all explicit fields.
    expr: @ast::Expr,
    /// The indices of fields to copy paired with their types.
    fields: Vec<(uint, ty::t)> }

/**
 * Constructs an ADT instance:
 *
 * - `fields` should be a list of field indices paired with the
 * expression to store into that field.  The initializers will be
 * evaluated in the order specified by `fields`.
 *
 * - `optbase` contains information on the base struct (if any) from
 * which remaining fields are copied; see comments on `StructBaseInfo`.
 */
fn trans_adt<'a>(
             bcx: &'a Block<'a>,
             repr: &adt::Repr,
             discr: ty::Disr,
             fields: &[(uint, @ast::Expr)],
             optbase: Option<StructBaseInfo>,
             dest: Dest)
             -> &'a Block<'a> {
    let _icx = push_ctxt("trans_adt");
    let fcx = bcx.fcx;
    let mut bcx = bcx;
    let addr = match dest {
        Ignore => {
            for &(_i, e) in fields.iter() {
                bcx = trans_into(bcx, e, Ignore);
            }
            for sbi in optbase.iter() {
                // FIXME #7261: this moves entire base, not just certain fields
                bcx = trans_into(bcx, sbi.expr, Ignore);
            }
            return bcx;
        }
        SaveIn(pos) => pos
    };

    // This scope holds intermediates that must be cleaned should
    // failure occur before the ADT as a whole is ready.
    let custom_cleanup_scope = fcx.push_custom_cleanup_scope();

    adt::trans_start_init(bcx, repr, addr, discr);

    for &(i, e) in fields.iter() {
        let dest = adt::trans_field_ptr(bcx, repr, addr, discr, i);
        let e_ty = expr_ty_adjusted(bcx, e);
        bcx = trans_into(bcx, e, SaveIn(dest));
        fcx.schedule_drop_mem(cleanup::CustomScope(custom_cleanup_scope),
                              dest, e_ty);
    }

    for base in optbase.iter() {
        // FIXME #6573: is it sound to use the destination's repr on the base?
        // And, would it ever be reasonable to be here with discr != 0?
        let base_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, base.expr, "base"));
        for &(i, t) in base.fields.iter() {
            let datum = base_datum.get_element(
                t,
                |srcval| adt::trans_field_ptr(bcx, repr, srcval, discr, i));
            let dest = adt::trans_field_ptr(bcx, repr, addr, discr, i);
            bcx = datum.store_to(bcx, dest);
        }
    }

    fcx.pop_custom_cleanup_scope(custom_cleanup_scope);

    return bcx;
}


fn trans_immediate_lit<'a>(bcx: &'a Block<'a>,
                           expr: &ast::Expr,
                           lit: ast::Lit)
                           -> DatumBlock<'a, Expr> {
    // must not be a string constant, that is a RvalueDpsExpr
    let _icx = push_ctxt("trans_immediate_lit");
    let ty = expr_ty(bcx, expr);
    let v = consts::const_lit(bcx.ccx(), expr, lit);
    immediate_rvalue_bcx(bcx, v, ty).to_expr_datumblock()
}

fn trans_unary<'a>(bcx: &'a Block<'a>,
                   expr: &ast::Expr,
                   op: ast::UnOp,
                   sub_expr: &ast::Expr)
                   -> DatumBlock<'a, Expr> {
    let ccx = bcx.ccx();
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_unary_datum");

    let method_call = MethodCall::expr(expr.id);

    // The only overloaded operator that is translated to a datum
    // is an overloaded deref, since it is always yields a `&T`.
    // Otherwise, we should be in the RvalueDpsExpr path.
    assert!(
        op == ast::UnDeref ||
        !ccx.maps.method_map.borrow().contains_key(&method_call));

    let un_ty = expr_ty(bcx, expr);

    match op {
        ast::UnNot => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            let llresult = if ty::type_is_bool(un_ty) {
                let val = datum.to_llscalarish(bcx);
                let llcond = ICmp(bcx,
                                  lib::llvm::IntEQ,
                                  val,
                                  C_bool(ccx, false));
                Select(bcx, llcond, C_bool(ccx, true), C_bool(ccx, false))
            } else {
                // Note: `Not` is bitwise, not suitable for logical not.
                Not(bcx, datum.to_llscalarish(bcx))
            };
            immediate_rvalue_bcx(bcx, llresult, un_ty).to_expr_datumblock()
        }
        ast::UnNeg => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            let val = datum.to_llscalarish(bcx);
            let llneg = {
                if ty::type_is_fp(un_ty) {
                    FNeg(bcx, val)
                } else {
                    Neg(bcx, val)
                }
            };
            immediate_rvalue_bcx(bcx, llneg, un_ty).to_expr_datumblock()
        }
        ast::UnBox => {
            trans_boxed_expr(bcx, un_ty, sub_expr, expr_ty(bcx, sub_expr), heap_managed)
        }
        ast::UnUniq => {
            trans_boxed_expr(bcx, un_ty, sub_expr, expr_ty(bcx, sub_expr), heap_exchange)
        }
        ast::UnDeref => {
            let datum = unpack_datum!(bcx, trans(bcx, sub_expr));
            deref_once(bcx, expr, datum, 0)
        }
    }
}

fn trans_boxed_expr<'a>(bcx: &'a Block<'a>,
                        box_ty: ty::t,
                        contents: &ast::Expr,
                        contents_ty: ty::t,
                        heap: heap)
                        -> DatumBlock<'a, Expr> {
    let _icx = push_ctxt("trans_boxed_expr");
    let fcx = bcx.fcx;
    if heap == heap_exchange {
        let llty = type_of::type_of(bcx.ccx(), contents_ty);
        let size = llsize_of(bcx.ccx(), llty);
        let Result { bcx: bcx, val: val } = malloc_raw_dyn(bcx, contents_ty,
                                                           heap_exchange, size);
        // Unique boxes do not allocate for zero-size types. The standard library may assume
        // that `free` is never called on the pointer returned for `~ZeroSizeType`.
        if llsize_of_alloc(bcx.ccx(), llty) == 0 {
            let bcx = trans_into(bcx, contents, SaveIn(val));
            immediate_rvalue_bcx(bcx, val, box_ty).to_expr_datumblock()
        } else {
            let custom_cleanup_scope = fcx.push_custom_cleanup_scope();
            fcx.schedule_free_value(cleanup::CustomScope(custom_cleanup_scope),
                                    val, heap_exchange);
            let bcx = trans_into(bcx, contents, SaveIn(val));
            fcx.pop_custom_cleanup_scope(custom_cleanup_scope);
            immediate_rvalue_bcx(bcx, val, box_ty).to_expr_datumblock()
        }
    } else {
        let base::MallocResult { bcx, smart_ptr: bx, body } =
            base::malloc_general(bcx, contents_ty, heap);
        let custom_cleanup_scope = fcx.push_custom_cleanup_scope();
        fcx.schedule_free_value(cleanup::CustomScope(custom_cleanup_scope),
                                bx, heap);
        let bcx = trans_into(bcx, contents, SaveIn(body));
        fcx.pop_custom_cleanup_scope(custom_cleanup_scope);
        immediate_rvalue_bcx(bcx, bx, box_ty).to_expr_datumblock()
    }
}

fn trans_addr_of<'a>(bcx: &'a Block<'a>,
                     expr: &ast::Expr,
                     subexpr: &ast::Expr)
                     -> DatumBlock<'a, Expr> {
    let _icx = push_ctxt("trans_addr_of");
    let mut bcx = bcx;
    let sub_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, subexpr, "addr_of"));
    let ty = expr_ty(bcx, expr);
    return immediate_rvalue_bcx(bcx, sub_datum.val, ty).to_expr_datumblock();
}

fn trans_gc<'a>(mut bcx: &'a Block<'a>,
                expr: &ast::Expr,
                contents: &ast::Expr,
                dest: Dest)
                -> &'a Block<'a> {
    let contents_ty = expr_ty(bcx, contents);
    let box_ty = ty::mk_box(bcx.tcx(), contents_ty);
    let expr_ty = expr_ty(bcx, expr);

    let addr = match dest {
        Ignore => {
            return trans_boxed_expr(bcx,
                                    box_ty,
                                    contents,
                                    contents_ty,
                                    heap_managed).bcx
        }
        SaveIn(addr) => addr,
    };

    let repr = adt::represent_type(bcx.ccx(), expr_ty);
    adt::trans_start_init(bcx, repr, addr, 0);
    let field_dest = adt::trans_field_ptr(bcx, repr, addr, 0, 0);
    let contents_datum = unpack_datum!(bcx, trans_boxed_expr(bcx,
                                                             box_ty,
                                                             contents,
                                                             contents_ty,
                                                             heap_managed));
    bcx = contents_datum.store_to(bcx, field_dest);

    // Next, wrap it up in the struct.
    bcx
}

// Important to get types for both lhs and rhs, because one might be _|_
// and the other not.
fn trans_eager_binop<'a>(
                     bcx: &'a Block<'a>,
                     binop_expr: &ast::Expr,
                     binop_ty: ty::t,
                     op: ast::BinOp,
                     lhs_t: ty::t,
                     lhs: ValueRef,
                     rhs_t: ty::t,
                     rhs: ValueRef)
                     -> DatumBlock<'a, Expr> {
    let _icx = push_ctxt("trans_eager_binop");

    let mut intype = {
        if ty::type_is_bot(lhs_t) { rhs_t }
        else { lhs_t }
    };
    let tcx = bcx.tcx();
    if ty::type_is_simd(tcx, intype) {
        intype = ty::simd_type(tcx, intype);
    }
    let is_float = ty::type_is_fp(intype);
    let signed = ty::type_is_signed(intype);

    let rhs = base::cast_shift_expr_rhs(bcx, op, lhs, rhs);

    let mut bcx = bcx;
    let val = match op {
      ast::BiAdd => {
        if is_float { FAdd(bcx, lhs, rhs) }
        else { Add(bcx, lhs, rhs) }
      }
      ast::BiSub => {
        if is_float { FSub(bcx, lhs, rhs) }
        else { Sub(bcx, lhs, rhs) }
      }
      ast::BiMul => {
        if is_float { FMul(bcx, lhs, rhs) }
        else { Mul(bcx, lhs, rhs) }
      }
      ast::BiDiv => {
        if is_float {
            FDiv(bcx, lhs, rhs)
        } else {
            // Only zero-check integers; fp /0 is NaN
            bcx = base::fail_if_zero(bcx, binop_expr.span,
                                     op, rhs, rhs_t);
            if signed {
                SDiv(bcx, lhs, rhs)
            } else {
                UDiv(bcx, lhs, rhs)
            }
        }
      }
      ast::BiRem => {
        if is_float {
            FRem(bcx, lhs, rhs)
        } else {
            // Only zero-check integers; fp %0 is NaN
            bcx = base::fail_if_zero(bcx, binop_expr.span,
                                     op, rhs, rhs_t);
            if signed {
                SRem(bcx, lhs, rhs)
            } else {
                URem(bcx, lhs, rhs)
            }
        }
      }
      ast::BiBitOr => Or(bcx, lhs, rhs),
      ast::BiBitAnd => And(bcx, lhs, rhs),
      ast::BiBitXor => Xor(bcx, lhs, rhs),
      ast::BiShl => Shl(bcx, lhs, rhs),
      ast::BiShr => {
        if signed {
            AShr(bcx, lhs, rhs)
        } else { LShr(bcx, lhs, rhs) }
      }
      ast::BiEq | ast::BiNe | ast::BiLt | ast::BiGe | ast::BiLe | ast::BiGt => {
        if ty::type_is_bot(rhs_t) {
            C_bool(bcx.ccx(), false)
        } else {
            if !ty::type_is_scalar(rhs_t) {
                bcx.tcx().sess.span_bug(binop_expr.span,
                                        "non-scalar comparison");
            }
            let cmpr = base::compare_scalar_types(bcx, lhs, rhs, rhs_t, op);
            bcx = cmpr.bcx;
            ZExt(bcx, cmpr.val, Type::i8(bcx.ccx()))
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

fn trans_lazy_binop<'a>(
                    bcx: &'a Block<'a>,
                    binop_expr: &ast::Expr,
                    op: lazy_binop_ty,
                    a: &ast::Expr,
                    b: &ast::Expr)
                    -> DatumBlock<'a, Expr> {
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

    let lhs_i1 = bool_to_i1(past_lhs, lhs);
    match op {
      lazy_and => CondBr(past_lhs, lhs_i1, before_rhs.llbb, join.llbb),
      lazy_or => CondBr(past_lhs, lhs_i1, join.llbb, before_rhs.llbb)
    }

    let DatumBlock {bcx: past_rhs, datum: rhs} = trans(before_rhs, b);
    let rhs = rhs.to_llscalarish(past_rhs);

    if past_rhs.unreachable.get() {
        return immediate_rvalue_bcx(join, lhs, binop_ty).to_expr_datumblock();
    }

    Br(past_rhs, join.llbb);
    let phi = Phi(join, Type::bool(bcx.ccx()), [lhs, rhs],
                  [past_lhs.llbb, past_rhs.llbb]);

    return immediate_rvalue_bcx(join, phi, binop_ty).to_expr_datumblock();
}

fn trans_binary<'a>(bcx: &'a Block<'a>,
                    expr: &ast::Expr,
                    op: ast::BinOp,
                    lhs: &ast::Expr,
                    rhs: &ast::Expr)
                    -> DatumBlock<'a, Expr> {
    let _icx = push_ctxt("trans_binary");
    let ccx = bcx.ccx();

    // if overloaded, would be RvalueDpsExpr
    assert!(!ccx.maps.method_map.borrow().contains_key(&MethodCall::expr(expr.id)));

    match op {
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
                   lhs_datum.to_str(ccx));
            let lhs_ty = lhs_datum.ty;
            let lhs = lhs_datum.to_llscalarish(bcx);

            debug!("trans_binary (expr {}): rhs_datum={}",
                   expr.id,
                   rhs_datum.to_str(ccx));
            let rhs_ty = rhs_datum.ty;
            let rhs = rhs_datum.to_llscalarish(bcx);
            trans_eager_binop(bcx, expr, binop_ty, op,
                              lhs_ty, lhs, rhs_ty, rhs)
        }
    }
}

fn trans_overloaded_op<'a, 'b>(
                       bcx: &'a Block<'a>,
                       expr: &ast::Expr,
                       method_call: MethodCall,
                       lhs: Datum<Expr>,
                       rhs: Option<(Datum<Expr>, ast::NodeId)>,
                       dest: Option<Dest>)
                       -> Result<'a> {
    let method_ty = bcx.ccx().maps.method_map.borrow().get(&method_call).ty;
    callee::trans_call_inner(bcx,
                             Some(expr_info(expr)),
                             monomorphize_type(bcx, method_ty),
                             |bcx, arg_cleanup_scope| {
                                meth::trans_method_callee(bcx,
                                                          method_call,
                                                          None,
                                                          arg_cleanup_scope)
                             },
                             callee::ArgOverloadedOp(lhs, rhs),
                             dest)
}

fn int_cast(bcx: &Block,
            lldsttype: Type,
            llsrctype: Type,
            llsrc: ValueRef,
            signed: bool)
            -> ValueRef {
    let _icx = push_ctxt("int_cast");
    unsafe {
        let srcsz = llvm::LLVMGetIntTypeWidth(llsrctype.to_ref());
        let dstsz = llvm::LLVMGetIntTypeWidth(lldsttype.to_ref());
        return if dstsz == srcsz {
            BitCast(bcx, llsrc, lldsttype)
        } else if srcsz > dstsz {
            TruncOrBitCast(bcx, llsrc, lldsttype)
        } else if signed {
            SExtOrBitCast(bcx, llsrc, lldsttype)
        } else {
            ZExtOrBitCast(bcx, llsrc, lldsttype)
        };
    }
}

fn float_cast(bcx: &Block,
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

#[deriving(Eq)]
pub enum cast_kind {
    cast_pointer,
    cast_integral,
    cast_float,
    cast_enum,
    cast_other,
}

pub fn cast_type_kind(t: ty::t) -> cast_kind {
    match ty::get(t).sty {
        ty::ty_char       => cast_integral,
        ty::ty_float(..)   => cast_float,
        ty::ty_ptr(..)     => cast_pointer,
        ty::ty_rptr(..)    => cast_pointer,
        ty::ty_bare_fn(..) => cast_pointer,
        ty::ty_int(..)     => cast_integral,
        ty::ty_uint(..)    => cast_integral,
        ty::ty_bool       => cast_integral,
        ty::ty_enum(..)    => cast_enum,
        _                 => cast_other
    }
}

fn trans_imm_cast<'a>(bcx: &'a Block<'a>,
                      expr: &ast::Expr,
                      id: ast::NodeId)
                      -> DatumBlock<'a, Expr> {
    let _icx = push_ctxt("trans_cast");
    let mut bcx = bcx;
    let ccx = bcx.ccx();

    let t_in = expr_ty(bcx, expr);
    let t_out = node_id_type(bcx, id);
    let k_in = cast_type_kind(t_in);
    let k_out = cast_type_kind(t_out);
    let s_in = k_in == cast_integral && ty::type_is_signed(t_in);
    let ll_t_in = type_of::type_of(ccx, t_in);
    let ll_t_out = type_of::type_of(ccx, t_out);

    // Convert the value to be cast into a ValueRef, either by-ref or
    // by-value as appropriate given its type:
    let datum = unpack_datum!(bcx, trans(bcx, expr));
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
                adt::trans_get_discr(bcx, repr, llexpr_ptr, Some(Type::i64(ccx)));
            match k_out {
                cast_integral => int_cast(bcx, ll_t_out,
                                          val_ty(lldiscrim_a),
                                          lldiscrim_a, true),
                cast_float => SIToFP(bcx, lldiscrim_a, ll_t_out),
                _ => ccx.sess().bug(format!("translating unsupported cast: \
                                            {} ({:?}) -> {} ({:?})",
                                            t_in.repr(bcx.tcx()), k_in,
                                            t_out.repr(bcx.tcx()), k_out))
            }
        }
        _ => ccx.sess().bug(format!("translating unsupported cast: \
                                    {} ({:?}) -> {} ({:?})",
                                    t_in.repr(bcx.tcx()), k_in,
                                    t_out.repr(bcx.tcx()), k_out))
    };
    return immediate_rvalue_bcx(bcx, newval, t_out).to_expr_datumblock();
}

fn trans_assign_op<'a>(
                   bcx: &'a Block<'a>,
                   expr: &ast::Expr,
                   op: ast::BinOp,
                   dst: &ast::Expr,
                   src: @ast::Expr)
                   -> &'a Block<'a> {
    let _icx = push_ctxt("trans_assign_op");
    let mut bcx = bcx;

    debug!("trans_assign_op(expr={})", bcx.expr_to_str(expr));

    // User-defined operator methods cannot be used with `+=` etc right now
    assert!(!bcx.ccx().maps.method_map.borrow().contains_key(&MethodCall::expr(expr.id)));

    // Evaluate LHS (destination), which should be an lvalue
    let dst_datum = unpack_datum!(bcx, trans_to_lvalue(bcx, dst, "assign_op"));
    assert!(!ty::type_needs_drop(bcx.tcx(), dst_datum.ty));
    let dst_ty = dst_datum.ty;
    let dst = Load(bcx, dst_datum.val);

    // Evaluate RHS
    let rhs_datum = unpack_datum!(bcx, trans(bcx, src));
    let rhs_ty = rhs_datum.ty;
    let rhs = rhs_datum.to_llscalarish(bcx);

    // Perform computation and store the result
    let result_datum = unpack_datum!(
        bcx, trans_eager_binop(bcx, expr, dst_datum.ty, op,
                               dst_ty, dst, rhs_ty, rhs));
    return result_datum.store_to(bcx, dst_datum.val);
}

fn auto_ref<'a>(bcx: &'a Block<'a>,
                datum: Datum<Expr>,
                expr: &ast::Expr)
                -> DatumBlock<'a, Expr> {
    let mut bcx = bcx;

    // Ensure cleanup of `datum` if not already scheduled and obtain
    // a "by ref" pointer.
    let lv_datum = unpack_datum!(bcx, datum.to_lvalue_datum(bcx, "autoref", expr.id));

    // Compute final type. Note that we are loose with the region and
    // mutability, since those things don't matter in trans.
    let referent_ty = lv_datum.ty;
    let ptr_ty = ty::mk_imm_rptr(bcx.tcx(), ty::ReStatic, referent_ty);

    // Get the pointer.
    let llref = lv_datum.to_llref();

    // Construct the resulting datum, using what was the "by ref"
    // ValueRef of type `referent_ty` to be the "by value" ValueRef
    // of type `&referent_ty`.
    DatumBlock(bcx, Datum(llref, ptr_ty, RvalueExpr(Rvalue(ByValue))))
}

fn deref_multiple<'a>(bcx: &'a Block<'a>,
                      expr: &ast::Expr,
                      datum: Datum<Expr>,
                      times: uint)
                      -> DatumBlock<'a, Expr> {
    let mut bcx = bcx;
    let mut datum = datum;
    for i in range(1, times+1) {
        datum = unpack_datum!(bcx, deref_once(bcx, expr, datum, i));
    }
    DatumBlock { bcx: bcx, datum: datum }
}

fn deref_once<'a>(bcx: &'a Block<'a>,
                  expr: &ast::Expr,
                  datum: Datum<Expr>,
                  derefs: uint)
                  -> DatumBlock<'a, Expr> {
    let ccx = bcx.ccx();
    let bcx = write_guard::root_and_write_guard(&datum, bcx, expr.span,
                                                expr.id, derefs);

    debug!("deref_once(expr={}, datum={}, derefs={})",
           expr.repr(bcx.tcx()),
           datum.to_str(ccx),
           derefs);

    let mut bcx = bcx;

    // Check for overloaded deref.
    let method_call = MethodCall {
        expr_id: expr.id,
        autoderef: derefs as u32
    };
    let method_ty = ccx.maps.method_map.borrow()
                       .find(&method_call).map(|method| method.ty);
    let datum = match method_ty {
        Some(method_ty) => {
            // Overloaded. Evaluate `trans_overloaded_op`, which will
            // invoke the user's deref() method, which basically
            // converts from the `Shaht<T>` pointer that we have into
            // a `&T` pointer.  We can then proceed down the normal
            // path (below) to dereference that `&T`.
            let datum = if derefs == 0 {
                datum
            } else {
                // Always perform an AutoPtr when applying an overloaded auto-deref.
                unpack_datum!(bcx, auto_ref(bcx, datum, expr))
            };
            let val = unpack_result!(bcx, trans_overloaded_op(bcx, expr, method_call,
                                                              datum, None, None));
            let ref_ty = ty::ty_fn_ret(monomorphize_type(bcx, method_ty));
            Datum(val, ref_ty, RvalueExpr(Rvalue(ByValue)))
        }
        None => {
            // Not overloaded. We already have a pointer we know how to deref.
            datum
        }
    };

    let r = match ty::get(datum.ty).sty {
        ty::ty_uniq(content_ty) => {
            deref_owned_pointer(bcx, expr, datum, content_ty)
        }

        ty::ty_box(content_ty) => {
            let datum = unpack_datum!(
                bcx, datum.to_lvalue_datum(bcx, "deref", expr.id));
            let llptrref = datum.to_llref();
            let llptr = Load(bcx, llptrref);
            let llbody = GEPi(bcx, llptr, [0u, abi::box_field_body]);
            DatumBlock(bcx, Datum(llbody, content_ty, LvalueExpr))
        }

        ty::ty_ptr(ty::mt { ty: content_ty, .. }) |
        ty::ty_rptr(_, ty::mt { ty: content_ty, .. }) => {
            assert!(!ty::type_needs_drop(bcx.tcx(), datum.ty));

            let ptr = datum.to_llscalarish(bcx);

            // Always generate an lvalue datum, even if datum.mode is
            // an rvalue.  This is because datum.mode is only an
            // rvalue for non-owning pointers like &T or *T, in which
            // case cleanup *is* scheduled elsewhere, by the true
            // owner (or, in the case of *T, by the user).
            DatumBlock(bcx, Datum(ptr, content_ty, LvalueExpr))
        }

        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                format!("deref invoked on expr of illegal type {}",
                        datum.ty.repr(bcx.tcx())));
        }
    };

    debug!("deref_once(expr={}, derefs={}, result={})",
           expr.id, derefs, r.datum.to_str(ccx));

    return r;

    fn deref_owned_pointer<'a>(bcx: &'a Block<'a>,
                               expr: &ast::Expr,
                               datum: Datum<Expr>,
                               content_ty: ty::t)
                               -> DatumBlock<'a, Expr> {
        /*!
         * We microoptimize derefs of owned pointers a bit here.
         * Basically, the idea is to make the deref of an rvalue
         * result in an rvalue. This helps to avoid intermediate stack
         * slots in the resulting LLVM. The idea here is that, if the
         * `~T` pointer is an rvalue, then we can schedule a *shallow*
         * free of the `~T` pointer, and then return a ByRef rvalue
         * into the pointer. Because the free is shallow, it is legit
         * to return an rvalue, because we know that the contents are
         * not yet scheduled to be freed. The language rules ensure that the
         * contents will be used (or moved) before the free occurs.
         */

        match datum.kind {
            RvalueExpr(Rvalue { mode: ByRef }) => {
                let scope = cleanup::temporary_scope(bcx.tcx(), expr.id);
                let ptr = Load(bcx, datum.val);
                bcx.fcx.schedule_free_value(scope, ptr, heap_exchange);
            }
            RvalueExpr(Rvalue { mode: ByValue }) => {
                let scope = cleanup::temporary_scope(bcx.tcx(), expr.id);
                bcx.fcx.schedule_free_value(scope, datum.val, heap_exchange);
            }
            LvalueExpr => { }
        }

        // If we had an rvalue in, we produce an rvalue out.
        let (llptr, kind) = match datum.kind {
            LvalueExpr => {
                (Load(bcx, datum.val), LvalueExpr)
            }
            RvalueExpr(Rvalue { mode: ByRef }) => {
                (Load(bcx, datum.val), RvalueExpr(Rvalue(ByRef)))
            }
            RvalueExpr(Rvalue { mode: ByValue }) => {
                (datum.val, RvalueExpr(Rvalue(ByRef)))
            }
        };

        let datum = Datum { ty: content_ty, val: llptr, kind: kind };
        DatumBlock { bcx: bcx, datum: datum }
    }
}
