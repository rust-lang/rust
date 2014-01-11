// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Translation of expressions.

## Recommended entry point

If you wish to translate an expression, the preferred way to do
so is to use:

    expr::trans_into(block, expr, Dest) -> block

This will generate code that evaluates `expr`, storing the result into
`Dest`, which must either be the special flag ignore (throw the result
away) or be a pointer to memory of the same type/size as the
expression.  It returns the resulting basic block.  This form will
handle all automatic adjustments for you. The value will be moved if
its type is linear and copied otherwise.

## Translation to a datum

In some cases, `trans_into()` is too narrow of an interface.
Generally this occurs either when you know that the result value is
going to be a scalar, or when you need to evaluate the expression into
some memory location so you can go and inspect it (e.g., assignments,
`match` expressions, the `&` operator).

In such cases, you want the following function:

    trans_to_datum(block, expr) -> DatumBlock

This function generates code to evaluate the expression and return a
`Datum` describing where the result is to be found.  This function
tries to return its result in the most efficient way possible, without
introducing extra copies or sacrificing information.  Therefore, for
lvalue expressions, you always get a by-ref `Datum` in return that
points at the memory for this lvalue.  For rvalue expressions, we will
return a by-value `Datum` whenever possible, but it is often necessary
to allocate a stack slot, store the result of the rvalue in there, and
then return a pointer to the slot (see the discussion later on about
the different kinds of rvalues).

NB: The `trans_to_datum()` function does perform adjustments, but
since it returns a pointer to the value "in place" it does not handle
moves.  If you wish to copy/move the value returned into a new
location, you should use the Datum method `store_to()` (move or copy
depending on type). You can also use `move_to()` (force move) or
`copy_to()` (force copy) for special situations.

## Translating local variables

`trans_local_var()` can be used to trans a ref to a local variable
that is not an expression.  This is needed for captures.

## Ownership and cleanups

The current system for cleanups associates required cleanups with
block contexts.  Block contexts are structured into a tree that
resembles the code itself.  Not every block context has cleanups
associated with it, only those blocks that have a kind of
`block_scope`.  See `common::block_kind` for more details.

If you invoke `trans_into()`, no cleanup is scheduled for you.  The
value is written into the given destination and is assumed to be owned
by that destination.

When you invoke `trans_to_datum()` on an rvalue, the resulting
datum/value will have an appropriate cleanup scheduled for the
innermost cleanup scope.  If you later use `move_to()` or
`drop_val()`, this cleanup will be canceled.

During the evaluation of an expression, temporary cleanups are created
and later canceled.  These represent intermediate or partial results
which must be cleaned up in the event of task failure.

## Implementation details

We divide expressions into three categories, based on how they are most
naturally implemented:

1. Lvalues
2. Datum rvalues
3. DPS rvalues
4. Statement rvalues

Lvalues always refer to user-assignable memory locations.
Translating those always results in a by-ref datum; this introduces
no inefficiencies into the generated code, because all lvalues are
naturally addressable.

Datum rvalues are rvalues that always generate datums as a result.
These are generally scalar results, such as `a+b` where `a` and `b`
are integers.

DPS rvalues are rvalues that, when translated, must be given a
memory location to write into (or the Ignore flag).  These are
generally expressions that produce structural results that are
larger than one word (e.g., a struct literal), but also expressions
(like `if`) that involve control flow (otherwise we'd have to
generate phi nodes).

Finally, statement rvalues are rvalues that always produce a nil
return type, such as `while` loops or assignments (`a = b`).

*/


use back::abi;
use back::link;
use lib::llvm::{ValueRef, llvm, SetLinkage, False};
use lib;
use metadata::csearch;
use middle::trans::_match;
use middle::trans::adt;
use middle::trans::asm;
use middle::trans::base::*;
use middle::trans::base;
use middle::trans::build::*;
use middle::trans::callee::DoAutorefArg;
use middle::trans::callee;
use middle::trans::closure;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::controlflow;
use middle::trans::datum::*;
use middle::trans::debuginfo;
use middle::trans::machine;
use middle::trans::meth;
use middle::trans::inline;
use middle::trans::tvec;
use middle::trans::type_of;
use middle::ty::struct_fields;
use middle::ty::{AutoBorrowObj, AutoDerefRef, AutoAddEnv, AutoObject, AutoUnsafe};
use middle::ty::{AutoPtr, AutoBorrowVec, AutoBorrowVecRef, AutoBorrowFn};
use middle::ty;
use util::common::indenter;
use util::ppaux::Repr;
use middle::trans::machine::llsize_of;

use middle::trans::type_::Type;

use std::hashmap::HashMap;
use std::vec;
use syntax::print::pprust::{expr_to_str};
use syntax::ast;
use syntax::ast_map::PathMod;
use syntax::codemap;

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

pub fn trans_to_datum<'a>(bcx: &'a Block<'a>, expr: &ast::Expr)
                      -> DatumBlock<'a> {
    debug!("trans_to_datum(expr={})", bcx.expr_to_str(expr));

    let mut bcx = bcx;
    let mut datum = unpack_datum!(bcx, trans_to_datum_unadjusted(bcx, expr));
    let adjustment = {
        let adjustments = bcx.tcx().adjustments.borrow();
        match adjustments.get().find_copy(&expr.id) {
            None => { return DatumBlock {bcx: bcx, datum: datum}; }
            Some(adj) => { adj }
        }
    };
    debug!("unadjusted datum: {}", datum.to_str(bcx.ccx()));
    match *adjustment {
        AutoAddEnv(..) => {
            datum = unpack_datum!(bcx, add_env(bcx, expr, datum));
        }
        AutoDerefRef(ref adj) => {
            if adj.autoderefs > 0 {
                datum =
                    unpack_datum!(
                        bcx,
                        datum.autoderef(bcx, expr.span,
                                        expr.id, adj.autoderefs));
            }

            datum = match adj.autoref {
                None => {
                    datum
                }
                Some(AutoUnsafe(..)) | // region + unsafe ptrs have same repr
                Some(AutoPtr(..)) => {
                    unpack_datum!(bcx, auto_ref(bcx, datum))
                }
                Some(AutoBorrowVec(..)) => {
                    unpack_datum!(bcx, auto_slice(bcx, adj.autoderefs,
                                                  expr, datum))
                }
                Some(AutoBorrowVecRef(..)) => {
                    unpack_datum!(bcx, auto_slice_and_ref(bcx, adj.autoderefs,
                                                          expr, datum))
                }
                Some(AutoBorrowFn(..)) => {
                    let adjusted_ty = ty::adjust_ty(bcx.tcx(), expr.span,
                                                    datum.ty, Some(adjustment));
                    unpack_datum!(bcx, auto_borrow_fn(bcx, adjusted_ty, datum))
                }
                Some(AutoBorrowObj(..)) => {
                    unpack_datum!(bcx, auto_borrow_obj(
                        bcx, adj.autoderefs, expr, datum))
                }
            };
        }
        AutoObject(ref sigil, ref region, _, _, _, _) => {

            let adjusted_ty = ty::expr_ty_adjusted(bcx.tcx(), expr);
            let scratch = scratch_datum(bcx, adjusted_ty, "__adjust", false);

            let trait_store = match *sigil {
                ast::BorrowedSigil => ty::RegionTraitStore(region.expect("expected valid region")),
                ast::OwnedSigil => ty::UniqTraitStore,
                ast::ManagedSigil => ty::BoxTraitStore
            };

            bcx = meth::trans_trait_cast(bcx, expr, expr.id, SaveIn(scratch.val),
                                         trait_store, false /* no adjustments */);

            datum = scratch.to_appropriate_datum(bcx);
            datum.add_clean(bcx);
        }
    }
    debug!("after adjustments, datum={}", datum.to_str(bcx.ccx()));
    return DatumBlock {bcx: bcx, datum: datum};

    fn auto_ref<'a>(bcx: &'a Block<'a>, datum: Datum) -> DatumBlock<'a> {
        DatumBlock {bcx: bcx, datum: datum.to_rptr(bcx)}
    }

    fn auto_borrow_fn<'a>(
                      bcx: &'a Block<'a>,
                      adjusted_ty: ty::t,
                      datum: Datum)
                      -> DatumBlock<'a> {
        // Currently, all closure types are represented precisely the
        // same, so no runtime adjustment is required, but we still
        // must patchup the type.
        DatumBlock {bcx: bcx,
                    datum: Datum {val: datum.val, ty: adjusted_ty,
                                  mode: datum.mode}}
    }

    fn auto_slice<'a>(
                  bcx: &'a Block<'a>,
                  autoderefs: uint,
                  expr: &ast::Expr,
                  datum: Datum)
                  -> DatumBlock<'a> {
        // This is not the most efficient thing possible; since slices
        // are two words it'd be better if this were compiled in
        // 'dest' mode, but I can't find a nice way to structure the
        // code and keep it DRY that accommodates that use case at the
        // moment.

        let tcx = bcx.tcx();
        let unit_ty = ty::sequence_element_type(tcx, datum.ty);

        let (bcx, base, len) =
            datum.get_vec_base_and_len(bcx, expr.span, expr.id, autoderefs+1);

        // this type may have a different region/mutability than the
        // real one, but it will have the same runtime representation
        let slice_ty = ty::mk_vec(tcx,
                                  ty::mt { ty: unit_ty, mutbl: ast::MutImmutable },
                                  ty::vstore_slice(ty::ReStatic));

        let scratch = scratch_datum(bcx, slice_ty, "__adjust", false);

        Store(bcx, base, GEPi(bcx, scratch.val, [0u, abi::slice_elt_base]));
        Store(bcx, len, GEPi(bcx, scratch.val, [0u, abi::slice_elt_len]));
        DatumBlock {bcx: bcx, datum: scratch}
    }

    fn add_env<'a>(bcx: &'a Block<'a>, expr: &ast::Expr, datum: Datum)
               -> DatumBlock<'a> {
        // This is not the most efficient thing possible; since closures
        // are two words it'd be better if this were compiled in
        // 'dest' mode, but I can't find a nice way to structure the
        // code and keep it DRY that accommodates that use case at the
        // moment.

        let tcx = bcx.tcx();
        let closure_ty = expr_ty_adjusted(bcx, expr);
        debug!("add_env(closure_ty={})", closure_ty.repr(tcx));
        let scratch = scratch_datum(bcx, closure_ty, "__adjust", false);
        let llfn = GEPi(bcx, scratch.val, [0u, abi::fn_field_code]);
        assert_eq!(datum.appropriate_mode(bcx.ccx()), ByValue);
        Store(bcx, datum.to_appropriate_llval(bcx), llfn);
        let llenv = GEPi(bcx, scratch.val, [0u, abi::fn_field_box]);
        Store(bcx, base::null_env_ptr(bcx.ccx()), llenv);
        DatumBlock {bcx: bcx, datum: scratch}
    }

    fn auto_slice_and_ref<'a>(
                          bcx: &'a Block<'a>,
                          autoderefs: uint,
                          expr: &ast::Expr,
                          datum: Datum)
                          -> DatumBlock<'a> {
        let DatumBlock { bcx, datum } = auto_slice(bcx, autoderefs, expr, datum);
        auto_ref(bcx, datum)
    }

    fn auto_borrow_obj<'a>(
                       mut bcx: &'a Block<'a>,
                       autoderefs: uint,
                       expr: &ast::Expr,
                       source_datum: Datum)
                       -> DatumBlock<'a> {
        let tcx = bcx.tcx();
        let target_obj_ty = expr_ty_adjusted(bcx, expr);
        debug!("auto_borrow_obj(target={})",
               target_obj_ty.repr(tcx));

        // Extract source store information
        let (source_store, source_mutbl) = match ty::get(source_datum.ty).sty {
            ty::ty_trait(_, _, s, m, _) => (s, m),
            _ => {
                bcx.sess().span_bug(
                    expr.span,
                    format!("auto_borrow_trait_obj expected a trait, found {}",
                         source_datum.ty.repr(bcx.tcx())));
            }
        };

        // check if any borrowing is really needed or we could reuse the source_datum instead
        match ty::get(target_obj_ty).sty {
            ty::ty_trait(_, _, ty::RegionTraitStore(target_scope), target_mutbl, _) => {
                if target_mutbl == ast::MutImmutable && target_mutbl == source_mutbl {
                    match source_store {
                        ty::RegionTraitStore(source_scope) => {
                            if tcx.region_maps.is_subregion_of(target_scope, source_scope) {
                                return DatumBlock { bcx: bcx, datum: source_datum };
                            }
                        },
                        _ => {}

                    };
                }
            },
            _ => {}
        }

        let scratch = scratch_datum(bcx, target_obj_ty,
                                    "__auto_borrow_obj", false);

        // Convert a @Object, ~Object, or &Object pair into an &Object pair.

        // Get a pointer to the source object, which is represented as
        // a (vtable, data) pair.
        let source_llval = source_datum.to_ref_llval(bcx);

        // Set the vtable field of the new pair
        let vtable_ptr = GEPi(bcx, source_llval, [0u, abi::trt_field_vtable]);
        let vtable = Load(bcx, vtable_ptr);
        Store(bcx, vtable, GEPi(bcx, scratch.val, [0u, abi::trt_field_vtable]));

        // Load the data for the source, which is either an @T,
        // ~T, or &T, depending on source_obj_ty.
        let source_data_ptr = GEPi(bcx, source_llval, [0u, abi::trt_field_box]);
        let source_data = Load(bcx, source_data_ptr); // always a ptr
        let target_data = match source_store {
            ty::BoxTraitStore(..) => {
                // For deref of @T, create a dummy datum and use the datum's
                // deref method. This is more work than just calling GEPi
                // ourselves. Note that we don't know the type T, so
                // just substitute `i8`-- it doesn't really matter for
                // our purposes right now.
                let source_ty = ty::mk_box(tcx, ty::mk_i8());
                let source_datum =
                    Datum {val: source_data,
                           ty: source_ty,
                           mode: ByValue};
                let derefd_datum =
                    unpack_datum!(bcx,
                                  source_datum.deref(bcx,
                                                     expr,
                                                     autoderefs));
                derefd_datum.to_rptr(bcx).to_value_llval(bcx)
            }
            ty::UniqTraitStore(..) => {
                // For a ~T box, there may or may not be a header,
                // depending on whether the type T references managed
                // boxes. However, since we do not *know* the type T
                // for objects, this presents a hurdle. Our solution is
                // to load the "borrow offset" from the type descriptor;
                // this value will either be 0 or sizeof(BoxHeader), depending
                // on the type T.
                let llopaque =
                    PointerCast(bcx, source_data, Type::opaque().ptr_to());
                let lltydesc_ptr_ptr =
                    PointerCast(bcx, vtable,
                                bcx.ccx().tydesc_type.ptr_to().ptr_to());
                let lltydesc_ptr =
                    Load(bcx, lltydesc_ptr_ptr);
                let borrow_offset_ptr =
                    GEPi(bcx, lltydesc_ptr,
                         [0, abi::tydesc_field_borrow_offset]);
                let borrow_offset =
                    Load(bcx, borrow_offset_ptr);
                InBoundsGEP(bcx, llopaque, [borrow_offset])
            }
            ty::RegionTraitStore(..) => {
                source_data
            }
        };
        Store(bcx, target_data,
              GEPi(bcx, scratch.val, [0u, abi::trt_field_box]));

        DatumBlock { bcx: bcx, datum: scratch }
    }
}

pub fn trans_into<'a>(bcx: &'a Block<'a>, expr: &ast::Expr, dest: Dest)
                  -> &'a Block<'a> {
    let adjustment_found = {
        let adjustments = bcx.tcx().adjustments.borrow();
        adjustments.get().contains_key(&expr.id)
    };
    if adjustment_found {
        // use trans_to_datum, which is mildly less efficient but
        // which will perform the adjustments:
        let datumblock = trans_to_datum(bcx, expr);
        return match dest {
            Ignore => datumblock.bcx,
            SaveIn(lldest) => datumblock.store_to(INIT, lldest)
        };
    }

    trans_into_unadjusted(bcx, expr, dest)
}

pub fn trans_into_unadjusted<'a>(
                             bcx: &'a Block<'a>,
                             expr: &ast::Expr,
                             dest: Dest)
                             -> &'a Block<'a> {
    let ty = expr_ty(bcx, expr);

    debug!("trans_into_unadjusted(expr={}, dest={})",
           bcx.expr_to_str(expr),
           dest.to_str(bcx.ccx()));
    let _indenter = indenter();

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    let dest = {
        if ty::type_is_voidish(bcx.tcx(), ty) {
            Ignore
        } else {
            dest
        }
    };

    let kind = bcx.expr_kind(expr);
    debug!("expr kind = {:?}", kind);
    return match kind {
        ty::LvalueExpr => {
            let datumblock = trans_lvalue_unadjusted(bcx, expr);
            match dest {
                Ignore => datumblock.bcx,
                SaveIn(lldest) => datumblock.store_to(INIT, lldest)
            }
        }
        ty::RvalueDatumExpr => {
            let datumblock = trans_rvalue_datum_unadjusted(bcx, expr);
            match dest {
                Ignore => datumblock.drop_val(),

                // When processing an rvalue, the value will be newly
                // allocated, so we always `move_to` so as not to
                // unnecessarily inc ref counts and so forth:
                SaveIn(lldest) => datumblock.move_to(INIT, lldest)
            }
        }
        ty::RvalueDpsExpr => {
            trans_rvalue_dps_unadjusted(bcx, expr, dest)
        }
        ty::RvalueStmtExpr => {
            trans_rvalue_stmt_unadjusted(bcx, expr)
        }
    };
}

fn trans_lvalue<'a>(bcx: &'a Block<'a>, expr: &ast::Expr) -> DatumBlock<'a> {
    /*!
     *
     * Translates an lvalue expression, always yielding a by-ref
     * datum.  Generally speaking you should call trans_to_datum()
     * instead, but sometimes we call trans_lvalue() directly as a
     * means of asserting that a particular expression is an lvalue. */

    let adjustment_opt = {
        let adjustments = bcx.tcx().adjustments.borrow();
        adjustments.get().find_copy(&expr.id)
    };
    match adjustment_opt {
        None => trans_lvalue_unadjusted(bcx, expr),
        Some(_) => {
            bcx.sess().span_bug(
                expr.span,
                format!("trans_lvalue() called on an expression \
                      with adjustments"));
        }
    }
}

fn trans_to_datum_unadjusted<'a>(bcx: &'a Block<'a>, expr: &ast::Expr)
                             -> DatumBlock<'a> {
    /*!
     * Translates an expression into a datum.  If this expression
     * is an rvalue, this will result in a temporary value being
     * created.  If you plan to store the value somewhere else,
     * you should prefer `trans_into()` instead.
     */

    let mut bcx = bcx;

    debug!("trans_to_datum_unadjusted(expr={})", bcx.expr_to_str(expr));
    let _indenter = indenter();

    debuginfo::set_source_location(bcx.fcx, expr.id, expr.span);

    match ty::expr_kind(bcx.tcx(), bcx.ccx().maps.method_map, expr) {
        ty::LvalueExpr => {
            return trans_lvalue_unadjusted(bcx, expr);
        }

        ty::RvalueDatumExpr => {
            let datum = unpack_datum!(bcx, {
                trans_rvalue_datum_unadjusted(bcx, expr)
            });
            datum.add_clean(bcx);
            return DatumBlock {bcx: bcx, datum: datum};
        }

        ty::RvalueStmtExpr => {
            bcx = trans_rvalue_stmt_unadjusted(bcx, expr);
            return nil(bcx, expr_ty(bcx, expr));
        }

        ty::RvalueDpsExpr => {
            let ty = expr_ty(bcx, expr);
            if ty::type_is_voidish(bcx.tcx(), ty) {
                bcx = trans_rvalue_dps_unadjusted(bcx, expr, Ignore);
                return nil(bcx, ty);
            } else {
                let scratch = scratch_datum(bcx, ty, "", false);
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
                let scratch = scratch.to_appropriate_datum(bcx);

                scratch.add_clean(bcx);
                return DatumBlock {bcx: bcx, datum: scratch};
            }
        }
    }

    fn nil<'a>(bcx: &'a Block<'a>, ty: ty::t) -> DatumBlock<'a> {
        let datum = immediate_rvalue(C_nil(), ty);
        DatumBlock {
            bcx: bcx,
            datum: datum,
        }
    }
}

fn trans_rvalue_datum_unadjusted<'a>(bcx: &'a Block<'a>, expr: &ast::Expr)
                                 -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_rvalue_datum_unadjusted");

    match expr.node {
        ast::ExprPath(_) | ast::ExprSelf => {
            return trans_def_datum_unadjusted(bcx, expr, bcx.def(expr.id));
        }
        ast::ExprVstore(contents, ast::ExprVstoreBox) => {
            return tvec::trans_uniq_or_managed_vstore(bcx, heap_managed,
                                                      expr, contents);
        }
        ast::ExprVstore(contents, ast::ExprVstoreUniq) => {
            let heap = heap_for_unique(bcx, expr_ty(bcx, contents));
            return tvec::trans_uniq_or_managed_vstore(bcx, heap,
                                                      expr, contents);
        }
        ast::ExprBox(_, contents) => {
            // Special case for `~T`. (The other case, for GC, is handled in
            // `trans_rvalue_dps_unadjusted`.)
            let box_ty = expr_ty(bcx, expr);
            let contents_ty = expr_ty(bcx, contents);
            let heap = heap_for_unique(bcx, contents_ty);
            return trans_boxed_expr(bcx, box_ty, contents, contents_ty, heap)
        }
        ast::ExprLit(lit) => {
            return trans_immediate_lit(bcx, expr, *lit);
        }
        ast::ExprBinary(_, op, lhs, rhs) => {
            // if overloaded, would be RvalueDpsExpr
            {
                let method_map = bcx.ccx().maps.method_map.borrow();
                assert!(!method_map.get().contains_key(&expr.id));
            }

            return trans_binary(bcx, expr, op, lhs, rhs);
        }
        ast::ExprUnary(_, op, x) => {
            return trans_unary_datum(bcx, expr, op, x);
        }
        ast::ExprAddrOf(_, x) => {
            return trans_addr_of(bcx, expr, x);
        }
        ast::ExprCast(val, _) => {
            return trans_imm_cast(bcx, val, expr.id);
        }
        ast::ExprParen(e) => {
            return trans_rvalue_datum_unadjusted(bcx, e);
        }
        ast::ExprLogLevel => {
            return trans_log_level(bcx);
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

fn trans_rvalue_stmt_unadjusted<'a>(bcx: &'a Block<'a>, expr: &ast::Expr)
                                -> &'a Block<'a> {
    let mut bcx = bcx;
    let _icx = push_ctxt("trans_rvalue_stmt");

    if bcx.unreachable.get() {
        return bcx;
    }

    match expr.node {
        ast::ExprBreak(label_opt) => {
            return controlflow::trans_break(bcx, label_opt);
        }
        ast::ExprAgain(label_opt) => {
            return controlflow::trans_cont(bcx, label_opt);
        }
        ast::ExprRet(ex) => {
            return controlflow::trans_ret(bcx, ex);
        }
        ast::ExprWhile(cond, body) => {
            return controlflow::trans_while(bcx, cond, body);
        }
        ast::ExprLoop(body, opt_label) => {
            // FIXME #6993: map can go away when ast.rs is changed
            return controlflow::trans_loop(bcx, body, opt_label.map(|x| x.name));
        }
        ast::ExprAssign(dst, src) => {
            let src_datum = unpack_datum!(
                bcx, trans_to_datum(bcx, src));
            let dst_datum = unpack_datum!(
                bcx, trans_lvalue(bcx, dst));
            return src_datum.store_to_datum(
                bcx, DROP_EXISTING, dst_datum);
        }
        ast::ExprAssignOp(callee_id, op, dst, src) => {
            return trans_assign_op(bcx, expr, callee_id, op, dst, src);
        }
        ast::ExprParen(a) => {
            return trans_rvalue_stmt_unadjusted(bcx, a);
        }
        ast::ExprInlineAsm(ref a) => {
            return asm::trans_inline_asm(bcx, a);
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                format!("trans_rvalue_stmt_unadjusted reached \
                      fall-through case: {:?}",
                     expr.node));
        }
    };
}

fn trans_rvalue_dps_unadjusted<'a>(
                               bcx: &'a Block<'a>,
                               expr: &ast::Expr,
                               dest: Dest)
                               -> &'a Block<'a> {
    let _icx = push_ctxt("trans_rvalue_dps_unadjusted");
    let tcx = bcx.tcx();

    match expr.node {
        ast::ExprParen(e) => {
            return trans_rvalue_dps_unadjusted(bcx, e, dest);
        }
        ast::ExprPath(_) | ast::ExprSelf => {
            return trans_def_dps_unadjusted(bcx, expr,
                                            bcx.def(expr.id), dest);
        }
        ast::ExprIf(cond, thn, els) => {
            return controlflow::trans_if(bcx, cond, thn, els, dest);
        }
        ast::ExprMatch(discr, ref arms) => {
            return _match::trans_match(bcx, expr, discr, *arms, dest);
        }
        ast::ExprBlock(blk) => {
            return base::with_scope(bcx,
                                    blk.info(),
                                    "block-expr body",
                                    |bcx| {
                controlflow::trans_block(bcx, blk, dest)
            });
        }
        ast::ExprStruct(_, ref fields, base) => {
            return trans_rec_or_struct(bcx, (*fields), base, expr.span, expr.id, dest);
        }
        ast::ExprTup(ref args) => {
            let repr = adt::represent_type(bcx.ccx(), expr_ty(bcx, expr));
            let numbered_fields: ~[(uint, @ast::Expr)] =
                args.iter().enumerate().map(|(i, arg)| (i, *arg)).collect();
            return trans_adt(bcx, repr, 0, numbered_fields, None, dest);
        }
        ast::ExprLit(@codemap::Spanned {node: ast::LitStr(s, _), ..}) => {
            return tvec::trans_lit_str(bcx, expr, s, dest);
        }
        ast::ExprVstore(contents, ast::ExprVstoreSlice) |
        ast::ExprVstore(contents, ast::ExprVstoreMutSlice) => {
            return tvec::trans_slice_vstore(bcx, expr, contents, dest);
        }
        ast::ExprVec(..) | ast::ExprRepeat(..) => {
            return tvec::trans_fixed_vstore(bcx, expr, expr, dest);
        }
        ast::ExprFnBlock(decl, body) |
        ast::ExprProc(decl, body) => {
            let expr_ty = expr_ty(bcx, expr);
            let sigil = ty::ty_closure_sigil(expr_ty);
            debug!("translating block function {} with type {}",
                   expr_to_str(expr, tcx.sess.intr()),
                   expr_ty.repr(tcx));
            return closure::trans_expr_fn(bcx, sigil, decl, body,
                                          expr.id, expr.id, dest);
        }
        ast::ExprDoBody(blk) => {
            return trans_into(bcx, blk, dest);
        }
        ast::ExprCall(f, ref args, _) => {
            return callee::trans_call(
                bcx, expr, f, callee::ArgExprs(*args), expr.id, dest);
        }
        ast::ExprMethodCall(callee_id, rcvr, _, _, ref args, _) => {
            return callee::trans_method_call(bcx,
                                             expr,
                                             callee_id,
                                             rcvr,
                                             callee::ArgExprs(*args),
                                             dest);
        }
        ast::ExprBinary(callee_id, _, lhs, rhs) => {
            // if not overloaded, would be RvalueDatumExpr
            return trans_overloaded_op(bcx,
                                       expr,
                                       callee_id,
                                       lhs,
                                       ~[rhs],
                                       expr_ty(bcx, expr),
                                       dest);
        }
        ast::ExprUnary(callee_id, _, subexpr) => {
            // if not overloaded, would be RvalueDatumExpr
            return trans_overloaded_op(bcx,
                                       expr,
                                       callee_id,
                                       subexpr,
                                       ~[],
                                       expr_ty(bcx, expr),
                                       dest);
        }
        ast::ExprIndex(callee_id, base, idx) => {
            // if not overloaded, would be RvalueDatumExpr
            return trans_overloaded_op(bcx,
                                       expr,
                                       callee_id,
                                       base,
                                       ~[idx],
                                       expr_ty(bcx, expr),
                                       dest);
        }
        ast::ExprCast(val, _) => {
            match ty::get(node_id_type(bcx, expr.id)).sty {
                ty::ty_trait(_, _, store, _, _) => {
                    return meth::trans_trait_cast(bcx, val, expr.id,
                                                  dest, store, true /* adjustments */);
                }
                _ => {
                    bcx.tcx().sess.span_bug(expr.span,
                                            "expr_cast of non-trait");
                }
            }
        }
        ast::ExprAssignOp(callee_id, op, dst, src) => {
            return trans_assign_op(bcx, expr, callee_id, op, dst, src);
        }
        ast::ExprBox(_, contents) => {
            // Special case for `Gc<T>` for now. The other case, for unique
            // pointers, is handled in `trans_rvalue_datum_unadjusted`.
            return trans_gc(bcx, expr, contents, dest)
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
    let ccx = bcx.ccx();

    let lldest = match dest {
        SaveIn(lldest) => lldest,
        Ignore => { return bcx; }
    };

    match def {
        ast::DefVariant(tid, vid, _) => {
            let variant_info = ty::enum_variant_with_id(ccx.tcx, tid, vid);
            if variant_info.args.len() > 0u {
                // N-ary variant.
                let fn_data = callee::trans_fn_ref(bcx, vid, ref_expr.id);
                Store(bcx, fn_data.llfn, lldest);
                return bcx;
            } else {
                // Nullary variant.
                let ty = expr_ty(bcx, ref_expr);
                let repr = adt::represent_type(ccx, ty);
                adt::trans_start_init(bcx, repr, lldest,
                                      variant_info.disr_val);
                return bcx;
            }
        }
        ast::DefStruct(def_id) => {
            let ty = expr_ty(bcx, ref_expr);
            match ty::get(ty).sty {
                ty::ty_struct(did, _) if ty::has_dtor(ccx.tcx, did) => {
                    let repr = adt::represent_type(ccx, ty);
                    adt::trans_start_init(bcx, repr, lldest, 0);
                }
                ty::ty_bare_fn(..) => {
                    let fn_data = callee::trans_fn_ref(bcx, def_id, ref_expr.id);
                    Store(bcx, fn_data.llfn, lldest);
                }
                _ => ()
            }
            return bcx;
        }
        _ => {
            bcx.tcx().sess.span_bug(ref_expr.span, format!(
                "Non-DPS def {:?} referened by {}",
                def, bcx.node_id_to_str(ref_expr.id)));
        }
    }
}

fn trans_def_datum_unadjusted<'a>(
                              bcx: &'a Block<'a>,
                              ref_expr: &ast::Expr,
                              def: ast::Def)
                              -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_def_datum_unadjusted");

    let fn_data = match def {
        ast::DefFn(did, _) |
        ast::DefStaticMethod(did, ast::FromImpl(_), _) => {
            callee::trans_fn_ref(bcx, did, ref_expr.id)
        }
        ast::DefStaticMethod(impl_did, ast::FromTrait(trait_did), _) => {
            meth::trans_static_method_callee(bcx,
                                             impl_did,
                                             trait_did,
                                             ref_expr.id)
        }
        _ => {
            bcx.tcx().sess.span_bug(ref_expr.span, format!(
                "Non-DPS def {:?} referened by {}",
                def, bcx.node_id_to_str(ref_expr.id)));
        }
    };

    let fn_ty = expr_ty(bcx, ref_expr);
    DatumBlock {
        bcx: bcx,
        datum: Datum {
            val: fn_data.llfn,
            ty: fn_ty,
            mode: ByValue
        }
    }
}

fn trans_lvalue_unadjusted<'a>(bcx: &'a Block<'a>, expr: &ast::Expr)
                           -> DatumBlock<'a> {
    /*!
     *
     * Translates an lvalue expression, always yielding a by-ref
     * datum.  Does not apply any adjustments. */

    let _icx = push_ctxt("trans_lval");
    let mut bcx = bcx;

    debug!("trans_lvalue(expr={})", bcx.expr_to_str(expr));
    let _indenter = indenter();

    return match expr.node {
        ast::ExprParen(e) => {
            trans_lvalue_unadjusted(bcx, e)
        }
        ast::ExprPath(_) | ast::ExprSelf => {
            trans_def_lvalue(bcx, expr, bcx.def(expr.id))
        }
        ast::ExprField(base, ident, _) => {
            trans_rec_field(bcx, base, ident)
        }
        ast::ExprIndex(_, base, idx) => {
            trans_index(bcx, expr, base, idx)
        }
        ast::ExprUnary(_, ast::UnDeref, base) => {
            let basedatum = unpack_datum!(bcx, trans_to_datum(bcx, base));
            basedatum.deref(bcx, expr, 0)
        }
        _ => {
            bcx.tcx().sess.span_bug(
                expr.span,
                format!("trans_lvalue reached fall-through case: {:?}",
                     expr.node));
        }
    };

    fn trans_rec_field<'a>(
                       bcx: &'a Block<'a>,
                       base: &ast::Expr,
                       field: ast::Ident)
                       -> DatumBlock<'a> {
        //! Translates `base.field`.

        let mut bcx = bcx;
        let _icx = push_ctxt("trans_rec_field");

        let base_datum = unpack_datum!(bcx, trans_to_datum(bcx, base));
        let repr = adt::represent_type(bcx.ccx(), base_datum.ty);
        with_field_tys(bcx.tcx(), base_datum.ty, None, |discr, field_tys| {
            let ix = ty::field_idx_strict(bcx.tcx(), field.name, field_tys);
            DatumBlock {
                datum: base_datum.get_element(bcx,
                                              field_tys[ix].mt.ty,
                                              ZeroMem,
                                              |srcval| {
                    adt::trans_field_ptr(bcx, repr, srcval, discr, ix)
                }),
                bcx: bcx
            }
        })
    }

    fn trans_index<'a>(
                   bcx: &'a Block<'a>,
                   index_expr: &ast::Expr,
                   base: &ast::Expr,
                   idx: &ast::Expr)
                   -> DatumBlock<'a> {
        //! Translates `base[idx]`.

        let _icx = push_ctxt("trans_index");
        let ccx = bcx.ccx();
        let mut bcx = bcx;

        let base_datum = unpack_datum!(bcx, trans_to_datum(bcx, base));

        // Translate index expression and cast to a suitable LLVM integer.
        // Rust is less strict than LLVM in this regard.
        let Result {bcx, val: ix_val} = trans_to_datum(bcx, idx).to_result();
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

        let (bcx, base, len) =
            base_datum.get_vec_base_and_len(bcx, index_expr.span, index_expr.id, 0);

        debug!("trans_index: base {}", bcx.val_to_str(base));
        debug!("trans_index: len {}", bcx.val_to_str(len));

        let bounds_check = ICmp(bcx, lib::llvm::IntUGE, ix_val, len);
        let expect = ccx.intrinsics.get_copy(&("llvm.expect.i1"));
        let expected = Call(bcx, expect, [bounds_check, C_i1(false)], []);
        let bcx = with_cond(bcx, expected, |bcx| {
            controlflow::trans_fail_bounds_check(bcx, index_expr.span, ix_val, len)
        });
        let elt = InBoundsGEP(bcx, base, [ix_val]);
        let elt = PointerCast(bcx, elt, vt.llunit_ty.ptr_to());
        return DatumBlock {
            bcx: bcx,
            datum: Datum {val: elt,
                          ty: vt.unit_ty,
                          mode: ByRef(ZeroMem)}
        };
    }

    fn trans_def_lvalue<'a>(
                        bcx: &'a Block<'a>,
                        ref_expr: &ast::Expr,
                        def: ast::Def)
                        -> DatumBlock<'a> {
        //! Translates a reference to a path.

        let _icx = push_ctxt("trans_def_lvalue");
        match def {
            ast::DefStatic(did, _) => {
                let const_ty = expr_ty(bcx, ref_expr);

                fn get_did(ccx: @CrateContext, did: ast::DefId)
                    -> ast::DefId {
                    if did.crate != ast::LOCAL_CRATE {
                        inline::maybe_instantiate_inline(ccx, did)
                    } else {
                        did
                    }
                }

                fn get_val(bcx: &Block, did: ast::DefId, const_ty: ty::t)
                           -> ValueRef {
                    // For external constants, we don't inline.
                    if did.crate == ast::LOCAL_CRATE {
                        // The LLVM global has the type of its initializer,
                        // which may not be equal to the enum's type for
                        // non-C-like enums.
                        let val = base::get_item_val(bcx.ccx(), did.node);
                        let pty = type_of::type_of(bcx.ccx(), const_ty).ptr_to();
                        PointerCast(bcx, val, pty)
                    } else {
                        {
                            let extern_const_values = bcx.ccx()
                                                         .extern_const_values
                                                         .borrow();
                            match extern_const_values.get().find(&did) {
                                None => {}  // Continue.
                                Some(llval) => {
                                    return *llval;
                                }
                            }
                        }

                        unsafe {
                            let llty = type_of::type_of(bcx.ccx(), const_ty);
                            let symbol = csearch::get_symbol(
                                bcx.ccx().sess.cstore,
                                did);
                            let llval = symbol.with_c_str(|buf| {
                                llvm::LLVMAddGlobal(bcx.ccx().llmod,
                                                    llty.to_ref(),
                                                    buf)
                            });
                            let mut extern_const_values =
                                bcx.ccx().extern_const_values.borrow_mut();
                            extern_const_values.get().insert(did, llval);
                            llval
                        }
                    }
                }

                let did = get_did(bcx.ccx(), did);
                let val = get_val(bcx, did, const_ty);
                DatumBlock {
                    bcx: bcx,
                    datum: Datum {val: val,
                                  ty: const_ty,
                                  mode: ByRef(ZeroMem)}
                }
            }
            _ => {
                DatumBlock {
                    bcx: bcx,
                    datum: trans_local_var(bcx, def)
                }
            }
        }
    }
}

pub fn trans_local_var(bcx: &Block, def: ast::Def) -> Datum {
    let _icx = push_ctxt("trans_local_var");

    return match def {
        ast::DefUpvar(nid, _, _, _) => {
            // Can't move upvars, so this is never a ZeroMemLastUse.
            let local_ty = node_id_type(bcx, nid);
            let llupvars = bcx.fcx.llupvars.borrow();
            match llupvars.get().find(&nid) {
                Some(&val) => {
                    Datum {
                        val: val,
                        ty: local_ty,
                        mode: ByRef(ZeroMem)
                    }
                }
                None => {
                    bcx.sess().bug(format!(
                        "trans_local_var: no llval for upvar {:?} found", nid));
                }
            }
        }
        ast::DefArg(nid, _) => {
            let llargs = bcx.fcx.llargs.borrow();
            take_local(bcx, llargs.get(), nid)
        }
        ast::DefLocal(nid, _) | ast::DefBinding(nid, _) => {
            let lllocals = bcx.fcx.lllocals.borrow();
            take_local(bcx, lllocals.get(), nid)
        }
        ast::DefSelf(nid, _) => {
            let self_info = match bcx.fcx.llself.get() {
                Some(self_info) => self_info,
                None => {
                    bcx.sess().bug(format!(
                        "trans_local_var: reference to self \
                         out of context with id {:?}", nid));
                }
            };

            debug!("def_self() reference, self_info.ty={}",
                   self_info.ty.repr(bcx.tcx()));

            self_info
        }
        _ => {
            bcx.sess().unimpl(format!(
                "unsupported def type in trans_local_var: {:?}", def));
        }
    };

    fn take_local(bcx: &Block,
                  table: &HashMap<ast::NodeId, Datum>,
                  nid: ast::NodeId) -> Datum {
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

// The optional node ID here is the node ID of the path identifying the enum
// variant in use. If none, this cannot possibly an enum variant (so, if it
// is and `node_id_opt` is none, this function fails).
pub fn with_field_tys<R>(
                      tcx: ty::ctxt,
                      ty: ty::t,
                      node_id_opt: Option<ast::NodeId>,
                      op: |ty::Disr, (&[ty::field])| -> R)
                      -> R {
    match ty::get(ty).sty {
        ty::ty_struct(did, ref substs) => {
            op(0, struct_fields(tcx, did, substs))
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
                    let opt_def = {
                        let def_map = tcx.def_map.borrow();
                        def_map.get().get_copy(&node_id)
                    };
                    match opt_def {
                        ast::DefVariant(enum_id, variant_id, _) => {
                            let variant_info = ty::enum_variant_with_id(
                                tcx, enum_id, variant_id);
                            op(variant_info.disr_val,
                               struct_fields(tcx, variant_id, substs))
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
        let mut need_base = vec::from_elem(field_tys.len(), true);

        let numbered_fields = fields.map(|field| {
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
        });
        let optbase = match base {
            Some(base_expr) => {
                let mut leftovers = ~[];
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
        trans_adt(bcx, repr, discr, numbered_fields, optbase, dest)
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
    fields: ~[(uint, ty::t)]
}

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
    let mut temp_cleanups = ~[];
    adt::trans_start_init(bcx, repr, addr, discr);
    for &(i, e) in fields.iter() {
        let dest = adt::trans_field_ptr(bcx, repr, addr, discr, i);
        let e_ty = expr_ty_adjusted(bcx, e);
        bcx = trans_into(bcx, e, SaveIn(dest));
        add_clean_temp_mem(bcx, dest, e_ty);
        temp_cleanups.push(dest);
    }
    for base in optbase.iter() {
        // FIXME #6573: is it sound to use the destination's repr on the base?
        // And, would it ever be reasonable to be here with discr != 0?
        let base_datum = unpack_datum!(bcx, trans_to_datum(bcx, base.expr));
        for &(i, t) in base.fields.iter() {
            let datum = base_datum.get_element(bcx, t, ZeroMem, |srcval| {
                adt::trans_field_ptr(bcx, repr, srcval, discr, i)
            });
            let dest = adt::trans_field_ptr(bcx, repr, addr, discr, i);
            bcx = datum.store_to(bcx, INIT, dest);
        }
    }

    for cleanup in temp_cleanups.iter() {
        revoke_clean(bcx, *cleanup);
    }
    return bcx;
}


fn trans_immediate_lit<'a>(
                       bcx: &'a Block<'a>,
                       expr: &ast::Expr,
                       lit: ast::Lit)
                       -> DatumBlock<'a> {
    // must not be a string constant, that is a RvalueDpsExpr
    let _icx = push_ctxt("trans_immediate_lit");
    let ty = expr_ty(bcx, expr);
    immediate_rvalue_bcx(bcx, consts::const_lit(bcx.ccx(), expr, lit), ty)
}

fn trans_unary_datum<'a>(
                     bcx: &'a Block<'a>,
                     un_expr: &ast::Expr,
                     op: ast::UnOp,
                     sub_expr: &ast::Expr)
                     -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_unary_datum");

    // if deref, would be LvalueExpr
    assert!(op != ast::UnDeref);

    // if overloaded, would be RvalueDpsExpr
    {
        let method_map = bcx.ccx().maps.method_map.borrow();
        assert!(!method_map.get().contains_key(&un_expr.id));
    }

    let un_ty = expr_ty(bcx, un_expr);
    let sub_ty = expr_ty(bcx, sub_expr);

    return match op {
        ast::UnNot => {
            let Result {bcx, val} = trans_to_datum(bcx, sub_expr).to_result();

            // If this is a boolean type, we must not use the LLVM Not
            // instruction, as that is a *bitwise* not and we want *logical*
            // not on our 8-bit boolean values.
            let llresult = match ty::get(un_ty).sty {
                ty::ty_bool => {
                    let llcond = ICmp(bcx,
                                      lib::llvm::IntEQ,
                                      val,
                                      C_bool(false));
                    Select(bcx, llcond, C_bool(true), C_bool(false))
                }
                _ => Not(bcx, val)
            };
            immediate_rvalue_bcx(bcx, llresult, un_ty)
        }
        ast::UnNeg => {
            let Result {bcx, val} = trans_to_datum(bcx, sub_expr).to_result();
            let llneg = {
                if ty::type_is_fp(un_ty) {
                    FNeg(bcx, val)
                } else {
                    Neg(bcx, val)
                }
            };
            immediate_rvalue_bcx(bcx, llneg, un_ty)
        }
        ast::UnBox => {
            trans_boxed_expr(bcx, un_ty, sub_expr, sub_ty, heap_managed)
        }
        ast::UnUniq => {
            let heap  = heap_for_unique(bcx, un_ty);
            trans_boxed_expr(bcx, un_ty, sub_expr, sub_ty, heap)
        }
        ast::UnDeref => {
            bcx.sess().bug("deref expressions should have been \
                            translated using trans_lvalue(), not \
                            trans_unary_datum()")
        }
    };
}

fn trans_boxed_expr<'a>(
                    bcx: &'a Block<'a>,
                    box_ty: ty::t,
                    contents: &ast::Expr,
                    contents_ty: ty::t,
                    heap: heap)
                    -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_boxed_expr");
    if heap == heap_exchange {
        let llty = type_of::type_of(bcx.ccx(), contents_ty);
        let size = llsize_of(bcx.ccx(), llty);
        let Result { bcx: bcx, val: val } = malloc_raw_dyn(bcx, contents_ty,
                                                           heap_exchange, size);
        add_clean_free(bcx, val, heap_exchange);
        let bcx = trans_into(bcx, contents, SaveIn(val));
        revoke_clean(bcx, val);
        return immediate_rvalue_bcx(bcx, val, box_ty);
    } else {
        let base::MallocResult {
            bcx,
            smart_ptr: bx,
            body
        } = base::malloc_general(bcx, contents_ty, heap);
        add_clean_free(bcx, bx, heap);
        let bcx = trans_into(bcx, contents, SaveIn(body));
        revoke_clean(bcx, bx);
        return immediate_rvalue_bcx(bcx, bx, box_ty);
    }
}

fn trans_addr_of<'a>(
                 bcx: &'a Block<'a>,
                 expr: &ast::Expr,
                 subexpr: &ast::Expr)
                 -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_addr_of");
    let mut bcx = bcx;
    let sub_datum = unpack_datum!(bcx, trans_to_datum(bcx, subexpr));
    let llval = sub_datum.to_ref_llval(bcx);
    return immediate_rvalue_bcx(bcx, llval, expr_ty(bcx, expr));
}

pub fn trans_gc<'a>(
                mut bcx: &'a Block<'a>,
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
    let contents_datum_block = trans_boxed_expr(bcx,
                                                box_ty,
                                                contents,
                                                contents_ty,
                                                heap_managed);
    bcx = contents_datum_block.bcx;
    bcx = contents_datum_block.datum.move_to(bcx, INIT, field_dest);

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
                     lhs_datum: &Datum,
                     rhs_datum: &Datum)
                     -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_eager_binop");

    let lhs = lhs_datum.to_appropriate_llval(bcx);
    let lhs_t = lhs_datum.ty;

    let rhs = rhs_datum.to_appropriate_llval(bcx);
    let rhs_t = rhs_datum.ty;

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
            C_bool(false)
        } else {
            if !ty::type_is_scalar(rhs_t) {
                bcx.tcx().sess.span_bug(binop_expr.span,
                                        "non-scalar comparison");
            }
            let cmpr = base::compare_scalar_types(bcx, lhs, rhs, rhs_t, op);
            bcx = cmpr.bcx;
            ZExt(bcx, cmpr.val, Type::i8())
        }
      }
      _ => {
        bcx.tcx().sess.span_bug(binop_expr.span, "unexpected binop");
      }
    };

    return immediate_rvalue_bcx(bcx, val, binop_ty);
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
                    -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_lazy_binop");
    let binop_ty = expr_ty(bcx, binop_expr);
    let bcx = bcx;

    let Result {bcx: past_lhs, val: lhs} = {
        base::with_scope_result(bcx, a.info(), "lhs", |bcx| {
            trans_to_datum(bcx, a).to_result()
        })
    };

    if past_lhs.unreachable.get() {
        return immediate_rvalue_bcx(past_lhs, lhs, binop_ty);
    }

    let join = base::sub_block(bcx, "join");
    let before_rhs = base::sub_block(bcx, "rhs");

    let lhs_i1 = bool_to_i1(past_lhs, lhs);
    match op {
      lazy_and => CondBr(past_lhs, lhs_i1, before_rhs.llbb, join.llbb),
      lazy_or => CondBr(past_lhs, lhs_i1, join.llbb, before_rhs.llbb)
    }

    let Result {bcx: past_rhs, val: rhs} = {
        base::with_scope_result(before_rhs, b.info(), "rhs", |bcx| {
            trans_to_datum(bcx, b).to_result()
        })
    };

    if past_rhs.unreachable.get() {
        return immediate_rvalue_bcx(join, lhs, binop_ty);
    }

    Br(past_rhs, join.llbb);
    let phi = Phi(join, Type::bool(), [lhs, rhs], [past_lhs.llbb,
                                               past_rhs.llbb]);

    return immediate_rvalue_bcx(join, phi, binop_ty);
}

fn trans_binary<'a>(
                bcx: &'a Block<'a>,
                binop_expr: &ast::Expr,
                op: ast::BinOp,
                lhs: &ast::Expr,
                rhs: &ast::Expr)
                -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_binary");

    match op {
        ast::BiAnd => {
            trans_lazy_binop(bcx, binop_expr, lazy_and, lhs, rhs)
        }
        ast::BiOr => {
            trans_lazy_binop(bcx, binop_expr, lazy_or, lhs, rhs)
        }
        _ => {
            let mut bcx = bcx;
            let lhs_datum = unpack_datum!(bcx, trans_to_datum(bcx, lhs));
            let rhs_datum = unpack_datum!(bcx, trans_to_datum(bcx, rhs));
            let binop_ty = expr_ty(bcx, binop_expr);
            trans_eager_binop(bcx, binop_expr, binop_ty, op,
                              &lhs_datum, &rhs_datum)
        }
    }
}

fn trans_overloaded_op<'a>(
                       bcx: &'a Block<'a>,
                       expr: &ast::Expr,
                       callee_id: ast::NodeId,
                       rcvr: &ast::Expr,
                       args: ~[@ast::Expr],
                       ret_ty: ty::t,
                       dest: Dest)
                       -> &'a Block<'a> {
    let origin = {
        let method_map = bcx.ccx().maps.method_map.borrow();
        method_map.get().get_copy(&expr.id)
    };
    let fty = node_id_type(bcx, callee_id);
    callee::trans_call_inner(bcx,
                             expr.info(),
                             fty,
                             ret_ty,
                             |bcx| {
                                meth::trans_method_callee(bcx,
                                                          callee_id,
                                                          rcvr,
                                                          origin)
                             },
                             callee::ArgExprs(args),
                             Some(dest),
                             DoAutorefArg).bcx
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

fn trans_imm_cast<'a>(bcx: &'a Block<'a>, expr: &ast::Expr, id: ast::NodeId)
                  -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_cast");
    let ccx = bcx.ccx();

    let t_out = node_id_type(bcx, id);

    let mut bcx = bcx;
    let llexpr = unpack_result!(bcx, trans_to_datum(bcx, expr).to_result());
    let ll_t_in = val_ty(llexpr);
    let t_in = expr_ty(bcx, expr);
    let ll_t_out = type_of::type_of(ccx, t_out);

    let k_in = cast_type_kind(t_in);
    let k_out = cast_type_kind(t_out);
    let s_in = k_in == cast_integral && ty::type_is_signed(t_in);

    let newval =
        match (k_in, k_out) {
            (cast_integral, cast_integral) => {
                int_cast(bcx, ll_t_out, ll_t_in, llexpr, s_in)
            }
            (cast_float, cast_float) => {
                float_cast(bcx, ll_t_out, ll_t_in, llexpr)
            }
            (cast_integral, cast_float) => {
                if s_in {
                    SIToFP(bcx, llexpr, ll_t_out)
                } else { UIToFP(bcx, llexpr, ll_t_out) }
            }
            (cast_float, cast_integral) => {
                if ty::type_is_signed(t_out) {
                    FPToSI(bcx, llexpr, ll_t_out)
                } else { FPToUI(bcx, llexpr, ll_t_out) }
            }
            (cast_integral, cast_pointer) => {
                IntToPtr(bcx, llexpr, ll_t_out)
            }
            (cast_pointer, cast_integral) => {
                PtrToInt(bcx, llexpr, ll_t_out)
            }
            (cast_pointer, cast_pointer) => {
                PointerCast(bcx, llexpr, ll_t_out)
            }
            (cast_enum, cast_integral) |
            (cast_enum, cast_float) => {
                let bcx = bcx;
                let repr = adt::represent_type(ccx, t_in);
                let llexpr_ptr;
                if type_is_immediate(ccx, t_in) {
                    llexpr_ptr = Alloca(bcx, ll_t_in, "");
                    Store(bcx, llexpr, llexpr_ptr);
                } else {
                    llexpr_ptr = llexpr;
                }
                let lldiscrim_a = adt::trans_get_discr(bcx, repr, llexpr_ptr, Some(Type::i64()));
                match k_out {
                    cast_integral => int_cast(bcx, ll_t_out,
                                              val_ty(lldiscrim_a),
                                              lldiscrim_a, true),
                    cast_float => SIToFP(bcx, lldiscrim_a, ll_t_out),
                    _ => ccx.sess.bug(format!("translating unsupported cast: \
                                           {} ({:?}) -> {} ({:?})",
                                           t_in.repr(ccx.tcx), k_in,
                                           t_out.repr(ccx.tcx), k_out))
                }
            }
            _ => ccx.sess.bug(format!("translating unsupported cast: \
                                   {} ({:?}) -> {} ({:?})",
                                   t_in.repr(ccx.tcx), k_in,
                                   t_out.repr(ccx.tcx), k_out))
        };
    return immediate_rvalue_bcx(bcx, newval, t_out);
}

fn trans_assign_op<'a>(
                   bcx: &'a Block<'a>,
                   expr: &ast::Expr,
                   callee_id: ast::NodeId,
                   op: ast::BinOp,
                   dst: &ast::Expr,
                   src: @ast::Expr)
                   -> &'a Block<'a> {
    let _icx = push_ctxt("trans_assign_op");
    let mut bcx = bcx;

    debug!("trans_assign_op(expr={})", bcx.expr_to_str(expr));

    // Evaluate LHS (destination), which should be an lvalue
    let dst_datum = unpack_datum!(bcx, trans_lvalue_unadjusted(bcx, dst));

    // A user-defined operator method
    let found = {
        let method_map = bcx.ccx().maps.method_map.borrow();
        method_map.get().find(&expr.id).is_some()
    };
    if found {
        // FIXME(#2528) evaluates the receiver twice!!
        let scratch = scratch_datum(bcx, dst_datum.ty, "__assign_op", false);
        let bcx = trans_overloaded_op(bcx,
                                      expr,
                                      callee_id,
                                      dst,
                                      ~[src],
                                      dst_datum.ty,
                                      SaveIn(scratch.val));
        return scratch.move_to_datum(bcx, DROP_EXISTING, dst_datum);
    }

    // Evaluate RHS (source)
    let src_datum = unpack_datum!(bcx, trans_to_datum(bcx, src));

    // Perform computation and store the result
    let result_datum =
        unpack_datum!(bcx,
                      trans_eager_binop(
                          bcx, expr, dst_datum.ty, op,
                          &dst_datum, &src_datum));
    return result_datum.copy_to_datum(bcx, DROP_EXISTING, dst_datum);
}

pub fn trans_log_level<'a>(bcx: &'a Block<'a>) -> DatumBlock<'a> {
    let _icx = push_ctxt("trans_log_level");
    let ccx = bcx.ccx();

    let (modpath, modname) = {
        let srccrate;
        {
            let external_srcs = ccx.external_srcs.borrow();
            srccrate = match external_srcs.get().find(&bcx.fcx.id) {
                Some(&src) => {
                    ccx.sess.cstore.get_crate_data(src.crate).name
                }
                None => ccx.link_meta.crateid.name.to_managed(),
            };
        };
        let mut modpath = ~[PathMod(ccx.sess.ident_of(srccrate))];
        for e in bcx.fcx.path.iter() {
            match *e {
                PathMod(_) => { modpath.push(*e) }
                _ => {}
            }
        }
        let modname = path_str(ccx.sess, modpath);
        (modpath, modname)
    };

    let module_data_exists;
    {
        let module_data = ccx.module_data.borrow();
        module_data_exists = module_data.get().contains_key(&modname);
    }

    let global = if module_data_exists {
        let mut module_data = ccx.module_data.borrow_mut();
        module_data.get().get_copy(&modname)
    } else {
        let s = link::mangle_internal_name_by_path_and_seq(
            ccx, modpath, "loglevel");
        let global;
        unsafe {
            global = s.with_c_str(|buf| {
                llvm::LLVMAddGlobal(ccx.llmod, Type::i32().to_ref(), buf)
            });
            llvm::LLVMSetGlobalConstant(global, False);
            llvm::LLVMSetInitializer(global, C_null(Type::i32()));
            lib::llvm::SetLinkage(global, lib::llvm::InternalLinkage);
        }
        {
            let mut module_data = ccx.module_data.borrow_mut();
            module_data.get().insert(modname, global);
            global
        }
    };

    return immediate_rvalue_bcx(bcx, Load(bcx, global), ty::mk_u32());
}

