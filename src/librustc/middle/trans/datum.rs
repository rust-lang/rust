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
 *
 * A `Datum` contains all the information you need to describe the LLVM
 * translation of a Rust value.  It describes where the value is stored,
 * what Rust type the value has, whether it is addressed by reference,
 * and so forth.
 *
 * The idea of a datum is that, to the extent possible, you should not
 * care about these details, but rather use the methods on the Datum
 * type to "do what you want to do".  For example, you can simply call
 * `copy_to()` or `move_to()` to copy or move the value into a new
 * home.
 *
 * # Datum location
 *
 * The primary two fields of a datum are the `val` and the `mode`.
 * The `val` is an LLVM value ref.  It may either *be the value* that
 * is being tracked, or it may be a *pointer to the value being
 * tracked*.  This is specified in the `mode` field, which can either
 * be `ByValue` or `ByRef`, respectively.  The (Rust) type of the
 * value stored in the datum is indicated in the field `ty`.
 *
 * Generally speaking, you probably do not want to access the `val` field
 * unless you know what mode the value is in.  Instead you should use one
 * of the following accessors:
 *
 * - `to_value_llval()` converts to by-value
 * - `to_ref_llval()` converts to by-ref, allocating a stack slot if necessary
 * - `to_appropriate_llval()` converts to by-value if this is an
 *   immediate type, by-ref otherwise.  This is particularly
 *   convenient for interfacing with the various code floating around
 *   that predates datums.
 *
 * # Datum cleanup styles
 *
 * Each datum carries with it an idea of how its value will be cleaned
 * up.  This is primarily determined by the mode: a `ByValue` datum
 * will always be cleaned up by revoking cleanup using
 * `revoke_clean()`, because there is no other option. By ref datums
 * can sometimes be cleaned up via `revoke_clean` (in particular,
 * by-ref datums that originated from rvalues), but sometimes they
 * must be zeroed. This is indicated by the `DatumCleanup`
 * parameter. Note that zeroing a by-ref datum *always works* to
 * cancel the cleanup, but using `revoke_clean` is preferable since
 * there is no runtime cost. Some older parts of the code (notably
 * `match_`, at least at the time of this writing) rely on this and
 * only use zeroing.
 *
 * # Copying, moving, and storing
 *
 * There are three methods for moving the value into a new
 * location:
 *
 * - `copy_to()` will copy the value into a new location, meaning that
 *    the value is first mem-copied and then the new location is "taken"
 *    via the take glue, in effect creating a deep clone.
 *
 * - `move_to()` will copy the value, meaning that the value is mem-copied
 *   into its new home and then the cleanup on the this datum is revoked.
 *   This is a "shallow" clone.  After `move_to()`, the current datum
 *   is invalid and should no longer be used.
 *
 * - `store_to()` either performs a copy or a move depending on the
 *   Rust type of the datum.
 *
 * # Scratch datum
 *
 * Sometimes you just need some temporary scratch space.  The
 * `scratch_datum()` function will yield you up a by-ref datum that
 * points into the stack.  It's your responsibility to ensure that
 * whatever you put in there gets cleaned up etc.
 *
 * # Other actions
 *
 * There are various other helper methods on Datum, such as `deref()`,
 * `get_base_and_len()` and so forth.  These are documented on the
 * methods themselves.  Most are only suitable for some types of
 * values. */


use lib;
use lib::llvm::ValueRef;
use middle::trans::adt;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::common;
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::tvec;
use middle::trans::type_of;
use middle::trans::write_guard;
use middle::ty;
use util::common::indenter;
use util::ppaux::ty_to_str;

use std::uint;
use syntax::ast;
use syntax::codemap::Span;
use syntax::parse::token::special_idents;

#[deriving(Eq)]
pub enum CopyAction {
    INIT,
    DROP_EXISTING
}

pub struct Datum {
    /// The llvm value.  This is either a pointer to the Rust value or
    /// the value itself, depending on `mode` below.
    val: ValueRef,

    /// The rust type of the value.
    ty: ty::t,

    /// Indicates whether this is by-ref or by-value.
    mode: DatumMode,
}

pub struct DatumBlock {
    bcx: @mut Block,
    datum: Datum,
}

#[deriving(Eq, IterBytes)]
pub enum DatumMode {
    /// `val` is a pointer to the actual value (and thus has type *T).
    /// The argument indicates how to cancel cleanup of this datum if
    /// the value is moved elsewhere, which can either be by zeroing
    /// the memory or by canceling a registered cleanup.
    ByRef(DatumCleanup),

    /// `val` is the actual value (*only used for immediates* like ints, ptrs)
    ByValue,
}

impl DatumMode {
    pub fn is_by_ref(&self) -> bool {
        match *self { ByRef(_) => true, ByValue => false }
    }

    pub fn is_by_value(&self) -> bool {
        match *self { ByRef(_) => false, ByValue => true }
    }
}

/// See `Datum cleanup styles` section at the head of this module.
#[deriving(Eq, IterBytes)]
pub enum DatumCleanup {
    RevokeClean,
    ZeroMem
}

pub fn immediate_rvalue(val: ValueRef, ty: ty::t) -> Datum {
    return Datum {val: val, ty: ty, mode: ByValue};
}

pub fn immediate_rvalue_bcx(bcx: @mut Block,
                            val: ValueRef,
                            ty: ty::t)
                         -> DatumBlock {
    return DatumBlock {bcx: bcx, datum: immediate_rvalue(val, ty)};
}

pub fn scratch_datum(bcx: @mut Block, ty: ty::t, name: &str, zero: bool) -> Datum {
    /*!
     * Allocates temporary space on the stack using alloca() and
     * returns a by-ref Datum pointing to it.  If `zero` is true, the
     * space will be zeroed when it is allocated; this is normally not
     * necessary, but in the case of automatic rooting in match
     * statements it is possible to have temporaries that may not get
     * initialized if a certain arm is not taken, so we must zero
     * them. You must arrange any cleanups etc yourself!
     */

    let llty = type_of::type_of(bcx.ccx(), ty);
    let scratch = alloca_maybe_zeroed(bcx, llty, name, zero);
    Datum { val: scratch, ty: ty, mode: ByRef(RevokeClean) }
}

pub fn appropriate_mode(tcx: ty::ctxt, ty: ty::t) -> DatumMode {
    /*!
     * Indicates the "appropriate" mode for this value,
     * which is either by ref or by value, depending
     * on whether type is immediate or not.
     */

    if ty::type_is_voidish(ty) {
        ByValue
    } else if ty::type_is_immediate(tcx, ty) {
        ByValue
    } else {
        ByRef(RevokeClean)
    }
}

impl Datum {
    pub fn store_to(&self,
                    bcx: @mut Block,
                    action: CopyAction,
                    dst: ValueRef)
                    -> @mut Block {
        /*!
         *
         * Stores this value into its final home.  This moves if
         * `id` is located in the move table, but copies otherwise.
         */

        if ty::type_moves_by_default(bcx.tcx(), self.ty) {
            self.move_to(bcx, action, dst)
        } else {
            self.copy_to(bcx, action, dst)
        }
    }

    pub fn store_to_dest(&self,
                         bcx: @mut Block,
                         dest: expr::Dest)
                         -> @mut Block {
        match dest {
            expr::Ignore => {
                return bcx;
            }
            expr::SaveIn(addr) => {
                return self.store_to(bcx, INIT, addr);
            }
        }
    }

    pub fn store_to_datum(&self,
                          bcx: @mut Block,
                          action: CopyAction,
                          datum: Datum)
                          -> @mut Block {
        debug!("store_to_datum(self=%s, action=%?, datum=%s)",
               self.to_str(bcx.ccx()), action, datum.to_str(bcx.ccx()));
        assert!(datum.mode.is_by_ref());
        self.store_to(bcx, action, datum.val)
    }

    pub fn move_to_datum(&self, bcx: @mut Block, action: CopyAction, datum: Datum)
                         -> @mut Block {
        assert!(datum.mode.is_by_ref());
        self.move_to(bcx, action, datum.val)
    }

    pub fn copy_to_datum(&self, bcx: @mut Block, action: CopyAction, datum: Datum)
                         -> @mut Block {
        assert!(datum.mode.is_by_ref());
        self.copy_to(bcx, action, datum.val)
    }

    pub fn copy_to(&self, bcx: @mut Block, action: CopyAction, dst: ValueRef)
                   -> @mut Block {
        /*!
         *
         * Copies the value into `dst`, which should be a pointer to a
         * memory location suitable for `self.ty`.  You PROBABLY want
         * `store_to()` instead, which will move if possible but copy if
         * neccessary. */

        let _icx = push_ctxt("copy_to");

        if ty::type_is_voidish(self.ty) {
            return bcx;
        }

        debug!("copy_to(self=%s, action=%?, dst=%s)",
               self.to_str(bcx.ccx()), action, bcx.val_to_str(dst));

        // Watch out for the case where we are writing the copying the
        // value into the same location we read it out from.  We want
        // to avoid the case where we drop the existing value, which
        // frees it, and then overwrite it with itself (which has been
        // freed).
        if action == DROP_EXISTING &&
            ty::type_needs_drop(bcx.tcx(), self.ty)
        {
            match self.mode {
                ByRef(_) => {
                    let cast = PointerCast(bcx, dst, val_ty(self.val));
                    let cmp = ICmp(bcx, lib::llvm::IntNE, cast, self.val);
                    do with_cond(bcx, cmp) |bcx| {
                        self.copy_to_no_check(bcx, action, dst)
                    }
                }
                ByValue => {
                    self.copy_to_no_check(bcx, action, dst)
                }
            }
        } else {
            self.copy_to_no_check(bcx, action, dst)
        }
    }

    pub fn copy_to_no_check(&self,
                            bcx: @mut Block,
                            action: CopyAction,
                            dst: ValueRef)
                            -> @mut Block {
        /*!
         *
         * A helper for `copy_to()` which does not check to see if we
         * are copying to/from the same value. */

        let _icx = push_ctxt("copy_to_no_check");
        let mut bcx = bcx;

        if action == DROP_EXISTING {
            bcx = glue::drop_ty(bcx, dst, self.ty);
        }

        match self.mode {
            ByValue => {
                Store(bcx, self.val, dst);
            }
            ByRef(_) => {
                memcpy_ty(bcx, dst, self.val, self.ty);
            }
        }

        return glue::take_ty(bcx, dst, self.ty);
    }

    // This works like copy_val, except that it deinitializes the source.
    // Since it needs to zero out the source, src also needs to be an lval.
    //
    pub fn move_to(&self, bcx: @mut Block, action: CopyAction, dst: ValueRef)
                   -> @mut Block {
        let _icx = push_ctxt("move_to");
        let mut bcx = bcx;

        debug!("move_to(self=%s, action=%?, dst=%s)",
               self.to_str(bcx.ccx()), action, bcx.val_to_str(dst));

        if ty::type_is_voidish(self.ty) {
            return bcx;
        }

        if action == DROP_EXISTING {
            bcx = glue::drop_ty(bcx, dst, self.ty);
        }

        match self.mode {
            ByRef(_) => {
                memcpy_ty(bcx, dst, self.val, self.ty);
            }
            ByValue => {
                Store(bcx, self.val, dst);
            }
        }

        self.cancel_clean(bcx);

        return bcx;
    }

    pub fn add_clean(&self, bcx: @mut Block) {
        /*!
         * Schedules this datum for cleanup in `bcx`.  The datum
         * must be an rvalue.
         */

        match self.mode {
            ByValue => {
                add_clean_temp_immediate(bcx, self.val, self.ty);
            }
            ByRef(RevokeClean) => {
                add_clean_temp_mem(bcx, self.val, self.ty);
            }
            ByRef(ZeroMem) => {
                bcx.tcx().sess.bug(
                    fmt!("Cannot add clean to a 'zero-mem' datum"));
            }
        }
    }

    pub fn cancel_clean(&self, bcx: @mut Block) {
        if ty::type_needs_drop(bcx.tcx(), self.ty) {
            match self.mode {
                ByValue |
                ByRef(RevokeClean) => {
                    revoke_clean(bcx, self.val);
                }
                ByRef(ZeroMem) => {
                    // Lvalues which potentially need to be dropped
                    // must be passed by ref, so that we can zero them
                    // out.
                    assert!(self.mode.is_by_ref());
                    zero_mem(bcx, self.val, self.ty);
                }
            }
        }
    }

    pub fn to_str(&self, ccx: &CrateContext) -> ~str {
        fmt!("Datum { val=%s, ty=%s, mode=%? }",
             ccx.tn.val_to_str(self.val),
             ty_to_str(ccx.tcx, self.ty),
             self.mode)
    }

    pub fn to_value_datum(&self, bcx: @mut Block) -> Datum {
        /*!
         *
         * Yields a by-value form of this datum.  This may involve
         * creation of a temporary stack slot.  The value returned by
         * this function is not separately rooted from this datum, so
         * it will not live longer than the current datum. */

        match self.mode {
            ByValue => *self,
            ByRef(_) => {
                Datum {val: self.to_value_llval(bcx), mode: ByValue,
                       ty: self.ty}
            }
        }
    }

    pub fn to_value_llval(&self, bcx: @mut Block) -> ValueRef {
        /*!
         *
         * Yields the value itself. */

        if ty::type_is_voidish(self.ty) {
            C_nil()
        } else {
            match self.mode {
                ByValue => self.val,
                ByRef(_) => {
                    if ty::type_is_bool(self.ty) {
                        LoadRangeAssert(bcx, self.val, 0, 2, lib::llvm::True)
                    } else {
                        Load(bcx, self.val)
                    }
                }
            }
        }
    }

    pub fn to_ref_datum(&self, bcx: @mut Block) -> Datum {
        /*!
         * Yields a by-ref form of this datum.  This may involve
         * creation of a temporary stack slot.  The value returned by
         * this function is not separately rooted from this datum, so
         * it will not live longer than the current datum.
         */

        match self.mode {
            ByRef(_) => *self,
            ByValue => {
                Datum {val: self.to_ref_llval(bcx), mode: ByRef(RevokeClean),
                       ty: self.ty}
            }
        }
    }

    pub fn to_ref_llval(&self, bcx: @mut Block) -> ValueRef {
        match self.mode {
            ByRef(_) => self.val,
            ByValue => {
                if ty::type_is_voidish(self.ty) {
                    C_null(type_of::type_of(bcx.ccx(), self.ty).ptr_to())
                } else {
                    let slot = alloc_ty(bcx, self.ty, "");
                    Store(bcx, self.val, slot);
                    slot
                }
            }
        }
    }

    pub fn to_zeroable_ref_llval(&self, bcx: @mut Block) -> ValueRef {
        /*!
         * Returns a by-ref llvalue that can be zeroed in order to
         * cancel cleanup. This is a kind of hokey bridge used
         * to adapt to the match code. Please don't use it for new code.
         */

        match self.mode {
            // All by-ref datums are zeroable, even if we *could* just
            // cancel the cleanup.
            ByRef(_) => self.val,

            // By value datums can't be zeroed (where would you store
            // the zero?) so we have to spill them. Add a temp cleanup
            // for this spilled value and cancel the cleanup on this
            // current value.
            ByValue => {
                let slot = self.to_ref_llval(bcx);
                self.cancel_clean(bcx);
                add_clean_temp_mem(bcx, slot, self.ty);
                slot
            }
        }
    }

    pub fn appropriate_mode(&self, tcx: ty::ctxt) -> DatumMode {
        /*! See the `appropriate_mode()` function */

        appropriate_mode(tcx, self.ty)
    }

    pub fn to_appropriate_llval(&self, bcx: @mut Block) -> ValueRef {
        /*!
         *
         * Yields an llvalue with the `appropriate_mode()`. */

        match self.appropriate_mode(bcx.tcx()) {
            ByValue => self.to_value_llval(bcx),
            ByRef(_) => self.to_ref_llval(bcx)
        }
    }

    pub fn to_appropriate_datum(&self, bcx: @mut Block) -> Datum {
        /*!
         *
         * Yields a datum with the `appropriate_mode()`. */

        match self.appropriate_mode(bcx.tcx()) {
            ByValue => self.to_value_datum(bcx),
            ByRef(_) => self.to_ref_datum(bcx)
        }
    }

    pub fn get_element(&self,
                       bcx: @mut Block,
                       ty: ty::t,
                       source: DatumCleanup,
                       gep: &fn(ValueRef) -> ValueRef)
                       -> Datum {
        let base_val = self.to_ref_llval(bcx);
        Datum {
            val: gep(base_val),
            mode: ByRef(source),
            ty: ty,
        }
    }

    pub fn drop_val(&self, bcx: @mut Block) -> @mut Block {
        if !ty::type_needs_drop(bcx.tcx(), self.ty) {
            return bcx;
        }

        return match self.mode {
            ByRef(_) => glue::drop_ty(bcx, self.val, self.ty),
            ByValue => glue::drop_ty_immediate(bcx, self.val, self.ty)
        };
    }

    pub fn box_body(&self, bcx: @mut Block) -> Datum {
        /*!
         *
         * This datum must represent an @T or ~T box.  Returns a new
         * by-ref datum of type T, pointing at the contents. */

        let (content_ty, header) = match ty::get(self.ty).sty {
            ty::ty_box(mt) => (mt.ty, true),
            ty::ty_uniq(mt) => (mt.ty, false),
            ty::ty_evec(_, ty::vstore_uniq) | ty::ty_estr(ty::vstore_uniq) => {
                let unit_ty = ty::sequence_element_type(bcx.tcx(), self.ty);
                let unboxed_vec_ty = ty::mk_mut_unboxed_vec(bcx.tcx(), unit_ty);
                (unboxed_vec_ty, true)
            }
            _ => {
                bcx.tcx().sess.bug(fmt!(
                    "box_body() invoked on non-box type %s",
                    ty_to_str(bcx.tcx(), self.ty)));
            }
        };

        if !header && !ty::type_contents(bcx.tcx(), content_ty).contains_managed() {
            let ptr = self.to_value_llval(bcx);
            let ty = type_of(bcx.ccx(), content_ty);
            let body = PointerCast(bcx, ptr, ty.ptr_to());
            Datum {val: body, ty: content_ty, mode: ByRef(ZeroMem)}
        } else { // has a header
            let ptr = self.to_value_llval(bcx);
            let body = opaque_box_body(bcx, content_ty, ptr);
            Datum {val: body, ty: content_ty, mode: ByRef(ZeroMem)}
        }
    }

    pub fn to_rptr(&self, bcx: @mut Block) -> Datum {
        //! Returns a new datum of region-pointer type containing the
        //! the same ptr as this datum (after converting to by-ref
        //! using `to_ref_llval()`).

        // Convert to ref, yielding lltype *T.  Then create a Rust
        // type &'static T (which translates to *T).  Construct new
        // result (which will be by-value).  Note that it is not
        // significant *which* region we pick here.
        let llval = self.to_ref_llval(bcx);
        let rptr_ty = ty::mk_imm_rptr(bcx.tcx(), ty::re_static,
                                      self.ty);
        Datum {val: llval, ty: rptr_ty, mode: ByValue}
    }

    /// bcx: Block wherein to generate insns.
    /// span: Location where deref occurs.
    /// expr_id: ID of deref expr.
    /// derefs: Number of times deref'd already.
    /// is_auto: If true, only deref if auto-derefable.
    pub fn try_deref(&self,
                     bcx: @mut Block,
                     span: Span,
                     expr_id: ast::NodeId,
                     derefs: uint,
                     is_auto: bool)
                     -> (Option<Datum>, @mut Block) {
        let ccx = bcx.ccx();

        debug!("try_deref(expr_id=%?, derefs=%?, is_auto=%b, self=%?)",
               expr_id, derefs, is_auto, self.to_str(bcx.ccx()));

        let bcx =
            write_guard::root_and_write_guard(
                self, bcx, span, expr_id, derefs);

        match ty::get(self.ty).sty {
            ty::ty_box(_) | ty::ty_uniq(_) => {
                return (Some(self.box_body(bcx)), bcx);
            }
            ty::ty_ptr(mt) => {
                if is_auto { // unsafe ptrs are not AUTO-derefable
                    return (None, bcx);
                } else {
                    return (Some(deref_ptr(bcx, self, mt.ty)), bcx);
                }
            }
            ty::ty_rptr(_, mt) => {
                return (Some(deref_ptr(bcx, self, mt.ty)), bcx);
            }
            ty::ty_enum(did, ref substs) => {
                // Check whether this enum is a newtype enum:
                let variants = ty::enum_variants(ccx.tcx, did);
                if (*variants).len() != 1 || variants[0].args.len() != 1 {
                    return (None, bcx);
                }

                let repr = adt::represent_type(ccx, self.ty);
                let ty = ty::subst(ccx.tcx, substs, variants[0].args[0]);
                return match self.mode {
                    ByRef(_) => {
                        // Recast lv.val as a pointer to the newtype
                        // rather than a ptr to the enum type.
                        (
                            Some(Datum {
                                val: adt::trans_field_ptr(bcx, repr, self.val,
                                                    0, 0),
                                ty: ty,
                                mode: ByRef(ZeroMem)
                            }),
                            bcx
                        )
                    }
                    ByValue => {
                        // Actually, this case cannot happen right
                        // now, because enums are never immediate.
                        assert!(ty::type_is_immediate(bcx.tcx(), ty));
                        (Some(Datum {ty: ty, ..*self}), bcx)
                    }
                };
            }
            ty::ty_struct(did, ref substs) => {
                // Check whether this struct is a newtype struct.
                let fields = ty::struct_fields(ccx.tcx, did, substs);
                if fields.len() != 1 || fields[0].ident !=
                    special_idents::unnamed_field {
                    return (None, bcx);
                }

                let repr = adt::represent_type(ccx, self.ty);
                let ty = fields[0].mt.ty;
                return match self.mode {
                    ByRef(_) => {
                        // Recast lv.val as a pointer to the newtype rather
                        // than a pointer to the struct type.
                        // FIXME #6572: This isn't correct for structs with
                        // destructors.
                        (
                            Some(Datum {
                                val: adt::trans_field_ptr(bcx, repr, self.val,
                                                    0, 0),
                                ty: ty,
                                mode: ByRef(ZeroMem)
                            }),
                            bcx
                        )
                    }
                    ByValue => {
                        assert!(ty::type_is_immediate(bcx.tcx(), ty));
                        (
                            Some(Datum {
                                val: ExtractValue(bcx, self.val, 0),
                                ty: ty,
                                mode: ByValue
                            }),
                            bcx
                        )
                    }
                }
            }
            _ => { // not derefable.
                return (None, bcx);
            }
        }

        fn deref_ptr(bcx: @mut Block, lv: &Datum, ty: ty::t) -> Datum {
            Datum {
                val: lv.to_value_llval(bcx),
                ty: ty,
                mode: ByRef(ZeroMem)
            }
        }
    }

    /// expr: The deref expression.
    pub fn deref(&self, bcx: @mut Block, expr: &ast::expr, derefs: uint)
                 -> DatumBlock {
        match self.try_deref(bcx, expr.span, expr.id, derefs, false) {
            (Some(lvres), bcx) => DatumBlock { bcx: bcx, datum: lvres },
            (None, _) => {
                bcx.ccx().sess.span_bug(expr.span,
                                        "Cannot deref this expression");
            }
        }
    }

    pub fn autoderef(&self,
                     bcx: @mut Block,
                     span: Span,
                     expr_id: ast::NodeId,
                     max: uint)
                     -> DatumBlock {
        let _icx = push_ctxt("autoderef");

        debug!("autoderef(expr_id=%d, max=%?, self=%?)",
               expr_id, max, self.to_str(bcx.ccx()));
        let _indenter = indenter();

        let mut datum = *self;
        let mut derefs = 0u;
        let mut bcx = bcx;
        while derefs < max {
            derefs += 1u;
            match datum.try_deref(bcx, span, expr_id, derefs, true) {
                (None, new_bcx) => { bcx = new_bcx; break }
                (Some(datum_deref), new_bcx) => {
                    datum = datum_deref;
                    bcx = new_bcx;
                }
            }
        }

        // either we were asked to deref a specific number of times,
        // in which case we should have, or we asked to deref as many
        // times as we can
        assert!(derefs == max || max == uint::max_value);
        DatumBlock { bcx: bcx, datum: datum }
    }

    pub fn get_vec_base_and_len(&self,
                                mut bcx: @mut Block,
                                span: Span,
                                expr_id: ast::NodeId,
                                derefs: uint)
                                -> (@mut Block, ValueRef, ValueRef) {
        //! Converts a vector into the slice pair. Performs rooting
        //! and write guards checks.

        // only imp't for @[] and @str, but harmless
        bcx = write_guard::root_and_write_guard(self, bcx, span, expr_id, derefs);
        let (base, len) = self.get_vec_base_and_len_no_root(bcx);
        (bcx, base, len)
    }

    pub fn get_vec_base_and_len_no_root(&self, bcx: @mut Block)
                                        -> (ValueRef, ValueRef) {
        //! Converts a vector into the slice pair. Des not root
        //! nor perform write guard checks.

        let llval = self.to_appropriate_llval(bcx);
        tvec::get_base_and_len(bcx, llval, self.ty)
    }

    pub fn root_and_write_guard(&self,
                                bcx: @mut Block,
                                span: Span,
                                expr_id: ast::NodeId,
                                derefs: uint)
                                -> @mut Block {
        write_guard::root_and_write_guard(self, bcx, span, expr_id, derefs)
    }

    pub fn to_result(&self, bcx: @mut Block) -> common::Result {
        rslt(bcx, self.to_appropriate_llval(bcx))
    }
}

impl DatumBlock {
    pub fn unpack(&self, bcx: &mut @mut Block) -> Datum {
        *bcx = self.bcx;
        return self.datum;
    }

    pub fn assert_by_ref(&self) -> DatumBlock {
        assert!(self.datum.mode.is_by_ref());
        *self
    }

    pub fn drop_val(&self) -> @mut Block {
        self.datum.drop_val(self.bcx)
    }

    pub fn store_to(&self,
                    action: CopyAction,
                    dst: ValueRef)
                    -> @mut Block {
        self.datum.store_to(self.bcx, action, dst)
    }

    pub fn copy_to(&self, action: CopyAction, dst: ValueRef) -> @mut Block {
        self.datum.copy_to(self.bcx, action, dst)
    }

    pub fn move_to(&self, action: CopyAction, dst: ValueRef) -> @mut Block {
        self.datum.move_to(self.bcx, action, dst)
    }

    pub fn to_value_llval(&self) -> ValueRef {
        self.datum.to_value_llval(self.bcx)
    }

    pub fn to_result(&self) -> common::Result {
        rslt(self.bcx, self.datum.to_appropriate_llval(self.bcx))
    }

    pub fn ccx(&self) -> @mut CrateContext {
        self.bcx.ccx()
    }

    pub fn tcx(&self) -> ty::ctxt {
        self.bcx.tcx()
    }

    pub fn to_str(&self) -> ~str {
        self.datum.to_str(self.ccx())
    }
}
