// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ## The Datum module
//!
//! A `Datum` encapsulates the result of evaluating a Rust expression.  It
//! contains a `ValueRef` indicating the result, a `Ty` describing
//! the Rust type, but also a *kind*. The kind indicates whether the datum
//! has cleanup scheduled (lvalue) or not (rvalue) and -- in the case of
//! rvalues -- whether or not the value is "by ref" or "by value".
//!
//! The datum API is designed to try and help you avoid memory errors like
//! forgetting to arrange cleanup or duplicating a value. The type of the
//! datum incorporates the kind, and thus reflects whether it has cleanup
//! scheduled:
//!
//! - `Datum<Lvalue>` -- by ref, cleanup scheduled
//! - `Datum<Rvalue>` -- by value or by ref, no cleanup scheduled
//! - `Datum<Expr>` -- either `Datum<Lvalue>` or `Datum<Rvalue>`
//!
//! Rvalue and expr datums are noncopyable, and most of the methods on
//! datums consume the datum itself (with some notable exceptions). This
//! reflects the fact that datums may represent affine values which ought
//! to be consumed exactly once, and if you were to try to (for example)
//! store an affine value multiple times, you would be duplicating it,
//! which would certainly be a bug.
//!
//! Some of the datum methods, however, are designed to work only on
//! copyable values such as ints or pointers. Those methods may borrow the
//! datum (`&self`) rather than consume it, but they always include
//! assertions on the type of the value represented to check that this
//! makes sense. An example is `shallow_copy()`, which duplicates
//! a datum value.
//!
//! Translating an expression always yields a `Datum<Expr>` result, but
//! the methods `to_[lr]value_datum()` can be used to coerce a
//! `Datum<Expr>` into a `Datum<Lvalue>` or `Datum<Rvalue>` as
//! needed. Coercing to an lvalue is fairly common, and generally occurs
//! whenever it is necessary to inspect a value and pull out its
//! subcomponents (for example, a match, or indexing expression). Coercing
//! to an rvalue is more unusual; it occurs when moving values from place
//! to place, such as in an assignment expression or parameter passing.
//!
//! ### Lvalues in detail
//!
//! An lvalue datum is one for which cleanup has been scheduled. Lvalue
//! datums are always located in memory, and thus the `ValueRef` for an
//! LLVM value is always a pointer to the actual Rust value. This means
//! that if the Datum has a Rust type of `int`, then the LLVM type of the
//! `ValueRef` will be `int*` (pointer to int).
//!
//! Because lvalues already have cleanups scheduled, the memory must be
//! zeroed to prevent the cleanup from taking place (presuming that the
//! Rust type needs drop in the first place, otherwise it doesn't
//! matter). The Datum code automatically performs this zeroing when the
//! value is stored to a new location, for example.
//!
//! Lvalues usually result from evaluating lvalue expressions. For
//! example, evaluating a local variable `x` yields an lvalue, as does a
//! reference to a field like `x.f` or an index `x[i]`.
//!
//! Lvalue datums can also arise by *converting* an rvalue into an lvalue.
//! This is done with the `to_lvalue_datum` method defined on
//! `Datum<Expr>`. Basically this method just schedules cleanup if the
//! datum is an rvalue, possibly storing the value into a stack slot first
//! if needed. Converting rvalues into lvalues occurs in constructs like
//! `&foo()` or `match foo() { ref x => ... }`, where the user is
//! implicitly requesting a temporary.
//!
//! ### Rvalues in detail
//!
//! Rvalues datums are values with no cleanup scheduled. One must be
//! careful with rvalue datums to ensure that cleanup is properly
//! arranged, usually by converting to an lvalue datum or by invoking the
//! `add_clean` method.
//!
//! ### Scratch datums
//!
//! Sometimes you need some temporary scratch space.  The functions
//! `[lr]value_scratch_datum()` can be used to get temporary stack
//! space. As their name suggests, they yield lvalues and rvalues
//! respectively. That is, the slot from `lvalue_scratch_datum` will have
//! cleanup arranged, and the slot from `rvalue_scratch_datum` does not.

pub use self::Expr::*;
pub use self::RvalueMode::*;

use llvm::ValueRef;
use trans::adt;
use trans::base::*;
use trans::build::{Load, Store};
use trans::common::*;
use trans::cleanup;
use trans::cleanup::{CleanupMethods, DropHintDatum, DropHintMethods};
use trans::expr;
use trans::tvec;
use middle::ty::Ty;

use std::fmt;
use syntax::ast;
use syntax::codemap::DUMMY_SP;

/// A `Datum` encapsulates the result of evaluating an expression.  It
/// describes where the value is stored, what Rust type the value has,
/// whether it is addressed by reference, and so forth. Please refer
/// the section on datums in `README.md` for more details.
#[derive(Clone, Copy, Debug)]
pub struct Datum<'tcx, K> {
    /// The llvm value.  This is either a pointer to the Rust value or
    /// the value itself, depending on `kind` below.
    pub val: ValueRef,

    /// The rust type of the value.
    pub ty: Ty<'tcx>,

    /// Indicates whether this is by-ref or by-value.
    pub kind: K,
}

pub struct DatumBlock<'blk, 'tcx: 'blk, K> {
    pub bcx: Block<'blk, 'tcx>,
    pub datum: Datum<'tcx, K>,
}

#[derive(Debug)]
pub enum Expr {
    /// a fresh value that was produced and which has no cleanup yet
    /// because it has not yet "landed" into its permanent home
    RvalueExpr(Rvalue),

    /// `val` is a pointer into memory for which a cleanup is scheduled
    /// (and thus has type *T). If you move out of an Lvalue, you must
    /// zero out the memory (FIXME #5016).
    LvalueExpr(Lvalue),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DropFlagInfo {
    DontZeroJustUse(ast::NodeId),
    ZeroAndMaintain(ast::NodeId),
    None,
}

impl DropFlagInfo {
    pub fn must_zero(&self) -> bool {
        match *self {
            DropFlagInfo::DontZeroJustUse(..) => false,
            DropFlagInfo::ZeroAndMaintain(..) => true,
            DropFlagInfo::None => true,
        }
    }

    pub fn hint_datum<'blk, 'tcx>(&self, bcx: Block<'blk, 'tcx>)
                              -> Option<DropHintDatum<'tcx>> {
        let id = match *self {
            DropFlagInfo::None => return None,
            DropFlagInfo::DontZeroJustUse(id) |
            DropFlagInfo::ZeroAndMaintain(id) => id,
        };

        let hints = bcx.fcx.lldropflag_hints.borrow();
        let retval = hints.hint_datum(id);
        assert!(retval.is_some(), "An id (={}) means must have a hint", id);
        retval
    }
}

// FIXME: having Lvalue be `Copy` is a bit of a footgun, since clients
// may not realize that subparts of an Lvalue can have a subset of
// drop-flags associated with them, while this as written will just
// memcpy the drop_flag_info. But, it is an easier way to get `_match`
// off the ground to just let this be `Copy` for now.
#[derive(Copy, Clone, Debug)]
pub struct Lvalue {
    pub source: &'static str,
    pub drop_flag_info: DropFlagInfo
}

#[derive(Debug)]
pub struct Rvalue {
    pub mode: RvalueMode
}

/// Classifies what action we should take when a value is moved away
/// with respect to its drop-flag.
///
/// Long term there will be no need for this classification: all flags
/// (which will be stored on the stack frame) will have the same
/// interpretation and maintenance code associated with them.
#[derive(Copy, Clone, Debug)]
pub enum HintKind {
    /// When the value is moved, set the drop-flag to "dropped"
    /// (i.e. "zero the flag", even when the specific representation
    /// is not literally 0) and when it is reinitialized, set the
    /// drop-flag back to "initialized".
    ZeroAndMaintain,

    /// When the value is moved, do not set the drop-flag to "dropped"
    /// However, continue to read the drop-flag in deciding whether to
    /// drop. (In essence, the path/fragment in question will never
    /// need to be dropped at the points where it is moved away by
    /// this code, but we are defending against the scenario where
    /// some *other* code could move away (or drop) the value and thus
    /// zero-the-flag, which is why we will still read from it.
    DontZeroJustUse,
}

impl Lvalue { // Constructors for various Lvalues.
    pub fn new<'blk, 'tcx>(source: &'static str) -> Lvalue {
        debug!("Lvalue at {} no drop flag info", source);
        Lvalue { source: source, drop_flag_info: DropFlagInfo::None }
    }

    pub fn new_dropflag_hint(source: &'static str) -> Lvalue {
        debug!("Lvalue at {} is drop flag hint", source);
        Lvalue { source: source, drop_flag_info: DropFlagInfo::None }
    }

    pub fn new_with_hint<'blk, 'tcx>(source: &'static str,
                                     bcx: Block<'blk, 'tcx>,
                                     id: ast::NodeId,
                                     k: HintKind) -> Lvalue {
        let (opt_id, info) = {
            let hint_available = Lvalue::has_dropflag_hint(bcx, id) &&
                bcx.tcx().sess.nonzeroing_move_hints();
            let info = match k {
                HintKind::ZeroAndMaintain if hint_available =>
                    DropFlagInfo::ZeroAndMaintain(id),
                HintKind::DontZeroJustUse if hint_available =>
                    DropFlagInfo::DontZeroJustUse(id),
                _ =>
                    DropFlagInfo::None,
            };
            (Some(id), info)
        };
        debug!("Lvalue at {}, id: {:?} info: {:?}", source, opt_id, info);
        Lvalue { source: source, drop_flag_info: info }
    }
} // end Lvalue constructor methods.

impl Lvalue {
    fn has_dropflag_hint<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     id: ast::NodeId) -> bool {
        let hints = bcx.fcx.lldropflag_hints.borrow();
        hints.has_hint(id)
    }
    pub fn dropflag_hint<'blk, 'tcx>(&self, bcx: Block<'blk, 'tcx>)
                                 -> Option<DropHintDatum<'tcx>> {
        self.drop_flag_info.hint_datum(bcx)
    }
}

impl Rvalue {
    pub fn new(m: RvalueMode) -> Rvalue {
        Rvalue { mode: m }
    }
}

// Make Datum linear for more type safety.
impl Drop for Rvalue {
    fn drop(&mut self) { }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum RvalueMode {
    /// `val` is a pointer to the actual value (and thus has type *T)
    ByRef,

    /// `val` is the actual value (*only used for immediates* like ints, ptrs)
    ByValue,
}

pub fn immediate_rvalue<'tcx>(val: ValueRef, ty: Ty<'tcx>) -> Datum<'tcx, Rvalue> {
    return Datum::new(val, ty, Rvalue::new(ByValue));
}

pub fn immediate_rvalue_bcx<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        val: ValueRef,
                                        ty: Ty<'tcx>)
                                        -> DatumBlock<'blk, 'tcx, Rvalue> {
    return DatumBlock::new(bcx, immediate_rvalue(val, ty))
}


/// Allocates temporary space on the stack using alloca() and returns a by-ref Datum pointing to
/// it. The memory will be dropped upon exit from `scope`. The callback `populate` should
/// initialize the memory.
pub fn lvalue_scratch_datum<'blk, 'tcx, A, F>(bcx: Block<'blk, 'tcx>,
                                              ty: Ty<'tcx>,
                                              name: &str,
                                              scope: cleanup::ScopeId,
                                              arg: A,
                                              populate: F)
                                              -> DatumBlock<'blk, 'tcx, Lvalue> where
    F: FnOnce(A, Block<'blk, 'tcx>, ValueRef) -> Block<'blk, 'tcx>,
{
    let scratch = alloc_ty(bcx, ty, name);

    // Subtle. Populate the scratch memory *before* scheduling cleanup.
    let bcx = populate(arg, bcx, scratch);
    bcx.fcx.schedule_drop_mem(scope, scratch, ty, None);

    DatumBlock::new(bcx, Datum::new(scratch, ty, Lvalue::new("datum::lvalue_scratch_datum")))
}

/// Allocates temporary space on the stack using alloca() and returns a by-ref Datum pointing to
/// it.  If `zero` is true, the space will be zeroed when it is allocated; this is normally not
/// necessary, but in the case of automatic rooting in match statements it is possible to have
/// temporaries that may not get initialized if a certain arm is not taken, so we must zero them.
/// You must arrange any cleanups etc yourself!
pub fn rvalue_scratch_datum<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        ty: Ty<'tcx>,
                                        name: &str)
                                        -> Datum<'tcx, Rvalue> {
    let scratch = alloc_ty(bcx, ty, name);
    call_lifetime_start(bcx, scratch);
    Datum::new(scratch, ty, Rvalue::new(ByRef))
}

/// Indicates the "appropriate" mode for this value, which is either by ref or by value, depending
/// on whether type is immediate or not.
pub fn appropriate_rvalue_mode<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                         ty: Ty<'tcx>) -> RvalueMode {
    if type_is_immediate(ccx, ty) {
        ByValue
    } else {
        ByRef
    }
}

fn add_rvalue_clean<'a, 'tcx>(mode: RvalueMode,
                              fcx: &FunctionContext<'a, 'tcx>,
                              scope: cleanup::ScopeId,
                              val: ValueRef,
                              ty: Ty<'tcx>) {
    match mode {
        ByValue => { fcx.schedule_drop_immediate(scope, val, ty); }
        ByRef => {
            fcx.schedule_lifetime_end(scope, val);
            fcx.schedule_drop_mem(scope, val, ty, None);
        }
    }
}

pub trait KindOps {

    /// Take appropriate action after the value in `datum` has been
    /// stored to a new location.
    fn post_store<'blk, 'tcx>(&self,
                              bcx: Block<'blk, 'tcx>,
                              val: ValueRef,
                              ty: Ty<'tcx>)
                              -> Block<'blk, 'tcx>;

    /// True if this mode is a reference mode, meaning that the datum's
    /// val field is a pointer to the actual value
    fn is_by_ref(&self) -> bool;

    /// Converts to an Expr kind
    fn to_expr_kind(self) -> Expr;

}

impl KindOps for Rvalue {
    fn post_store<'blk, 'tcx>(&self,
                              bcx: Block<'blk, 'tcx>,
                              _val: ValueRef,
                              _ty: Ty<'tcx>)
                              -> Block<'blk, 'tcx> {
        // No cleanup is scheduled for an rvalue, so we don't have
        // to do anything after a move to cancel or duplicate it.
        if self.is_by_ref() {
            call_lifetime_end(bcx, _val);
        }
        bcx
    }

    fn is_by_ref(&self) -> bool {
        self.mode == ByRef
    }

    fn to_expr_kind(self) -> Expr {
        RvalueExpr(self)
    }
}

impl KindOps for Lvalue {
    /// If an lvalue is moved, we must zero out the memory in which it resides so as to cancel
    /// cleanup. If an @T lvalue is copied, we must increment the reference count.
    fn post_store<'blk, 'tcx>(&self,
                              bcx: Block<'blk, 'tcx>,
                              val: ValueRef,
                              ty: Ty<'tcx>)
                              -> Block<'blk, 'tcx> {
        let _icx = push_ctxt("<Lvalue as KindOps>::post_store");
        if bcx.fcx.type_needs_drop(ty) {
            // cancel cleanup of affine values:
            // 1. if it has drop-hint, mark as moved; then code
            //    aware of drop-hint won't bother calling the
            //    drop-glue itself.
            if let Some(hint_datum) = self.drop_flag_info.hint_datum(bcx) {
                let moved_hint_byte = adt::DTOR_MOVED_HINT;
                let hint_llval = hint_datum.to_value().value();
                Store(bcx, C_u8(bcx.fcx.ccx, moved_hint_byte), hint_llval);
            }
            // 2. if the drop info says its necessary, drop-fill the memory.
            if self.drop_flag_info.must_zero() {
                let () = drop_done_fill_mem(bcx, val, ty);
            }
            bcx
        } else {
            // FIXME (#5016) would be nice to assert this, but we have
            // to allow for e.g. DontZeroJustUse flags, for now.
            //
            // (The dropflag hint construction should be taking
            // !type_needs_drop into account; earlier analysis phases
            // may not have all the info they need to include such
            // information properly, I think; in particular the
            // fragments analysis works on a non-monomorphized view of
            // the code.)
            //
            // assert_eq!(self.drop_flag_info, DropFlagInfo::None);
            bcx
        }
    }

    fn is_by_ref(&self) -> bool {
        true
    }

    fn to_expr_kind(self) -> Expr {
        LvalueExpr(self)
    }
}

impl KindOps for Expr {
    fn post_store<'blk, 'tcx>(&self,
                              bcx: Block<'blk, 'tcx>,
                              val: ValueRef,
                              ty: Ty<'tcx>)
                              -> Block<'blk, 'tcx> {
        match *self {
            LvalueExpr(ref l) => l.post_store(bcx, val, ty),
            RvalueExpr(ref r) => r.post_store(bcx, val, ty),
        }
    }

    fn is_by_ref(&self) -> bool {
        match *self {
            LvalueExpr(ref l) => l.is_by_ref(),
            RvalueExpr(ref r) => r.is_by_ref()
        }
    }

    fn to_expr_kind(self) -> Expr {
        self
    }
}

impl<'tcx> Datum<'tcx, Rvalue> {
    /// Schedules a cleanup for this datum in the given scope. That means that this datum is no
    /// longer an rvalue datum; hence, this function consumes the datum and returns the contained
    /// ValueRef.
    pub fn add_clean<'a>(self,
                         fcx: &FunctionContext<'a, 'tcx>,
                         scope: cleanup::ScopeId)
                         -> ValueRef {
        add_rvalue_clean(self.kind.mode, fcx, scope, self.val, self.ty);
        self.val
    }

    /// Returns an lvalue datum (that is, a by ref datum with cleanup scheduled). If `self` is not
    /// already an lvalue, cleanup will be scheduled in the temporary scope for `expr_id`.
    pub fn to_lvalue_datum_in_scope<'blk>(self,
                                          bcx: Block<'blk, 'tcx>,
                                          name: &str,
                                          scope: cleanup::ScopeId)
                                          -> DatumBlock<'blk, 'tcx, Lvalue> {
        let fcx = bcx.fcx;

        match self.kind.mode {
            ByRef => {
                add_rvalue_clean(ByRef, fcx, scope, self.val, self.ty);
                DatumBlock::new(bcx, Datum::new(
                    self.val,
                    self.ty,
                    Lvalue::new("datum::to_lvalue_datum_in_scope")))
            }

            ByValue => {
                lvalue_scratch_datum(
                    bcx, self.ty, name, scope, self,
                    |this, bcx, llval| {
                        call_lifetime_start(bcx, llval);
                        let bcx = this.store_to(bcx, llval);
                        bcx.fcx.schedule_lifetime_end(scope, llval);
                        bcx
                    })
            }
        }
    }

    pub fn to_ref_datum<'blk>(self, bcx: Block<'blk, 'tcx>)
                              -> DatumBlock<'blk, 'tcx, Rvalue> {
        let mut bcx = bcx;
        match self.kind.mode {
            ByRef => DatumBlock::new(bcx, self),
            ByValue => {
                let scratch = rvalue_scratch_datum(bcx, self.ty, "to_ref");
                bcx = self.store_to(bcx, scratch.val);
                DatumBlock::new(bcx, scratch)
            }
        }
    }

    pub fn to_appropriate_datum<'blk>(self, bcx: Block<'blk, 'tcx>)
                                      -> DatumBlock<'blk, 'tcx, Rvalue> {
        match self.appropriate_rvalue_mode(bcx.ccx()) {
            ByRef => {
                self.to_ref_datum(bcx)
            }
            ByValue => {
                match self.kind.mode {
                    ByValue => DatumBlock::new(bcx, self),
                    ByRef => {
                        let llval = load_ty(bcx, self.val, self.ty);
                        call_lifetime_end(bcx, self.val);
                        DatumBlock::new(bcx, Datum::new(llval, self.ty, Rvalue::new(ByValue)))
                    }
                }
            }
        }
    }
}

/// Methods suitable for "expr" datums that could be either lvalues or
/// rvalues. These include coercions into lvalues/rvalues but also a number
/// of more general operations. (Some of those operations could be moved to
/// the more general `impl<K> Datum<K>`, but it's convenient to have them
/// here since we can `match self.kind` rather than having to implement
/// generic methods in `KindOps`.)
impl<'tcx> Datum<'tcx, Expr> {
    fn match_kind<R, F, G>(self, if_lvalue: F, if_rvalue: G) -> R where
        F: FnOnce(Datum<'tcx, Lvalue>) -> R,
        G: FnOnce(Datum<'tcx, Rvalue>) -> R,
    {
        let Datum { val, ty, kind } = self;
        match kind {
            LvalueExpr(l) => if_lvalue(Datum::new(val, ty, l)),
            RvalueExpr(r) => if_rvalue(Datum::new(val, ty, r)),
        }
    }

    /// Asserts that this datum *is* an lvalue and returns it.
    #[allow(dead_code)] // potentially useful
    pub fn assert_lvalue(self, bcx: Block) -> Datum<'tcx, Lvalue> {
        self.match_kind(
            |d| d,
            |_| bcx.sess().bug("assert_lvalue given rvalue"))
    }

    pub fn store_to_dest<'blk>(self,
                               bcx: Block<'blk, 'tcx>,
                               dest: expr::Dest,
                               expr_id: ast::NodeId)
                               -> Block<'blk, 'tcx> {
        match dest {
            expr::Ignore => {
                self.add_clean_if_rvalue(bcx, expr_id);
                bcx
            }
            expr::SaveIn(addr) => {
                self.store_to(bcx, addr)
            }
        }
    }

    /// Arranges cleanup for `self` if it is an rvalue. Use when you are done working with a value
    /// that may need drop.
    pub fn add_clean_if_rvalue<'blk>(self,
                                     bcx: Block<'blk, 'tcx>,
                                     expr_id: ast::NodeId) {
        self.match_kind(
            |_| { /* Nothing to do, cleanup already arranged */ },
            |r| {
                let scope = cleanup::temporary_scope(bcx.tcx(), expr_id);
                r.add_clean(bcx.fcx, scope);
            })
    }

    pub fn to_lvalue_datum<'blk>(self,
                                 bcx: Block<'blk, 'tcx>,
                                 name: &str,
                                 expr_id: ast::NodeId)
                                 -> DatumBlock<'blk, 'tcx, Lvalue> {
        debug!("to_lvalue_datum self: {}", self.to_string(bcx.ccx()));

        self.match_kind(
            |l| DatumBlock::new(bcx, l),
            |r| {
                let scope = cleanup::temporary_scope(bcx.tcx(), expr_id);
                r.to_lvalue_datum_in_scope(bcx, name, scope)
            })
    }

    /// Ensures that we have an rvalue datum (that is, a datum with no cleanup scheduled).
    pub fn to_rvalue_datum<'blk>(self,
                                 bcx: Block<'blk, 'tcx>,
                                 name: &'static str)
                                 -> DatumBlock<'blk, 'tcx, Rvalue> {
        self.match_kind(
            |l| {
                let mut bcx = bcx;
                match l.appropriate_rvalue_mode(bcx.ccx()) {
                    ByRef => {
                        let scratch = rvalue_scratch_datum(bcx, l.ty, name);
                        bcx = l.store_to(bcx, scratch.val);
                        DatumBlock::new(bcx, scratch)
                    }
                    ByValue => {
                        let v = load_ty(bcx, l.val, l.ty);
                        bcx = l.kind.post_store(bcx, l.val, l.ty);
                        DatumBlock::new(bcx, Datum::new(v, l.ty, Rvalue::new(ByValue)))
                    }
                }
            },
            |r| DatumBlock::new(bcx, r))
    }

}

/// Methods suitable only for lvalues. These include the various
/// operations to extract components out of compound data structures,
/// such as extracting the field from a struct or a particular element
/// from an array.
impl<'tcx> Datum<'tcx, Lvalue> {
    /// Converts a datum into a by-ref value. The datum type must be one which is always passed by
    /// reference.
    pub fn to_llref(self) -> ValueRef {
        self.val
    }

    // Extracts a component of a compound data structure (e.g., a field from a
    // struct). Note that if self is an opened, unsized type then the returned
    // datum may also be unsized _without the size information_. It is the
    // callers responsibility to package the result in some way to make a valid
    // datum in that case (e.g., by making a fat pointer or opened pair).
    pub fn get_element<'blk, F>(&self, bcx: Block<'blk, 'tcx>, ty: Ty<'tcx>,
                                gep: F)
                                -> Datum<'tcx, Lvalue> where
        F: FnOnce(adt::MaybeSizedValue) -> ValueRef,
    {
        let val = if type_is_sized(bcx.tcx(), self.ty) {
            let val = adt::MaybeSizedValue::sized(self.val);
            gep(val)
        } else {
            let val = adt::MaybeSizedValue::unsized_(
                Load(bcx, expr::get_dataptr(bcx, self.val)),
                Load(bcx, expr::get_meta(bcx, self.val)));
            gep(val)
        };
        Datum {
            val: val,
            kind: Lvalue::new("Datum::get_element"),
            ty: ty,
        }
    }

    pub fn get_vec_base_and_len<'blk>(&self, bcx: Block<'blk, 'tcx>)
                                      -> (ValueRef, ValueRef) {
        //! Converts a vector into the slice pair.

        tvec::get_base_and_len(bcx, self.val, self.ty)
    }
}

/// Generic methods applicable to any sort of datum.
impl<'tcx, K: KindOps + fmt::Debug> Datum<'tcx, K> {
    pub fn new(val: ValueRef, ty: Ty<'tcx>, kind: K) -> Datum<'tcx, K> {
        Datum { val: val, ty: ty, kind: kind }
    }

    pub fn to_expr_datum(self) -> Datum<'tcx, Expr> {
        let Datum { val, ty, kind } = self;
        Datum { val: val, ty: ty, kind: kind.to_expr_kind() }
    }

    /// Moves or copies this value into a new home, as appropriate depending on the type of the
    /// datum. This method consumes the datum, since it would be incorrect to go on using the datum
    /// if the value represented is affine (and hence the value is moved).
    pub fn store_to<'blk>(self,
                          bcx: Block<'blk, 'tcx>,
                          dst: ValueRef)
                          -> Block<'blk, 'tcx> {
        self.shallow_copy_raw(bcx, dst);

        self.kind.post_store(bcx, self.val, self.ty)
    }

    /// Helper function that performs a shallow copy of this value into `dst`, which should be a
    /// pointer to a memory location suitable for `self.ty`. `dst` should contain uninitialized
    /// memory (either newly allocated, zeroed, or dropped).
    ///
    /// This function is private to datums because it leaves memory in an unstable state, where the
    /// source value has been copied but not zeroed. Public methods are `store_to` (if you no
    /// longer need the source value) or `shallow_copy` (if you wish the source value to remain
    /// valid).
    fn shallow_copy_raw<'blk>(&self,
                              bcx: Block<'blk, 'tcx>,
                              dst: ValueRef)
                              -> Block<'blk, 'tcx> {
        let _icx = push_ctxt("copy_to_no_check");

        if type_is_zero_size(bcx.ccx(), self.ty) {
            return bcx;
        }

        if self.kind.is_by_ref() {
            memcpy_ty(bcx, dst, self.val, self.ty);
        } else {
            store_ty(bcx, self.val, dst, self.ty);
        }

        return bcx;
    }

    /// Copies the value into a new location. This function always preserves the existing datum as
    /// a valid value. Therefore, it does not consume `self` and, also, cannot be applied to affine
    /// values (since they must never be duplicated).
    pub fn shallow_copy<'blk>(&self,
                              bcx: Block<'blk, 'tcx>,
                              dst: ValueRef)
                              -> Block<'blk, 'tcx> {
        /*!
         * Copies the value into a new location. This function always
         * preserves the existing datum as a valid value. Therefore,
         * it does not consume `self` and, also, cannot be applied to
         * affine values (since they must never be duplicated).
         */

        assert!(!self.ty
                     .moves_by_default(&bcx.tcx().empty_parameter_environment(), DUMMY_SP));
        self.shallow_copy_raw(bcx, dst)
    }

    #[allow(dead_code)] // useful for debugging
    pub fn to_string<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> String {
        format!("Datum({}, {:?}, {:?})",
                ccx.tn().val_to_string(self.val),
                self.ty,
                self.kind)
    }

    /// See the `appropriate_rvalue_mode()` function
    pub fn appropriate_rvalue_mode<'a>(&self, ccx: &CrateContext<'a, 'tcx>)
                                       -> RvalueMode {
        appropriate_rvalue_mode(ccx, self.ty)
    }

    /// Converts `self` into a by-value `ValueRef`. Consumes this datum (i.e., absolves you of
    /// responsibility to cleanup the value). For this to work, the value must be something
    /// scalar-ish (like an int or a pointer) which (1) does not require drop glue and (2) is
    /// naturally passed around by value, and not by reference.
    pub fn to_llscalarish<'blk>(self, bcx: Block<'blk, 'tcx>) -> ValueRef {
        assert!(!bcx.fcx.type_needs_drop(self.ty));
        assert!(self.appropriate_rvalue_mode(bcx.ccx()) == ByValue);
        if self.kind.is_by_ref() {
            load_ty(bcx, self.val, self.ty)
        } else {
            self.val
        }
    }

    pub fn to_llbool<'blk>(self, bcx: Block<'blk, 'tcx>) -> ValueRef {
        assert!(self.ty.is_bool());
        self.to_llscalarish(bcx)
    }
}

impl<'blk, 'tcx, K> DatumBlock<'blk, 'tcx, K> {
    pub fn new(bcx: Block<'blk, 'tcx>, datum: Datum<'tcx, K>)
               -> DatumBlock<'blk, 'tcx, K> {
        DatumBlock { bcx: bcx, datum: datum }
    }
}

impl<'blk, 'tcx, K: KindOps + fmt::Debug> DatumBlock<'blk, 'tcx, K> {
    pub fn to_expr_datumblock(self) -> DatumBlock<'blk, 'tcx, Expr> {
        DatumBlock::new(self.bcx, self.datum.to_expr_datum())
    }
}

impl<'blk, 'tcx> DatumBlock<'blk, 'tcx, Expr> {
    pub fn store_to_dest(self,
                         dest: expr::Dest,
                         expr_id: ast::NodeId) -> Block<'blk, 'tcx> {
        let DatumBlock { bcx, datum } = self;
        datum.store_to_dest(bcx, dest, expr_id)
    }

    pub fn to_llbool(self) -> Result<'blk, 'tcx> {
        let DatumBlock { datum, bcx } = self;
        Result::new(bcx, datum.to_llbool(bcx))
    }
}
