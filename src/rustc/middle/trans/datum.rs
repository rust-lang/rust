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
 * unless you know what mode the value is in.  Intead you should use one
 * of the following accessors:
 *
 * - `to_value_llval()` converts to by-value
 * - `to_ref_llval()` converts to by-ref, allocating a stack slot if necessary
 * - `to_appropriate_llval()` converts to by-value if this is an
 *   immediate type, by-ref otherwise.  This is particularly
 *   convenient for interfacing with the various code floating around
 *   that predates datums.
 *
 * # Datum sources
 *
 * Each datum carries with it an idea of its "source".  This indicates
 * the kind of expression from which the datum originated.  The source
 * affects what happens when the datum is stored or moved.
 *
 * There are three options:
 *
 * 1. `FromRvalue`: This value originates from some temporary rvalue.
 *    This is therefore the owning reference to the datum.  If the
 *    datum is stored, then, it will be *moved* into its new home.
 *    Furthermore, we will not zero out the datum but rather use
 *    `revoke_clean()` to cancel any cleanup.
 *
 * 2. `FromLvalue`: This value originates from an lvalue.  If the datum
 *    is stored, it will be *copied* into its new home.  If the datum
 *    is moved, it will be zeroed out.
 *
 * 3. `FromLastUseLvalue`: The same as FromLvalue, except that it
 *    originates from the *last use* of an lvalue.  If the datum is
 *    stored, then, it will be moved (and zeroed out).
 *
 * # Storing, copying, and moving
 *
 * There are three kinds of methods for moving the value into a new
 * location.  *Storing* a datum is probably the one you want to reach
 * for first: it is used when you will no longer use the datum and
 * would like to place it somewhere.  It may translate to a copy or a
 * move, depending on the source of the datum.  After a store, the
 * datum may or may not be usable anymore, so you must assume it is
 * not.
 *
 * Sometimes, though, you want to use an explicit copy or move.  A
 * copy copies the data from the datum into a new location and
 * executes the take glue on that location, thus leaving the datum
 * valid for further use.  Moving, in contrast, copies the data into
 * the new location and then cancels any cleanups on the current datum
 * (as appropriate for the source).  No glue code is executed.  After
 * a move, the datum is no longer usable.
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

use lib::llvm::ValueRef;
use base::*;
use common::*;
use build::*;
use util::ppaux::ty_to_str;
use util::common::indenter;

enum CopyAction {
    INIT,
    DROP_EXISTING
}

struct Datum {
    /// The llvm value.  This is either a pointer to the Rust value or
    /// the value itself, depending on `mode` below.
    val: ValueRef,

    /// The rust type of the value.
    ty: ty::t,

    /// Indicates whether this is by-ref or by-value.
    mode: DatumMode,

    /// How did this value originate?  This is particularly important
    /// if the value is MOVED or prematurely DROPPED, because it
    /// describes how to cancel the cleanup that was scheduled before.
    /// See the def'n of the `DatumSource` type.
    source: DatumSource
}

struct DatumBlock {
    bcx: block,
    datum: Datum,
}

enum DatumMode {
    /// `val` is a pointer to the actual value (and thus has type *T)
    ByRef,

    /// `val` is the actual value (*only used for immediates* like ints, ptrs)
    ByValue,
}

impl DatumMode {
    fn is_by_ref() -> bool {
        match self { ByRef => true, ByValue => false }
    }

    fn is_by_value() -> bool {
        match self { ByRef => false, ByValue => true }
    }
}

/// See `Datum Sources` section at the head of this module.
enum DatumSource {
    FromRvalue,
    FromLvalue,
    FromLastUseLvalue,
}

impl DatumSource {
    fn is_rvalue() -> bool {
        match self {
            FromRvalue => true,
            FromLvalue | FromLastUseLvalue => false
        }
    }

    fn is_any_lvalue() -> bool {
        match self {
            FromRvalue => false,
            FromLvalue | FromLastUseLvalue => true
        }
    }
}

fn immediate_rvalue(val: ValueRef, ty: ty::t) -> Datum {
    return Datum {val: val, ty: ty,
                  mode: ByValue, source: FromRvalue};
}

fn immediate_rvalue_bcx(bcx: block, val: ValueRef, ty: ty::t) -> DatumBlock {
    return DatumBlock {bcx: bcx, datum: immediate_rvalue(val, ty)};
}

fn scratch_datum(bcx: block, ty: ty::t, zero: bool) -> Datum {
    /*!
     *
     * Allocates temporary space on the stack using alloca() and
     * returns a by-ref Datum pointing to it.  You must arrange
     * any cleanups etc yourself! */

    let llty = type_of::type_of(bcx.ccx(), ty);
    let scratch = alloca_maybe_zeroed(bcx, llty, zero);
    Datum { val: scratch, ty: ty, mode: ByRef, source: FromRvalue }
}

impl Datum {
    fn store_will_move() -> bool {
        match self.source {
            FromRvalue | FromLastUseLvalue => true,
            FromLvalue => false
        }
    }

    fn store_to(bcx: block, action: CopyAction, dst: ValueRef) -> block {
        /*!
         *
         * Stores this value into its final home.  This moves if
         * possible, but copies otherwise. */

        if self.store_will_move() {
            self.move_to(bcx, action, dst)
        } else {
            self.copy_to(bcx, action, dst)
        }
    }

    fn store_to_dest(bcx: block, dest: expr::Dest) -> block {
        match dest {
            expr::Ignore => {
                return bcx;
            }
            expr::SaveIn(addr) => {
                return self.store_to(bcx, INIT, addr);
            }
        }
    }

    fn store_to_datum(bcx: block, action: CopyAction, datum: Datum) -> block {
        debug!("store_to_datum(self=%s, action=%?, datum=%s)",
               self.to_str(bcx.ccx()), action, datum.to_str(bcx.ccx()));
        assert datum.mode.is_by_ref();
        self.store_to(bcx, action, datum.val)
    }

    fn move_to_datum(bcx: block, action: CopyAction, datum: Datum) -> block {
        assert datum.mode.is_by_ref();
        self.move_to(bcx, action, datum.val)
    }

    fn copy_to_datum(bcx: block, action: CopyAction, datum: Datum) -> block {
        assert datum.mode.is_by_ref();
        self.copy_to(bcx, action, datum.val)
    }

    fn copy_to(bcx: block, action: CopyAction, dst: ValueRef) -> block {
        /*!
         *
         * Copies the value into `dst`, which should be a pointer to a
         * memory location suitable for `self.ty`.  You PROBABLY want
         * `store_to()` instead, which will move if possible but copy if
         * neccessary. */

        let _icx = bcx.insn_ctxt("copy_to");

        if ty::type_is_nil(self.ty) || ty::type_is_bot(self.ty) {
            return bcx;
        }

        debug!("copy_to(self=%s, action=%?, dst=%s)",
               self.to_str(bcx.ccx()), action, bcx.val_str(dst));

        // Watch out for the case where we are writing the copying the
        // value into the same location we read it out from.  We want
        // to avoid the case where we drop the existing value, which
        // frees it, and then overwrite it with itself (which has been
        // freed).
        if action == DROP_EXISTING &&
            ty::type_needs_drop(bcx.tcx(), self.ty)
        {
            match self.mode {
                ByRef => {
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

    fn copy_to_no_check(bcx: block, action: CopyAction,
                        dst: ValueRef) -> block
    {
        /*!
         *
         * A helper for `copy_to()` which does not check to see if we
         * are copying to/from the same value. */

        let _icx = bcx.insn_ctxt("copy_to_no_check");
        let mut bcx = bcx;

        if action == DROP_EXISTING {
            bcx = glue::drop_ty(bcx, dst, self.ty);
        }

        match self.mode {
            ByValue => {
                Store(bcx, self.val, dst);
            }
            ByRef => {
                memmove_ty(bcx, dst, self.val, self.ty);
            }
        }

        return glue::take_ty(bcx, dst, self.ty);
    }

    // This works like copy_val, except that it deinitializes the source.
    // Since it needs to zero out the source, src also needs to be an lval.
    //
    // FIXME (#839): We always zero out the source. Ideally we would
    // detect the case where a variable is always deinitialized by
    // block exit and thus doesn't need to be dropped.
    fn move_to(bcx: block, action: CopyAction, dst: ValueRef) -> block {
        let _icx = bcx.insn_ctxt("move_to");
        let mut bcx = bcx;

        debug!("move_to(self=%s, action=%?, dst=%s)",
               self.to_str(bcx.ccx()), action, bcx.val_str(dst));

        if ty::type_is_nil(self.ty) || ty::type_is_bot(self.ty) {
            return bcx;
        }

        if action == DROP_EXISTING {
            bcx = glue::drop_ty(bcx, dst, self.ty);
        }

        match self.mode {
            ByRef => {
                glue::memmove_ty(bcx, dst, self.val, self.ty);
            }
            ByValue => {
                Store(bcx, self.val, dst);
            }
        }

        self.cancel_clean(bcx);

        return bcx;
    }

    fn add_clean(bcx: block) {
        /*!
         *
         * Schedules this datum for cleanup in `bcx`.  The datum
         * must be an rvalue. */

        assert self.source.is_rvalue();
        match self.mode {
            ByValue => {
                add_clean_temp_immediate(bcx, self.val, self.ty);
            }
            ByRef => {
                add_clean_temp_mem(bcx, self.val, self.ty);
            }
        }
    }

    fn cancel_clean(bcx: block) {
        if ty::type_needs_drop(bcx.tcx(), self.ty) {
            match self.source {
                FromRvalue => {
                    revoke_clean(bcx, self.val);
                }
                FromLvalue | FromLastUseLvalue => {
                    // Lvalues which potentially need to be dropped
                    // must be passed by ref, so that we can zero them
                    // out.
                    assert self.mode.is_by_ref();
                    zero_mem(bcx, self.val, self.ty);
                }
            }
        }
    }

    fn to_str(ccx: &crate_ctxt) -> ~str {
        fmt!("Datum { val=%s, ty=%s, mode=%?, source=%? }",
             val_str(ccx.tn, self.val),
             ty_to_str(ccx.tcx, self.ty),
             self.mode,
             self.source)
    }

    fn to_value_datum(bcx: block) -> Datum {
        /*!
         *
         * Yields a by-ref form of this datum.  This may involve
         * creation of a temporary stack slot.  The value returned by
         * this function is not separately rooted from this datum, so
         * it will not live longer than the current datum. */

        match self.mode {
            ByValue => self,
            ByRef => {
                Datum {val: self.to_value_llval(bcx), mode: ByValue,
                       ty: self.ty, source: FromRvalue}
            }
        }
    }

    fn to_value_llval(bcx: block) -> ValueRef {
        /*!
         *
         * Yields the value itself. */

        if ty::type_is_nil(self.ty) || ty::type_is_bot(self.ty) {
            C_nil()
        } else {
            match self.mode {
                ByValue => self.val,
                ByRef => Load(bcx, self.val)
            }
        }
    }

    fn to_ref_datum(bcx: block) -> Datum {
        /*!
         *
         * Yields a by-ref form of this datum.  This may involve
         * creation of a temporary stack slot.  The value returned by
         * this function is not separately rooted from this datum, so
         * it will not live longer than the current datum. */

        match self.mode {
            ByRef => self,
            ByValue => {
                Datum {val: self.to_ref_llval(bcx), mode: ByRef,
                       ty: self.ty, source: FromRvalue}
            }
        }
    }

    fn to_ref_llval(bcx: block) -> ValueRef {
        match self.mode {
            ByRef => self.val,
            ByValue => {
                if ty::type_is_nil(self.ty) || ty::type_is_bot(self.ty) {
                    C_null(T_ptr(type_of::type_of(bcx.ccx(), self.ty)))
                } else {
                    let slot = alloc_ty(bcx, self.ty);
                    Store(bcx, self.val, slot);
                    slot
                }
            }
        }
    }

    fn appropriate_mode() -> DatumMode {
        /*!
         *
         * Indicates the "appropriate" mode for this value,
         * which is either by ref or by value, depending
         * on whether type is iimmediate or what. */

        if ty::type_is_nil(self.ty) || ty::type_is_bot(self.ty) {
            ByValue
        } else if ty::type_is_immediate(self.ty) {
            ByValue
        } else {
            ByRef
        }
    }

    fn to_appropriate_llval(bcx: block) -> ValueRef {
        /*!
         *
         * Yields an llvalue with the `appropriate_mode()`. */

        match self.appropriate_mode() {
            ByValue => self.to_value_llval(bcx),
            ByRef => self.to_ref_llval(bcx)
        }
    }

    fn to_appropriate_datum(bcx: block) -> Datum {
        /*!
         *
         * Yields a datum with the `appropriate_mode()`. */

        match self.appropriate_mode() {
            ByValue => self.to_value_datum(bcx),
            ByRef => self.to_ref_datum(bcx)
        }
    }

    fn GEPi(bcx: block, ixs: &[uint], ty: ty::t) -> Datum {
        let base_val = self.to_ref_llval(bcx);
        Datum {
            val: GEPi(bcx, base_val, ixs),
            mode: ByRef,
            ty: ty,
            source: FromLvalue
        }
    }

    fn root(bcx: block, scope_id: ast::node_id) {
        /*!
         *
         * In some cases, borrowck will decide that an @T/@[]/@str
         * value must be rooted for the program to be safe.  In that
         * case, we will call this function, which will stash a copy
         * away until we exit the scope `scope_id`. */

        debug!("root(scope_id=%?, self=%?)",
               scope_id, self.to_str(bcx.ccx()));

        if bcx.sess().trace() {
            trans_trace(
                bcx, None,
                fmt!("preserving until end of scope %d", scope_id));
        }

        let scratch = scratch_datum(bcx, self.ty, true);
        self.copy_to_datum(bcx, INIT, scratch);
        base::add_root_cleanup(bcx, scope_id, scratch.val, scratch.ty);
    }

    fn drop_val(bcx: block) -> block {
        if !ty::type_needs_drop(bcx.tcx(), self.ty) {
            return bcx;
        }

        return match self.mode {
            ByRef => glue::drop_ty(bcx, self.val, self.ty),
            ByValue => glue::drop_ty_immediate(bcx, self.val, self.ty)
        };
    }

    fn box_body(bcx: block) -> Datum {
        /*!
         *
         * This datum must represent an @T or ~T box.  Returns a new
         * by-ref datum of type T, pointing at the contents. */

        let content_ty = match ty::get(self.ty).sty {
            ty::ty_box(mt) | ty::ty_uniq(mt) => mt.ty,
            _ => {
                bcx.tcx().sess.bug(fmt!(
                    "box_body() invoked on non-box type %s",
                    ty_to_str(bcx.tcx(), self.ty)));
            }
        };

        let ptr = self.to_value_llval(bcx);
        let body = opaque_box_body(bcx, content_ty, ptr);
        Datum {val: body, ty: content_ty, mode: ByRef, source: FromLvalue}
    }

    fn to_rptr(bcx: block) -> Datum {
        //!
        //
        // Returns a new datum of region-pointer type containing the
        // the same ptr as this datum (after converting to by-ref
        // using `to_ref_llval()`).

        // Convert to ref, yielding lltype *T.  Then create a Rust
        // type &static/T (which translates to *T).  Construct new
        // result (which will be by-value).  Note that it is not
        // significant *which* region we pick here.
        let llval = self.to_ref_llval(bcx);
        let rptr_ty = ty::mk_imm_rptr(bcx.tcx(), ty::re_static,
                                      self.ty);
        Datum {val: llval, ty: rptr_ty,
               mode: ByValue, source: FromRvalue}
    }

    fn try_deref(
        bcx: block,            // block wherein to generate insn's
        expr_id: ast::node_id, // id of expr being deref'd
        derefs: uint,          // number of times deref'd already
        is_auto: bool)         // if true, only deref if auto-derefable
        -> Option<Datum>
    {
        let ccx = bcx.ccx();

        debug!("try_deref(expr_id=%d, derefs=%?, is_auto=%b, self=%?)",
               expr_id, derefs, is_auto, self.to_str(bcx.ccx()));
        let _indenter = indenter();

        // root the autoderef'd value, if necessary:
        //
        // (Note: root'd values are always boxes)
        match ccx.maps.root_map.find({id:expr_id, derefs:derefs}) {
            None => (),
            Some(scope_id) => {
                self.root(bcx, scope_id);
            }
        }

        match ty::get(self.ty).sty {
            ty::ty_box(_) | ty::ty_uniq(_) => {
                return Some(self.box_body(bcx));
            }
            ty::ty_ptr(mt) => {
                if is_auto { // unsafe ptrs are not AUTO-derefable
                    return None;
                } else {
                    return Some(deref_ptr(bcx, &self, mt.ty));
                }
            }
            ty::ty_rptr(_, mt) => {
                return Some(deref_ptr(bcx, &self, mt.ty));
            }
            ty::ty_enum(did, ref substs) => {
                // Check whether this enum is a newtype enum:
                let variants = ty::enum_variants(ccx.tcx, did);
                if (*variants).len() != 1u || variants[0].args.len() != 1u {
                    return None;
                }

                let ty = ty::subst(ccx.tcx, substs, variants[0].args[0]);
                return match self.mode {
                    ByRef => {
                        // Recast lv.val as a pointer to the newtype
                        // rather than a ptr to the enum type.
                        let llty = T_ptr(type_of::type_of(ccx, ty));
                        Some(Datum {
                            val: PointerCast(bcx, self.val, llty),
                            ty: ty,
                            mode: ByRef,
                            source: FromLvalue
                        })
                    }
                    ByValue => {
                        // Actually, this case cannot happen right
                        // now, because enums are never immediate.
                        // But in principle newtype'd immediate
                        // values should be immediate, and in that
                        // case the * would be a no-op except for
                        // changing the type, so I am putting this
                        // code in place here to do the right
                        // thing if this change ever goes through.
                        assert ty::type_is_immediate(ty);
                        Some(Datum {ty: ty, ..self})
                    }
                };
            }
            _ => { // not derefable.
                return None;
            }
        }

        fn deref_ptr(bcx: block, lv: &Datum, ty: ty::t) -> Datum {
            Datum {
                val: lv.to_value_llval(bcx),
                ty: ty,
                mode: ByRef,
                source: FromLvalue // *p is an lvalue
            }
        }
    }

    fn deref(bcx: block,
             expr: @ast::expr,  // the expression whose value is being deref'd
             derefs: uint) -> Datum {
        match self.try_deref(bcx, expr.id, derefs, false) {
            Some(lvres) => lvres,
            None => {
                bcx.ccx().sess.span_bug(
                    expr.span, ~"Cannot deref this expression");
            }
        }
    }

    fn autoderef(bcx: block,
                 expr_id: ast::node_id,
                 max: uint) -> Datum {
        let _icx = bcx.insn_ctxt("autoderef");

        debug!("autoderef(expr_id=%d, max=%?, self=%?)",
               expr_id, max, self.to_str(bcx.ccx()));
        let _indenter = indenter();

        let mut datum = self;
        let mut derefs = 0u;
        while derefs < max {
            derefs += 1u;
            match datum.try_deref(bcx, expr_id, derefs, true) {
                None => break,
                Some(datum_deref) => {
                    datum = datum_deref;
                }
            }
        }

        // either we were asked to deref a specific number of times,
        // in which case we should have, or we asked to deref as many
        // times as we can
        assert derefs == max || max == uint::max_value;
        datum
    }

    fn get_base_and_len(bcx: block) -> (ValueRef, ValueRef) {
        tvec::get_base_and_len(bcx, self.to_appropriate_llval(bcx), self.ty)
    }

    fn to_result(bcx: block) -> common::Result {
        rslt(bcx, self.to_appropriate_llval(bcx))
    }
}

impl DatumBlock {
    fn unpack(bcx: &mut block) -> Datum {
        *bcx = self.bcx;
        return self.datum;
    }

    fn assert_by_ref() -> DatumBlock {
        assert self.datum.mode.is_by_ref();
        self
    }

    fn drop_val() -> block {
        self.datum.drop_val(self.bcx)
    }

    fn store_to(action: CopyAction, dst: ValueRef) -> block {
        self.datum.store_to(self.bcx, action, dst)
    }

    fn copy_to(action: CopyAction, dst: ValueRef) -> block {
        self.datum.copy_to(self.bcx, action, dst)
    }

    fn move_to(action: CopyAction, dst: ValueRef) -> block {
        self.datum.move_to(self.bcx, action, dst)
    }

    fn to_value_llval() -> ValueRef {
        self.datum.to_value_llval(self.bcx)
    }

    fn to_result() -> common::Result {
        rslt(self.bcx, self.datum.to_appropriate_llval(self.bcx))
    }

    fn ccx() -> @crate_ctxt {
        self.bcx.ccx()
    }

    fn tcx() -> ty::ctxt {
        self.bcx.tcx()
    }

    fn to_str() -> ~str {
        self.datum.to_str(self.ccx())
    }
}

impl CopyAction : cmp::Eq {
    pure fn eq(&&other: CopyAction) -> bool {
        match (self, other) {
            (INIT, INIT) => true,
            (DROP_EXISTING, DROP_EXISTING) => true,
            (INIT, _) => false,
            (DROP_EXISTING, _) => false,
        }
    }
    pure fn ne(&&other: CopyAction) -> bool { !self.eq(other) }
}
