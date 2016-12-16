// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ## The Cleanup module
//!
//! The cleanup module tracks what values need to be cleaned up as scopes
//! are exited, either via panic or just normal control flow.
//!
//! Cleanup items can be scheduled into any of the scopes on the stack.
//! Typically, when a scope is finished, we generate the cleanup code. This
//! corresponds to a normal exit from a block (for example, an expression
//! completing evaluation successfully without panic).

use llvm::{BasicBlockRef, ValueRef};
use base::{self, Lifetime};
use common;
use common::{BlockAndBuilder, FunctionContext, Funclet};
use glue;
use type_::Type;
use value::Value;
use rustc::ty::Ty;

pub struct CleanupScope<'tcx> {
    // Cleanup to run upon scope exit.
    cleanup: DropValue<'tcx>,

    // Computed on creation if compiling with landing pads (!sess.no_landing_pads)
    pub landing_pad: Option<BasicBlockRef>,
}

#[derive(Copy, Clone)]
pub struct DropValue<'tcx> {
    val: ValueRef,
    ty: Ty<'tcx>,
    skip_dtor: bool,
}

impl<'tcx> DropValue<'tcx> {
    fn trans<'blk>(&self, funclet: Option<&'blk Funclet>, bcx: &BlockAndBuilder<'blk, 'tcx>) {
        glue::call_drop_glue(bcx, self.val, self.ty, self.skip_dtor, funclet)
    }
}

#[derive(Copy, Clone, Debug)]
enum UnwindKind {
    LandingPad,
    CleanupPad(ValueRef),
}

impl UnwindKind {
    /// Generates a branch going from `bcx` to `to_llbb` where `self` is
    /// the exit label attached to the start of `bcx`.
    ///
    /// Transitions from an exit label to other exit labels depend on the type
    /// of label. For example with MSVC exceptions unwind exit labels will use
    /// the `cleanupret` instruction instead of the `br` instruction.
    fn branch(&self, bcx: &BlockAndBuilder, to_llbb: BasicBlockRef) {
        match *self {
            UnwindKind::CleanupPad(pad) => {
                bcx.cleanup_ret(pad, Some(to_llbb));
            }
            UnwindKind::LandingPad => {
                bcx.br(to_llbb);
            }
        }
    }

    fn get_funclet(&self, bcx: &BlockAndBuilder) -> Option<Funclet> {
        match *self {
            UnwindKind::CleanupPad(_) => {
                let pad = bcx.cleanup_pad(None, &[]);
                Funclet::msvc(pad)
            },
            UnwindKind::LandingPad => Funclet::gnu(),
        }
    }
}

impl PartialEq for UnwindKind {
    fn eq(&self, label: &UnwindKind) -> bool {
        match (*self, *label) {
            (UnwindKind::LandingPad, UnwindKind::LandingPad) |
            (UnwindKind::CleanupPad(..), UnwindKind::CleanupPad(..)) => true,
            _ => false,
        }
    }
}
impl<'blk, 'tcx> FunctionContext<'blk, 'tcx> {
    pub fn trans_scope(
        &self,
        bcx: &BlockAndBuilder<'blk, 'tcx>,
        custom_scope: Option<CleanupScope<'tcx>>
    ) {
        if let Some(scope) = custom_scope {
            scope.cleanup.trans(None, &bcx);
        }
    }

    /// Schedules a (deep) drop of `val`, which is a pointer to an instance of
    /// `ty`
    pub fn schedule_drop_mem(&self, val: ValueRef, ty: Ty<'tcx>) -> Option<CleanupScope<'tcx>> {
        if !self.type_needs_drop(ty) { return None; }
        let drop = DropValue {
            val: val,
            ty: ty,
            skip_dtor: false,
        };

        debug!("schedule_drop_mem(val={:?}, ty={:?}) skip_dtor={}", Value(val), ty, drop.skip_dtor);

        Some(CleanupScope::new(self, drop))
    }

    /// Issue #23611: Schedules a (deep) drop of the contents of
    /// `val`, which is a pointer to an instance of struct/enum type
    /// `ty`. The scheduled code handles extracting the discriminant
    /// and dropping the contents associated with that variant
    /// *without* executing any associated drop implementation.
    pub fn schedule_drop_adt_contents(&self, val: ValueRef, ty: Ty<'tcx>)
        -> Option<CleanupScope<'tcx>> {
        // `if` below could be "!contents_needs_drop"; skipping drop
        // is just an optimization, so sound to be conservative.
        if !self.type_needs_drop(ty) { return None; }

        let drop = DropValue {
            val: val,
            ty: ty,
            skip_dtor: true,
        };

        debug!("schedule_drop_adt_contents(val={:?}, ty={:?}) skip_dtor={}",
               Value(val), ty, drop.skip_dtor);

        Some(CleanupScope::new(self, drop))
    }

}

impl<'tcx> CleanupScope<'tcx> {
    fn new<'a>(fcx: &FunctionContext<'a, 'tcx>, drop_val: DropValue<'tcx>) -> CleanupScope<'tcx> {
        CleanupScope {
            cleanup: drop_val,
            landing_pad: if !fcx.ccx.sess().no_landing_pads() {
                Some(CleanupScope::get_landing_pad(fcx, &drop_val))
            } else {
                None
            },
        }
    }

    /// Creates a landing pad for the top scope, if one does not exist. The
    /// landing pad will perform all cleanups necessary for an unwind and then
    /// `resume` to continue error propagation:
    ///
    ///     landing_pad -> ... cleanups ... -> [resume]
    ///
    /// (The cleanups and resume instruction are created by
    /// `trans_cleanups_to_exit_scope()`, not in this function itself.)
    fn get_landing_pad<'a>(fcx: &FunctionContext<'a, 'tcx>, drop_val: &DropValue<'tcx>)
        -> BasicBlockRef {
        debug!("get_landing_pad");

        let mut pad_bcx = fcx.build_new_block("unwind_custom_");

        let llpersonality = pad_bcx.fcx().eh_personality();

        let val = if base::wants_msvc_seh(fcx.ccx.sess()) {
            // A cleanup pad requires a personality function to be specified, so
            // we do that here explicitly (happens implicitly below through
            // creation of the landingpad instruction). We then create a
            // cleanuppad instruction which has no filters to run cleanup on all
            // exceptions.
            pad_bcx.set_personality_fn(llpersonality);
            let llretval = pad_bcx.cleanup_pad(None, &[]);
            UnwindKind::CleanupPad(llretval)
        } else {
            // The landing pad return type (the type being propagated). Not sure
            // what this represents but it's determined by the personality
            // function and this is what the EH proposal example uses.
            let llretty = Type::struct_(fcx.ccx,
                                        &[Type::i8p(fcx.ccx), Type::i32(fcx.ccx)],
                                        false);

            // The only landing pad clause will be 'cleanup'
            let llretval = pad_bcx.landing_pad(llretty, llpersonality, 1, pad_bcx.fcx().llfn);

            // The landing pad block is a cleanup
            pad_bcx.set_cleanup(llretval);

            let addr = match fcx.landingpad_alloca.get() {
                Some(addr) => addr,
                None => {
                    let addr = base::alloca(&pad_bcx, common::val_ty(llretval), "");
                    Lifetime::Start.call(&pad_bcx, addr);
                    fcx.landingpad_alloca.set(Some(addr));
                    addr
                }
            };
            pad_bcx.store(llretval, addr);
            UnwindKind::LandingPad
        };

        // Generate a block that will resume unwinding to the calling function
        let bcx = fcx.build_new_block("resume");
        match val {
            UnwindKind::LandingPad => {
                let addr = fcx.landingpad_alloca.get().unwrap();
                let lp = bcx.load(addr);
                Lifetime::End.call(&bcx, addr);
                if !bcx.sess().target.target.options.custom_unwind_resume {
                    bcx.resume(lp);
                } else {
                    let exc_ptr = bcx.extract_value(lp, 0);
                    bcx.call(bcx.fcx().eh_unwind_resume().reify(bcx.ccx()), &[exc_ptr], None);
                }
            }
            UnwindKind::CleanupPad(_) => {
                bcx.cleanup_ret(bcx.cleanup_pad(None, &[]), None);
            }
        }

        let mut cleanup = fcx.build_new_block("clean_custom_");

        // Insert cleanup instructions into the cleanup block
        drop_val.trans(val.get_funclet(&cleanup).as_ref(), &cleanup);

        // Insert instruction into cleanup block to branch to the exit
        val.branch(&mut cleanup, bcx.llbb());

        // Branch into the cleanup block
        val.branch(&mut pad_bcx, cleanup.llbb());

        return pad_bcx.llbb();
    }
}
