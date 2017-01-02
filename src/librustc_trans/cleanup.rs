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

use llvm::BasicBlockRef;
use base;
use mir::lvalue::LvalueRef;
use rustc::mir::tcx::LvalueTy;
use builder::Builder;
use common::Funclet;
use glue;
use type_::Type;

pub struct CleanupScope<'tcx> {
    // Cleanup to run upon scope exit.
    cleanup: Option<DropValue<'tcx>>,

    // Computed on creation if compiling with landing pads (!sess.no_landing_pads)
    pub landing_pad: Option<BasicBlockRef>,
}

#[derive(Copy, Clone)]
pub struct DropValue<'tcx> {
    val: LvalueRef<'tcx>,
    skip_dtor: bool,
}

impl<'tcx> DropValue<'tcx> {
    fn trans<'a>(&self, funclet: Option<&'a Funclet>, bcx: &Builder<'a, 'tcx>) {
        glue::call_drop_glue(bcx, self.val, self.skip_dtor, funclet)
    }

    /// Creates a landing pad for the top scope. The landing pad will perform all cleanups necessary
    /// for an unwind and then `resume` to continue error propagation:
    ///
    ///     landing_pad -> ... cleanups ... -> [resume]
    ///
    /// This should only be called once per function, as it creates an alloca for the landingpad.
    fn get_landing_pad<'a>(&self, bcx: &Builder<'a, 'tcx>) -> BasicBlockRef {
        debug!("get_landing_pad");
        let bcx = bcx.build_sibling_block("cleanup_unwind");
        let llpersonality = bcx.ccx.eh_personality();
        bcx.set_personality_fn(llpersonality);

        if base::wants_msvc_seh(bcx.sess()) {
            let pad = bcx.cleanup_pad(None, &[]);
            let funclet = Some(Funclet::new(pad));
            self.trans(funclet.as_ref(), &bcx);

            bcx.cleanup_ret(pad, None);
        } else {
            // The landing pad return type (the type being propagated). Not sure
            // what this represents but it's determined by the personality
            // function and this is what the EH proposal example uses.
            let llretty = Type::struct_(bcx.ccx, &[Type::i8p(bcx.ccx), Type::i32(bcx.ccx)], false);

            // The only landing pad clause will be 'cleanup'
            let llretval = bcx.landing_pad(llretty, llpersonality, 1, bcx.llfn());

            // The landing pad block is a cleanup
            bcx.set_cleanup(llretval);

            // Insert cleanup instructions into the cleanup block
            self.trans(None, &bcx);

            if !bcx.sess().target.target.options.custom_unwind_resume {
                bcx.resume(llretval);
            } else {
                let exc_ptr = bcx.extract_value(llretval, 0);
                bcx.call(bcx.ccx.eh_unwind_resume(), &[exc_ptr], None);
                bcx.unreachable();
            }
        }

        bcx.llbb()
    }
}

impl<'a, 'tcx> CleanupScope<'tcx> {
    /// Schedules a (deep) drop of `val`, which is a pointer to an instance of `ty`
    pub fn schedule_drop_mem(
        bcx: &Builder<'a, 'tcx>, val: LvalueRef<'tcx>
    ) -> CleanupScope<'tcx> {
        if let LvalueTy::Downcast { .. } = val.ty {
            bug!("Cannot drop downcast ty yet");
        }
        if !bcx.ccx.shared().type_needs_drop(val.ty.to_ty(bcx.tcx())) {
            return CleanupScope::noop();
        }
        let drop = DropValue {
            val: val,
            skip_dtor: false,
        };

        CleanupScope::new(bcx, drop)
    }

    /// Issue #23611: Schedules a (deep) drop of the contents of
    /// `val`, which is a pointer to an instance of struct/enum type
    /// `ty`. The scheduled code handles extracting the discriminant
    /// and dropping the contents associated with that variant
    /// *without* executing any associated drop implementation.
    pub fn schedule_drop_adt_contents(
        bcx: &Builder<'a, 'tcx>, val: LvalueRef<'tcx>
    ) -> CleanupScope<'tcx> {
        if let LvalueTy::Downcast { .. } = val.ty {
            bug!("Cannot drop downcast ty yet");
        }
        // `if` below could be "!contents_needs_drop"; skipping drop
        // is just an optimization, so sound to be conservative.
        if !bcx.ccx.shared().type_needs_drop(val.ty.to_ty(bcx.tcx())) {
            return CleanupScope::noop();
        }

        let drop = DropValue {
            val: val,
            skip_dtor: true,
        };

        CleanupScope::new(bcx, drop)
    }

    fn new(bcx: &Builder<'a, 'tcx>, drop_val: DropValue<'tcx>) -> CleanupScope<'tcx> {
        CleanupScope {
            cleanup: Some(drop_val),
            landing_pad: if !bcx.sess().no_landing_pads() {
                Some(drop_val.get_landing_pad(bcx))
            } else {
                None
            },
        }
    }

    pub fn noop() -> CleanupScope<'tcx> {
        CleanupScope {
            cleanup: None,
            landing_pad: None,
        }
    }

    pub fn trans(self, bcx: &'a Builder<'a, 'tcx>) {
        if let Some(cleanup) = self.cleanup {
            cleanup.trans(None, &bcx);
        }
    }
}
