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
use base;
use common::{BlockAndBuilder, FunctionContext, Funclet};
use glue;
use type_::Type;
use value::Value;
use rustc::ty::Ty;

pub struct CleanupScope<'tcx> {
    // Cleanup to run upon scope exit.
    cleanup: Option<DropValue<'tcx>>,

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
    fn trans<'a>(&self, funclet: Option<&'a Funclet>, bcx: &BlockAndBuilder<'a, 'tcx>) {
        glue::call_drop_glue(bcx, self.val, self.ty, self.skip_dtor, funclet)
    }

    /// Creates a landing pad for the top scope. The landing pad will perform all cleanups necessary
    /// for an unwind and then `resume` to continue error propagation:
    ///
    ///     landing_pad -> ... cleanups ... -> [resume]
    ///
    /// This should only be called once per function, as it creates an alloca for the landingpad.
    fn get_landing_pad<'a>(&self, fcx: &FunctionContext<'a, 'tcx>) -> BasicBlockRef {
        debug!("get_landing_pad");
        let bcx = fcx.build_new_block("cleanup_unwind");
        let llpersonality = bcx.ccx.eh_personality();
        bcx.set_personality_fn(llpersonality);

        if base::wants_msvc_seh(fcx.ccx.sess()) {
            let pad = bcx.cleanup_pad(None, &[]);
            let funclet = Some(Funclet::new(pad));
            self.trans(funclet.as_ref(), &bcx);

            bcx.cleanup_ret(pad, None);
        } else {
            // The landing pad return type (the type being propagated). Not sure
            // what this represents but it's determined by the personality
            // function and this is what the EH proposal example uses.
            let llretty = Type::struct_(fcx.ccx, &[Type::i8p(fcx.ccx), Type::i32(fcx.ccx)], false);

            // The only landing pad clause will be 'cleanup'
            let llretval = bcx.landing_pad(llretty, llpersonality, 1, bcx.fcx().llfn);

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

impl<'a, 'tcx> FunctionContext<'a, 'tcx> {
    /// Schedules a (deep) drop of `val`, which is a pointer to an instance of `ty`
    pub fn schedule_drop_mem(&self, val: ValueRef, ty: Ty<'tcx>) -> CleanupScope<'tcx> {
        if !self.ccx.shared().type_needs_drop(ty) { return CleanupScope::noop(); }
        let drop = DropValue {
            val: val,
            ty: ty,
            skip_dtor: false,
        };

        debug!("schedule_drop_mem(val={:?}, ty={:?}) skip_dtor={}", Value(val), ty, drop.skip_dtor);

        CleanupScope::new(self, drop)
    }

    /// Issue #23611: Schedules a (deep) drop of the contents of
    /// `val`, which is a pointer to an instance of struct/enum type
    /// `ty`. The scheduled code handles extracting the discriminant
    /// and dropping the contents associated with that variant
    /// *without* executing any associated drop implementation.
    pub fn schedule_drop_adt_contents(&self, val: ValueRef, ty: Ty<'tcx>) -> CleanupScope<'tcx> {
        // `if` below could be "!contents_needs_drop"; skipping drop
        // is just an optimization, so sound to be conservative.
        if !self.ccx.shared().type_needs_drop(ty) { return CleanupScope::noop(); }

        let drop = DropValue {
            val: val,
            ty: ty,
            skip_dtor: true,
        };

        debug!("schedule_drop_adt_contents(val={:?}, ty={:?}) skip_dtor={}",
               Value(val), ty, drop.skip_dtor);

        CleanupScope::new(self, drop)
    }
}

impl<'tcx> CleanupScope<'tcx> {
    fn new<'a>(fcx: &FunctionContext<'a, 'tcx>, drop_val: DropValue<'tcx>) -> CleanupScope<'tcx> {
        CleanupScope {
            cleanup: Some(drop_val),
            landing_pad: if !fcx.ccx.sess().no_landing_pads() {
                Some(drop_val.get_landing_pad(fcx))
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

    pub fn trans<'a>(self, bcx: &'a BlockAndBuilder<'a, 'tcx>) {
        if let Some(cleanup) = self.cleanup {
            cleanup.trans(None, &bcx);
        }
    }
}
