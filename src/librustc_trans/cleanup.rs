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
//! are exited, either via panic or just normal control flow. The basic
//! idea is that the function context maintains a stack of cleanup scopes
//! that are pushed/popped as we traverse the AST tree. There is typically
//! at least one cleanup scope per AST node; some AST nodes may introduce
//! additional temporary scopes.
//!
//! Cleanup items can be scheduled into any of the scopes on the stack.
//! Typically, when a scope is popped, we will also generate the code for
//! each of its cleanups at that time. This corresponds to a normal exit
//! from a block (for example, an expression completing evaluation
//! successfully without panic). However, it is also possible to pop a
//! block *without* executing its cleanups; this is typically used to
//! guard intermediate values that must be cleaned up on panic, but not
//! if everything goes right. See the section on custom scopes below for
//! more details.
//!
//! Cleanup scopes come in three kinds:
//!
//! - **AST scopes:** each AST node in a function body has a corresponding
//!   AST scope. We push the AST scope when we start generate code for an AST
//!   node and pop it once the AST node has been fully generated.
//! - **Loop scopes:** loops have an additional cleanup scope. Cleanups are
//!   never scheduled into loop scopes; instead, they are used to record the
//!   basic blocks that we should branch to when a `continue` or `break` statement
//!   is encountered.
//! - **Custom scopes:** custom scopes are typically used to ensure cleanup
//!   of intermediate values.
//!
//! ### When to schedule cleanup
//!
//! Although the cleanup system is intended to *feel* fairly declarative,
//! it's still important to time calls to `schedule_clean()` correctly.
//! Basically, you should not schedule cleanup for memory until it has
//! been initialized, because if an unwind should occur before the memory
//! is fully initialized, then the cleanup will run and try to free or
//! drop uninitialized memory. If the initialization itself produces
//! byproducts that need to be freed, then you should use temporary custom
//! scopes to ensure that those byproducts will get freed on unwind.  For
//! example, an expression like `box foo()` will first allocate a box in the
//! heap and then call `foo()` -- if `foo()` should panic, this box needs
//! to be *shallowly* freed.
//!
//! ### Long-distance jumps
//!
//! In addition to popping a scope, which corresponds to normal control
//! flow exiting the scope, we may also *jump out* of a scope into some
//! earlier scope on the stack. This can occur in response to a `return`,
//! `break`, or `continue` statement, but also in response to panic. In
//! any of these cases, we will generate a series of cleanup blocks for
//! each of the scopes that is exited. So, if the stack contains scopes A
//! ... Z, and we break out of a loop whose corresponding cleanup scope is
//! X, we would generate cleanup blocks for the cleanups in X, Y, and Z.
//! After cleanup is done we would branch to the exit point for scope X.
//! But if panic should occur, we would generate cleanups for all the
//! scopes from A to Z and then resume the unwind process afterwards.
//!
//! To avoid generating tons of code, we cache the cleanup blocks that we
//! create for breaks, returns, unwinds, and other jumps. Whenever a new
//! cleanup is scheduled, though, we must clear these cached blocks. A
//! possible improvement would be to keep the cached blocks but simply
//! generate a new block which performs the additional cleanup and then
//! branches to the existing cached blocks.
//!
//! ### AST and loop cleanup scopes
//!
//! AST cleanup scopes are pushed when we begin and end processing an AST
//! node. They are used to house cleanups related to rvalue temporary that
//! get referenced (e.g., due to an expression like `&Foo()`). Whenever an
//! AST scope is popped, we always trans all the cleanups, adding the cleanup
//! code after the postdominator of the AST node.
//!
//! AST nodes that represent breakable loops also push a loop scope; the
//! loop scope never has any actual cleanups, it's just used to point to
//! the basic blocks where control should flow after a "continue" or
//! "break" statement. Popping a loop scope never generates code.
//!
//! ### Custom cleanup scopes
//!
//! Custom cleanup scopes are used for a variety of purposes. The most
//! common though is to handle temporary byproducts, where cleanup only
//! needs to occur on panic. The general strategy is to push a custom
//! cleanup scope, schedule *shallow* cleanups into the custom scope, and
//! then pop the custom scope (without transing the cleanups) when
//! execution succeeds normally. This way the cleanups are only trans'd on
//! unwind, and only up until the point where execution succeeded, at
//! which time the complete value should be stored in an lvalue or some
//! other place where normal cleanup applies.
//!
//! To spell it out, here is an example. Imagine an expression `box expr`.
//! We would basically:
//!
//! 1. Push a custom cleanup scope C.
//! 2. Allocate the box.
//! 3. Schedule a shallow free in the scope C.
//! 4. Trans `expr` into the box.
//! 5. Pop the scope C.
//! 6. Return the box as an rvalue.
//!
//! This way, if a panic occurs while transing `expr`, the custom
//! cleanup scope C is pushed and hence the box will be freed. The trans
//! code for `expr` itself is responsible for freeing any other byproducts
//! that may be in play.

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

    cached_early_exit: Option<CachedEarlyExit>,
    cached_landing_pad: Option<BasicBlockRef>,
}

#[derive(Copy, Clone, Debug)]
pub struct CustomScopeIndex {
    index: usize
}

#[derive(Copy, Clone, Debug)]
enum UnwindKind {
    LandingPad,
    CleanupPad(ValueRef),
}

#[derive(Copy, Clone)]
struct CachedEarlyExit {
    label: UnwindKind,
    cleanup_block: BasicBlockRef,
}

impl<'blk, 'tcx> FunctionContext<'blk, 'tcx> {
    pub fn trans_scope(
        &self,
        bcx: &BlockAndBuilder<'blk, 'tcx>,
        custom_scope: Option<CleanupScope<'tcx>>
    ) {
        if let Some(scope) = custom_scope {
            scope.cleanup.trans(bcx.funclet(), &bcx);
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

        Some(CleanupScope::new(drop))
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
               Value(val),
               ty,
               drop.skip_dtor);

        Some(CleanupScope::new(drop))
    }

    /// Creates a landing pad for the top scope, if one does not exist. The
    /// landing pad will perform all cleanups necessary for an unwind and then
    /// `resume` to continue error propagation:
    ///
    ///     landing_pad -> ... cleanups ... -> [resume]
    ///
    /// (The cleanups and resume instruction are created by
    /// `trans_cleanups_to_exit_scope()`, not in this function itself.)
    pub fn get_landing_pad(&'blk self, scope: &mut CleanupScope<'tcx>) -> BasicBlockRef {
        debug!("get_landing_pad");

        // Check if a landing pad block exists; if not, create one.
        let mut pad_bcx = match scope.cached_landing_pad {
            Some(llbb) => return llbb,
            None => {
                let name = scope.block_name("unwind");
                let pad_bcx = self.build_new_block(&name[..]);
                scope.cached_landing_pad = Some(pad_bcx.llbb());
                pad_bcx
            }
        };

        let llpersonality = pad_bcx.fcx().eh_personality();

        let val = if base::wants_msvc_seh(self.ccx.sess()) {
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
            let llretty = Type::struct_(self.ccx,
                                        &[Type::i8p(self.ccx), Type::i32(self.ccx)],
                                        false);

            // The only landing pad clause will be 'cleanup'
            let llretval = pad_bcx.landing_pad(llretty, llpersonality, 1, pad_bcx.fcx().llfn);

            // The landing pad block is a cleanup
            pad_bcx.set_cleanup(llretval);

            let addr = match self.landingpad_alloca.get() {
                Some(addr) => addr,
                None => {
                    let addr = base::alloca(&pad_bcx, common::val_ty(llretval), "");
                    Lifetime::Start.call(&pad_bcx, addr);
                    self.landingpad_alloca.set(Some(addr));
                    addr
                }
            };
            pad_bcx.store(llretval, addr);
            UnwindKind::LandingPad
        };

        // Generate the cleanup block and branch to it.
        let cleanup_llbb = self.trans_cleanups_to_exit_scope(val, scope);
        val.branch(&mut pad_bcx, cleanup_llbb);

        return pad_bcx.llbb();
    }

    /// Used when the caller wishes to jump to an early exit, such as a return,
    /// break, continue, or unwind. This function will generate all cleanups
    /// between the top of the stack and the exit `label` and return a basic
    /// block that the caller can branch to.
    fn trans_cleanups_to_exit_scope(
        &'blk self,
        label: UnwindKind,
        scope: &mut CleanupScope<'tcx>
    ) -> BasicBlockRef {
        debug!("trans_cleanups_to_exit_scope label={:?}`", label);
        let cached_exit = scope.cached_early_exit(label);

        // Check if we have already cached the unwinding of this
        // scope for this label. If so, we can just branch to the cached block.
        let exit_llbb = cached_exit.unwrap_or_else(|| {
            // Generate a block that will resume unwinding to the calling function
            let bcx = self.build_new_block("resume");
            match label {
                UnwindKind::LandingPad => {
                    let addr = self.landingpad_alloca.get().unwrap();
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
            bcx.llbb()
        });

        let name = scope.block_name("clean");
        debug!("generating cleanup for {}", name);

        let mut cleanup = self.build_new_block(&name[..]);

        // Insert cleanup instructions into the cleanup block
        scope.cleanup.trans(label.get_funclet(&cleanup).as_ref(), &cleanup);

        // Insert instruction into cleanup block to branch to the exit
        label.branch(&mut cleanup, exit_llbb);

        // Cache the work we've done here
        // FIXME: Can this get called more than once per scope? If not, no need to cache.
        scope.add_cached_early_exit(label, cleanup.llbb());

        debug!("trans_cleanups_to_exit_scope: llbb={:?}", cleanup.llbb());

        cleanup.llbb()
    }
}

impl<'tcx> CleanupScope<'tcx> {
    fn new(drop_val: DropValue<'tcx>) -> CleanupScope<'tcx> {
        CleanupScope {
            cleanup: drop_val,
            cached_early_exit: None,
            cached_landing_pad: None,
        }
    }

    fn cached_early_exit(&self, label: UnwindKind) -> Option<BasicBlockRef> {
        if let Some(e) = self.cached_early_exit {
            if e.label == label {
                return Some(e.cleanup_block);
            }
        }
        None
    }

    fn add_cached_early_exit(&mut self,
                             label: UnwindKind,
                             blk: BasicBlockRef) {
        assert!(self.cached_early_exit.is_none());
        self.cached_early_exit = Some(CachedEarlyExit {
            label: label,
            cleanup_block: blk,
        });
    }

    /// Returns a suitable name to use for the basic block that handles this cleanup scope
    fn block_name(&self, prefix: &str) -> String {
        format!("{}_custom_", prefix)
    }
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

///////////////////////////////////////////////////////////////////////////
// Cleanup types

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
