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

pub use self::EarlyExitLabel::*;

use llvm::{BasicBlockRef, ValueRef};
use base;
use build;
use common;
use common::{Block, FunctionContext, LandingPad};
use debuginfo::{DebugLoc};
use glue;
use type_::Type;
use value::Value;
use rustc::ty::Ty;

pub struct CleanupScope<'tcx> {
    // Cleanups to run upon scope exit.
    cleanups: Vec<DropValue<'tcx>>,

    // The debug location any drop calls generated for this scope will be
    // associated with.
    debug_loc: DebugLoc,

    cached_early_exits: Vec<CachedEarlyExit>,
    cached_landing_pad: Option<BasicBlockRef>,
}

#[derive(Copy, Clone, Debug)]
pub struct CustomScopeIndex {
    index: usize
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum EarlyExitLabel {
    UnwindExit(UnwindKind),
}

#[derive(Copy, Clone, Debug)]
pub enum UnwindKind {
    LandingPad,
    CleanupPad(ValueRef),
}

#[derive(Copy, Clone)]
pub struct CachedEarlyExit {
    label: EarlyExitLabel,
    cleanup_block: BasicBlockRef,
    last_cleanup: usize,
}

impl<'blk, 'tcx> FunctionContext<'blk, 'tcx> {
    pub fn push_custom_cleanup_scope(&self) -> CustomScopeIndex {
        let index = self.scopes_len();
        debug!("push_custom_cleanup_scope(): {}", index);

        // Just copy the debuginfo source location from the enclosing scope
        let debug_loc = self.scopes
                            .borrow()
                            .last()
                            .map(|opt_scope| opt_scope.debug_loc)
                            .unwrap_or(DebugLoc::None);

        self.push_scope(CleanupScope::new(debug_loc));
        CustomScopeIndex { index: index }
    }

    /// Removes the top cleanup scope from the stack without executing its cleanups. The top
    /// cleanup scope must be the temporary scope `custom_scope`.
    pub fn pop_custom_cleanup_scope(&self,
                                    custom_scope: CustomScopeIndex) {
        debug!("pop_custom_cleanup_scope({})", custom_scope.index);
        assert!(self.is_valid_to_pop_custom_scope(custom_scope));
        let _ = self.pop_scope();
    }

    /// Removes the top cleanup scope from the stack, which must be a temporary scope, and
    /// generates the code to do its cleanups for normal exit.
    pub fn pop_and_trans_custom_cleanup_scope(&self,
                                              bcx: Block<'blk, 'tcx>,
                                              custom_scope: CustomScopeIndex)
                                              -> Block<'blk, 'tcx> {
        debug!("pop_and_trans_custom_cleanup_scope({:?})", custom_scope);
        assert!(self.is_valid_to_pop_custom_scope(custom_scope));

        let scope = self.pop_scope();
        self.trans_scope_cleanups(bcx, &scope)
    }

    /// Schedules a (deep) drop of `val`, which is a pointer to an instance of
    /// `ty`
    pub fn schedule_drop_mem(&self,
                             cleanup_scope: CustomScopeIndex,
                             val: ValueRef,
                             ty: Ty<'tcx>) {
        if !self.type_needs_drop(ty) { return; }
        let drop = DropValue {
            is_immediate: false,
            val: val,
            ty: ty,
            skip_dtor: false,
        };

        debug!("schedule_drop_mem({:?}, val={:?}, ty={:?}) skip_dtor={}",
               cleanup_scope,
               Value(val),
               ty,
               drop.skip_dtor);

        self.schedule_clean(cleanup_scope, drop);
    }

    /// Issue #23611: Schedules a (deep) drop of the contents of
    /// `val`, which is a pointer to an instance of struct/enum type
    /// `ty`. The scheduled code handles extracting the discriminant
    /// and dropping the contents associated with that variant
    /// *without* executing any associated drop implementation.
    pub fn schedule_drop_adt_contents(&self,
                                      cleanup_scope: CustomScopeIndex,
                                      val: ValueRef,
                                      ty: Ty<'tcx>) {
        // `if` below could be "!contents_needs_drop"; skipping drop
        // is just an optimization, so sound to be conservative.
        if !self.type_needs_drop(ty) { return; }

        let drop = DropValue {
            is_immediate: false,
            val: val,
            ty: ty,
            skip_dtor: true,
        };

        debug!("schedule_drop_adt_contents({:?}, val={:?}, ty={:?}) skip_dtor={}",
               cleanup_scope,
               Value(val),
               ty,
               drop.skip_dtor);

        self.schedule_clean(cleanup_scope, drop);
    }

    /// Schedules a (deep) drop of `val`, which is an instance of `ty`
    pub fn schedule_drop_immediate(&self,
                                   cleanup_scope: CustomScopeIndex,
                                   val: ValueRef,
                                   ty: Ty<'tcx>) {

        if !self.type_needs_drop(ty) { return; }
        let drop = DropValue {
            is_immediate: true,
            val: val,
            ty: ty,
            skip_dtor: false,
        };

        debug!("schedule_drop_immediate({:?}, val={:?}, ty={:?}) skip_dtor={}",
               cleanup_scope,
               Value(val),
               ty,
               drop.skip_dtor);

        self.schedule_clean(cleanup_scope, drop);
    }

    /// Schedules a cleanup to occur in the top-most scope, which must be a temporary scope.
    fn schedule_clean(&self, custom_scope: CustomScopeIndex, cleanup: DropValue<'tcx>) {
        debug!("schedule_clean_in_custom_scope(custom_scope={})",
               custom_scope.index);

        assert!(self.is_valid_custom_scope(custom_scope));

        let mut scopes = self.scopes.borrow_mut();
        let scope = &mut (*scopes)[custom_scope.index];
        scope.cleanups.push(cleanup);
        scope.cached_landing_pad = None;
    }

    /// Returns true if there are pending cleanups that should execute on panic.
    pub fn needs_invoke(&self) -> bool {
        self.scopes.borrow().iter().rev().any(|s| s.needs_invoke())
    }

    /// Returns a basic block to branch to in the event of a panic. This block
    /// will run the panic cleanups and eventually resume the exception that
    /// caused the landing pad to be run.
    pub fn get_landing_pad(&'blk self) -> BasicBlockRef {
        let _icx = base::push_ctxt("get_landing_pad");

        debug!("get_landing_pad");

        let orig_scopes_len = self.scopes_len();
        assert!(orig_scopes_len > 0);

        // Remove any scopes that do not have cleanups on panic:
        let mut popped_scopes = vec!();
        while !self.top_scope(|s| s.needs_invoke()) {
            debug!("top scope does not need invoke");
            popped_scopes.push(self.pop_scope());
        }

        // Check for an existing landing pad in the new topmost scope:
        let llbb = self.get_or_create_landing_pad();

        // Push the scopes we removed back on:
        loop {
            match popped_scopes.pop() {
                Some(scope) => self.push_scope(scope),
                None => break
            }
        }

        assert_eq!(self.scopes_len(), orig_scopes_len);

        return llbb;
    }

    fn is_valid_to_pop_custom_scope(&self, custom_scope: CustomScopeIndex) -> bool {
        self.is_valid_custom_scope(custom_scope) &&
            custom_scope.index == self.scopes.borrow().len() - 1
    }

    fn is_valid_custom_scope(&self, custom_scope: CustomScopeIndex) -> bool {
        let scopes = self.scopes.borrow();
        custom_scope.index < scopes.len()
    }

    /// Generates the cleanups for `scope` into `bcx`
    fn trans_scope_cleanups(&self, // cannot borrow self, will recurse
                            bcx: Block<'blk, 'tcx>,
                            scope: &CleanupScope<'tcx>) -> Block<'blk, 'tcx> {

        let mut bcx = bcx;
        if !bcx.unreachable.get() {
            for cleanup in scope.cleanups.iter().rev() {
                bcx = cleanup.trans(bcx, scope.debug_loc);
            }
        }
        bcx
    }

    fn scopes_len(&self) -> usize {
        self.scopes.borrow().len()
    }

    fn push_scope(&self, scope: CleanupScope<'tcx>) {
        self.scopes.borrow_mut().push(scope)
    }

    fn pop_scope(&self) -> CleanupScope<'tcx> {
        debug!("popping cleanup scope {}, {} scopes remaining",
               self.top_scope(|s| s.block_name("")),
               self.scopes_len() - 1);

        self.scopes.borrow_mut().pop().unwrap()
    }

    fn top_scope<R, F>(&self, f: F) -> R where F: FnOnce(&CleanupScope<'tcx>) -> R {
        f(self.scopes.borrow().last().unwrap())
    }

    /// Used when the caller wishes to jump to an early exit, such as a return,
    /// break, continue, or unwind. This function will generate all cleanups
    /// between the top of the stack and the exit `label` and return a basic
    /// block that the caller can branch to.
    ///
    /// For example, if the current stack of cleanups were as follows:
    ///
    ///      AST 22
    ///      Custom 1
    ///      AST 23
    ///      Loop 23
    ///      Custom 2
    ///      AST 24
    ///
    /// and the `label` specifies a break from `Loop 23`, then this function
    /// would generate a series of basic blocks as follows:
    ///
    ///      Cleanup(AST 24) -> Cleanup(Custom 2) -> break_blk
    ///
    /// where `break_blk` is the block specified in `Loop 23` as the target for
    /// breaks. The return value would be the first basic block in that sequence
    /// (`Cleanup(AST 24)`). The caller could then branch to `Cleanup(AST 24)`
    /// and it will perform all cleanups and finally branch to the `break_blk`.
    fn trans_cleanups_to_exit_scope(&'blk self,
                                    label: EarlyExitLabel)
                                    -> BasicBlockRef {
        debug!("trans_cleanups_to_exit_scope label={:?} scopes={}",
               label, self.scopes_len());

        let orig_scopes_len = self.scopes_len();
        let mut prev_llbb;
        let mut popped_scopes = vec!();
        let mut skip = 0;

        // First we pop off all the cleanup stacks that are
        // traversed until the exit is reached, pushing them
        // onto the side vector `popped_scopes`. No code is
        // generated at this time.
        //
        // So, continuing the example from above, we would wind up
        // with a `popped_scopes` vector of `[AST 24, Custom 2]`.
        // (Presuming that there are no cached exits)
        loop {
            if self.scopes_len() == 0 {
                match label {
                    UnwindExit(val) => {
                        // Generate a block that will resume unwinding to the
                        // calling function
                        let bcx = self.new_block("resume");
                        match val {
                            UnwindKind::LandingPad => {
                                let addr = self.landingpad_alloca.get()
                                               .unwrap();
                                let lp = build::Load(bcx, addr);
                                base::call_lifetime_end(bcx, addr);
                                base::trans_unwind_resume(bcx, lp);
                            }
                            UnwindKind::CleanupPad(_) => {
                                let pad = build::CleanupPad(bcx, None, &[]);
                                build::CleanupRet(bcx, pad, None);
                            }
                        }
                        prev_llbb = bcx.llbb;
                        break;
                    }
                }
            }

            // Pop off the scope, since we may be generating
            // unwinding code for it.
            let top_scope = self.pop_scope();
            let cached_exit = top_scope.cached_early_exit(label);
            popped_scopes.push(top_scope);

            // Check if we have already cached the unwinding of this
            // scope for this label. If so, we can stop popping scopes
            // and branch to the cached label, since it contains the
            // cleanups for any subsequent scopes.
            if let Some((exit, last_cleanup)) = cached_exit {
                prev_llbb = exit;
                skip = last_cleanup;
                break;
            }
        }

        debug!("trans_cleanups_to_exit_scope: popped {} scopes",
               popped_scopes.len());

        // Now push the popped scopes back on. As we go,
        // we track in `prev_llbb` the exit to which this scope
        // should branch when it's done.
        //
        // So, continuing with our example, we will start out with
        // `prev_llbb` being set to `break_blk` (or possibly a cached
        // early exit). We will then pop the scopes from `popped_scopes`
        // and generate a basic block for each one, prepending it in the
        // series and updating `prev_llbb`. So we begin by popping `Custom 2`
        // and generating `Cleanup(Custom 2)`. We make `Cleanup(Custom 2)`
        // branch to `prev_llbb == break_blk`, giving us a sequence like:
        //
        //     Cleanup(Custom 2) -> prev_llbb
        //
        // We then pop `AST 24` and repeat the process, giving us the sequence:
        //
        //     Cleanup(AST 24) -> Cleanup(Custom 2) -> prev_llbb
        //
        // At this point, `popped_scopes` is empty, and so the final block
        // that we return to the user is `Cleanup(AST 24)`.
        while let Some(mut scope) = popped_scopes.pop() {
            if !scope.cleanups.is_empty() {
                let name = scope.block_name("clean");
                debug!("generating cleanups for {}", name);

                let bcx_in = self.new_block(&name[..]);
                let exit_label = label.start(bcx_in);
                let mut bcx_out = bcx_in;
                let len = scope.cleanups.len();
                for cleanup in scope.cleanups.iter().rev().take(len - skip) {
                    bcx_out = cleanup.trans(bcx_out, scope.debug_loc);
                }
                skip = 0;
                exit_label.branch(bcx_out, prev_llbb);
                prev_llbb = bcx_in.llbb;

                scope.add_cached_early_exit(exit_label, prev_llbb, len);
            }
            self.push_scope(scope);
        }

        debug!("trans_cleanups_to_exit_scope: prev_llbb={:?}", prev_llbb);

        assert_eq!(self.scopes_len(), orig_scopes_len);
        prev_llbb
    }

    /// Creates a landing pad for the top scope, if one does not exist.  The
    /// landing pad will perform all cleanups necessary for an unwind and then
    /// `resume` to continue error propagation:
    ///
    ///     landing_pad -> ... cleanups ... -> [resume]
    ///
    /// (The cleanups and resume instruction are created by
    /// `trans_cleanups_to_exit_scope()`, not in this function itself.)
    fn get_or_create_landing_pad(&'blk self) -> BasicBlockRef {
        let pad_bcx;

        debug!("get_or_create_landing_pad");

        // Check if a landing pad block exists; if not, create one.
        {
            let mut scopes = self.scopes.borrow_mut();
            let last_scope = scopes.last_mut().unwrap();
            match last_scope.cached_landing_pad {
                Some(llbb) => return llbb,
                None => {
                    let name = last_scope.block_name("unwind");
                    pad_bcx = self.new_block(&name[..]);
                    last_scope.cached_landing_pad = Some(pad_bcx.llbb);
                }
            }
        };

        let llpersonality = pad_bcx.fcx.eh_personality();

        let val = if base::wants_msvc_seh(self.ccx.sess()) {
            // A cleanup pad requires a personality function to be specified, so
            // we do that here explicitly (happens implicitly below through
            // creation of the landingpad instruction). We then create a
            // cleanuppad instruction which has no filters to run cleanup on all
            // exceptions.
            build::SetPersonalityFn(pad_bcx, llpersonality);
            let llretval = build::CleanupPad(pad_bcx, None, &[]);
            UnwindKind::CleanupPad(llretval)
        } else {
            // The landing pad return type (the type being propagated). Not sure
            // what this represents but it's determined by the personality
            // function and this is what the EH proposal example uses.
            let llretty = Type::struct_(self.ccx,
                                        &[Type::i8p(self.ccx), Type::i32(self.ccx)],
                                        false);

            // The only landing pad clause will be 'cleanup'
            let llretval = build::LandingPad(pad_bcx, llretty, llpersonality, 1);

            // The landing pad block is a cleanup
            build::SetCleanup(pad_bcx, llretval);

            let addr = match self.landingpad_alloca.get() {
                Some(addr) => addr,
                None => {
                    let addr = base::alloca(pad_bcx, common::val_ty(llretval),
                                            "");
                    base::call_lifetime_start(pad_bcx, addr);
                    self.landingpad_alloca.set(Some(addr));
                    addr
                }
            };
            build::Store(pad_bcx, llretval, addr);
            UnwindKind::LandingPad
        };

        // Generate the cleanup block and branch to it.
        let label = UnwindExit(val);
        let cleanup_llbb = self.trans_cleanups_to_exit_scope(label);
        label.branch(pad_bcx, cleanup_llbb);

        return pad_bcx.llbb;
    }
}

impl<'tcx> CleanupScope<'tcx> {
    fn new(debug_loc: DebugLoc) -> CleanupScope<'tcx> {
        CleanupScope {
            debug_loc: debug_loc,
            cleanups: vec!(),
            cached_early_exits: vec!(),
            cached_landing_pad: None,
        }
    }

    fn cached_early_exit(&self,
                         label: EarlyExitLabel)
                         -> Option<(BasicBlockRef, usize)> {
        self.cached_early_exits.iter().rev().
            find(|e| e.label == label).
            map(|e| (e.cleanup_block, e.last_cleanup))
    }

    fn add_cached_early_exit(&mut self,
                             label: EarlyExitLabel,
                             blk: BasicBlockRef,
                             last_cleanup: usize) {
        self.cached_early_exits.push(
            CachedEarlyExit { label: label,
                              cleanup_block: blk,
                              last_cleanup: last_cleanup});
    }

    /// True if this scope has cleanups that need unwinding
    fn needs_invoke(&self) -> bool {
        self.cached_landing_pad.is_some() ||
            !self.cleanups.is_empty()
    }

    /// Returns a suitable name to use for the basic block that handles this cleanup scope
    fn block_name(&self, prefix: &str) -> String {
        format!("{}_custom_", prefix)
    }
}

impl EarlyExitLabel {
    /// Generates a branch going from `from_bcx` to `to_llbb` where `self` is
    /// the exit label attached to the start of `from_bcx`.
    ///
    /// Transitions from an exit label to other exit labels depend on the type
    /// of label. For example with MSVC exceptions unwind exit labels will use
    /// the `cleanupret` instruction instead of the `br` instruction.
    fn branch(&self, from_bcx: Block, to_llbb: BasicBlockRef) {
        if let UnwindExit(UnwindKind::CleanupPad(pad)) = *self {
            build::CleanupRet(from_bcx, pad, Some(to_llbb));
        } else {
            build::Br(from_bcx, to_llbb, DebugLoc::None);
        }
    }

    /// Generates the necessary instructions at the start of `bcx` to prepare
    /// for the same kind of early exit label that `self` is.
    ///
    /// This function will appropriately configure `bcx` based on the kind of
    /// label this is. For UnwindExit labels, the `lpad` field of the block will
    /// be set to `Some`, and for MSVC exceptions this function will generate a
    /// `cleanuppad` instruction at the start of the block so it may be jumped
    /// to in the future (e.g. so this block can be cached as an early exit).
    ///
    /// Returns a new label which will can be used to cache `bcx` in the list of
    /// early exits.
    fn start(&self, bcx: Block) -> EarlyExitLabel {
        match *self {
            UnwindExit(UnwindKind::CleanupPad(..)) => {
                let pad = build::CleanupPad(bcx, None, &[]);
                bcx.lpad.set(Some(bcx.fcx.lpad_arena.alloc(LandingPad::msvc(pad))));
                UnwindExit(UnwindKind::CleanupPad(pad))
            }
            UnwindExit(UnwindKind::LandingPad) => {
                bcx.lpad.set(Some(bcx.fcx.lpad_arena.alloc(LandingPad::gnu())));
                *self
            }
        }
    }
}

impl PartialEq for UnwindKind {
    fn eq(&self, val: &UnwindKind) -> bool {
        match (*self, *val) {
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
    is_immediate: bool,
    val: ValueRef,
    ty: Ty<'tcx>,
    skip_dtor: bool,
}

impl<'tcx> DropValue<'tcx> {
    fn trans<'blk>(&self,
                   bcx: Block<'blk, 'tcx>,
                   debug_loc: DebugLoc)
                   -> Block<'blk, 'tcx> {
        let skip_dtor = self.skip_dtor;
        let _icx = if skip_dtor {
            base::push_ctxt("<DropValue as Cleanup>::trans skip_dtor=true")
        } else {
            base::push_ctxt("<DropValue as Cleanup>::trans skip_dtor=false")
        };
        let bcx = if self.is_immediate {
            glue::drop_ty_immediate(bcx, self.val, self.ty, debug_loc, self.skip_dtor)
        } else {
            glue::drop_ty_core(bcx, self.val, self.ty, debug_loc, self.skip_dtor)
        };
        bcx
    }
}
