// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Code pertaining to cleanup of temporaries as well as execution of
 * drop glue. See discussion in `doc.rs` for a high-level summary.
 */

use lib::llvm::{BasicBlockRef, ValueRef};
use middle::lang_items::{EhPersonalityLangItem};
use middle::trans::base;
use middle::trans::build;
use middle::trans::callee;
use middle::trans::common;
use middle::trans::common::{Block, FunctionContext};
use middle::trans::glue;
use middle::trans::type_::Type;
use middle::ty;
use syntax::ast;
use syntax::opt_vec;
use syntax::opt_vec::OptVec;
use util::ppaux::Repr;

pub struct CleanupScope<'a> {
    // The id of this cleanup scope. If the id is None,
    // this is a *temporary scope* that is pushed during trans to
    // cleanup miscellaneous garbage that trans may generate whose
    // lifetime is a subset of some expression.  See module doc for
    // more details.
    kind: CleanupScopeKind<'a>,

    // Cleanups to run upon scope exit.
    cleanups: OptVec<~Cleanup>,

    cached_early_exits: OptVec<CachedEarlyExit>,
    cached_landing_pad: Option<BasicBlockRef>,
}

pub struct CustomScopeIndex {
    priv index: uint
}

pub static EXIT_BREAK: uint = 0;
pub static EXIT_LOOP: uint = 1;
pub static EXIT_MAX: uint = 2;

enum CleanupScopeKind<'a> {
    CustomScopeKind,
    AstScopeKind(ast::NodeId),
    LoopScopeKind(ast::NodeId, [&'a Block<'a>, ..EXIT_MAX])
}

#[deriving(Eq)]
enum EarlyExitLabel {
    UnwindExit,
    ReturnExit,
    LoopExit(ast::NodeId, uint)
}

struct CachedEarlyExit {
    label: EarlyExitLabel,
    cleanup_block: BasicBlockRef,
}

pub trait Cleanup {
    fn clean_on_unwind(&self) -> bool;
    fn trans<'a>(&self, bcx: &'a Block<'a>) -> &'a Block<'a>;
}

pub enum ScopeId {
    AstScope(ast::NodeId),
    CustomScope(CustomScopeIndex)
}

impl<'a> CleanupMethods<'a> for FunctionContext<'a> {
    fn push_ast_cleanup_scope(&self, id: ast::NodeId) {
        /*!
         * Invoked when we start to trans the code contained
         * within a new cleanup scope.
         */

        debug!("push_ast_cleanup_scope({})",
               self.ccx.tcx.map.node_to_str(id));

        // FIXME(#2202) -- currently closure bodies have a parent
        // region, which messes up the assertion below, since there
        // are no cleanup scopes on the stack at the start of
        // trans'ing a closure body.  I think though that this should
        // eventually be fixed by closure bodies not having a parent
        // region, though that's a touch unclear, and it might also be
        // better just to narrow this assertion more (i.e., by
        // excluding id's that correspond to closure bodies only). For
        // now we just say that if there is already an AST scope on the stack,
        // this new AST scope had better be its immediate child.
        let top_scope = self.top_ast_scope();
        if top_scope.is_some() {
            assert_eq!(self.ccx.tcx.region_maps.opt_encl_scope(id), top_scope);
        }

        self.push_scope(CleanupScope::new(AstScopeKind(id)));
    }

    fn push_loop_cleanup_scope(&self,
                               id: ast::NodeId,
                               exits: [&'a Block<'a>, ..EXIT_MAX]) {
        debug!("push_loop_cleanup_scope({})",
               self.ccx.tcx.map.node_to_str(id));
        assert_eq!(Some(id), self.top_ast_scope());

        self.push_scope(CleanupScope::new(LoopScopeKind(id, exits)));
    }

    fn push_custom_cleanup_scope(&self) -> CustomScopeIndex {
        let index = self.scopes_len();
        debug!("push_custom_cleanup_scope(): {}", index);
        self.push_scope(CleanupScope::new(CustomScopeKind));
        CustomScopeIndex { index: index }
    }

    fn pop_and_trans_ast_cleanup_scope(&self,
                                       bcx: &'a Block<'a>,
                                       cleanup_scope: ast::NodeId)
                                       -> &'a Block<'a> {
        /*!
         * Removes the cleanup scope for id `cleanup_scope`, which
         * must be at the top of the cleanup stack, and generates the
         * code to do its cleanups for normal exit.
         */

        debug!("pop_and_trans_ast_cleanup_scope({})",
               self.ccx.tcx.map.node_to_str(cleanup_scope));

        assert!(self.top_scope(|s| s.kind.is_ast_with_id(cleanup_scope)));

        let scope = self.pop_scope();
        self.trans_scope_cleanups(bcx, &scope)

    }

    fn pop_loop_cleanup_scope(&self,
                              cleanup_scope: ast::NodeId) {
        /*!
         * Removes the loop cleanup scope for id `cleanup_scope`, which
         * must be at the top of the cleanup stack. Does not generate
         * any cleanup code, since loop scopes should exit by
         * branching to a block generated by `normal_exit_block`.
         */

        debug!("pop_loop_cleanup_scope({})",
               self.ccx.tcx.map.node_to_str(cleanup_scope));

        assert!(self.top_scope(|s| s.kind.is_loop_with_id(cleanup_scope)));

        let _ = self.pop_scope();
    }

    fn pop_custom_cleanup_scope(&self,
                                custom_scope: CustomScopeIndex) {
        /*!
         * Removes the top cleanup scope from the stack without
         * executing its cleanups. The top cleanup scope must
         * be the temporary scope `custom_scope`.
         */

        debug!("pop_custom_cleanup_scope({})", custom_scope.index);
        assert!(self.is_valid_to_pop_custom_scope(custom_scope));
        let _ = self.pop_scope();
    }

    fn pop_and_trans_custom_cleanup_scope(&self,
                                        bcx: &'a Block<'a>,
                                        custom_scope: CustomScopeIndex)
                                        -> &'a Block<'a> {
        /*!
         * Removes the top cleanup scope from the stack, which must be
         * a temporary scope, and generates the code to do its
         * cleanups for normal exit.
         */

        debug!("pop_and_trans_custom_cleanup_scope({:?})", custom_scope);
        assert!(self.is_valid_to_pop_custom_scope(custom_scope));

        let scope = self.pop_scope();
        self.trans_scope_cleanups(bcx, &scope)
    }

    fn top_loop_scope(&self) -> ast::NodeId {
        /*!
         * Returns the id of the top-most loop scope
         */

        let scopes = self.scopes.borrow();
        for scope in scopes.get().iter().rev() {
            match scope.kind {
                LoopScopeKind(id, _) => {
                    return id;
                }
                _ => {}
            }
        }
        self.ccx.tcx.sess.bug("no loop scope found");
    }

    fn normal_exit_block(&'a self,
                         cleanup_scope: ast::NodeId,
                         exit: uint) -> BasicBlockRef {
        /*!
         * Returns a block to branch to which will perform all pending
         * cleanups and then break/continue (depending on `exit`) out
         * of the loop with id `cleanup_scope`
         */

        self.trans_cleanups_to_exit_scope(LoopExit(cleanup_scope, exit))
    }

    fn return_exit_block(&'a self) -> BasicBlockRef {
        /*!
         * Returns a block to branch to which will perform all pending
         * cleanups and then return from this function
         */

        self.trans_cleanups_to_exit_scope(ReturnExit)
    }

    fn schedule_drop_mem(&self,
                         cleanup_scope: ScopeId,
                         val: ValueRef,
                         ty: ty::t) {
        /*!
         * Schedules a (deep) drop of `val`, which is a pointer to an
         * instance of `ty`
         */

        if !ty::type_needs_drop(self.ccx.tcx, ty) { return; }
        let drop = ~DropValue {
            is_immediate: false,
            on_unwind: ty::type_needs_unwind_cleanup(self.ccx.tcx, ty),
            val: val,
            ty: ty
        };

        debug!("schedule_drop_mem({:?}, val={}, ty={})",
               cleanup_scope,
               self.ccx.tn.val_to_str(val),
               ty.repr(self.ccx.tcx));

        self.schedule_clean(cleanup_scope, drop as ~Cleanup);
    }

    fn schedule_drop_immediate(&self,
                               cleanup_scope: ScopeId,
                               val: ValueRef,
                               ty: ty::t) {
        /*!
         * Schedules a (deep) drop of `val`, which is an instance of `ty`
         */

        if !ty::type_needs_drop(self.ccx.tcx, ty) { return; }
        let drop = ~DropValue {
            is_immediate: true,
            on_unwind: ty::type_needs_unwind_cleanup(self.ccx.tcx, ty),
            val: val,
            ty: ty
        };

        debug!("schedule_drop_immediate({:?}, val={}, ty={})",
               cleanup_scope,
               self.ccx.tn.val_to_str(val),
               ty.repr(self.ccx.tcx));

        self.schedule_clean(cleanup_scope, drop as ~Cleanup);
    }

    fn schedule_free_value(&self,
                           cleanup_scope: ScopeId,
                           val: ValueRef,
                           heap: common::heap) {
        /*!
         * Schedules a call to `free(val)`. Note that this is a shallow
         * operation.
         */

        let drop = ~FreeValue { ptr: val, heap: heap };

        debug!("schedule_free_value({:?}, val={}, heap={:?})",
               cleanup_scope,
               self.ccx.tn.val_to_str(val),
               heap);

        self.schedule_clean(cleanup_scope, drop as ~Cleanup);
    }

    fn schedule_clean(&self,
                      cleanup_scope: ScopeId,
                      cleanup: ~Cleanup) {
        match cleanup_scope {
            AstScope(id) => self.schedule_clean_in_ast_scope(id, cleanup),
            CustomScope(id) => self.schedule_clean_in_custom_scope(id, cleanup),
        }
    }

    fn schedule_clean_in_ast_scope(&self,
                                   cleanup_scope: ast::NodeId,
                                   cleanup: ~Cleanup) {
        /*!
         * Schedules a cleanup to occur upon exit from `cleanup_scope`.
         * If `cleanup_scope` is not provided, then the cleanup is scheduled
         * in the topmost scope, which must be a temporary scope.
         */

        debug!("schedule_clean_in_ast_scope(cleanup_scope={:?})",
               cleanup_scope);

        let mut scopes = self.scopes.borrow_mut();
        for scope in scopes.get().mut_iter().rev() {
            if scope.kind.is_ast_with_id(cleanup_scope) {
                scope.cleanups.push(cleanup);
                scope.clear_cached_exits();
                return;
            } else {
                // will be adding a cleanup to some enclosing scope
                scope.clear_cached_exits();
            }
        }

        self.ccx.tcx.sess.bug(
            format!("no cleanup scope {} found",
                    self.ccx.tcx.map.node_to_str(cleanup_scope)));
    }

    fn schedule_clean_in_custom_scope(&self,
                                      custom_scope: CustomScopeIndex,
                                      cleanup: ~Cleanup) {
        /*!
         * Schedules a cleanup to occur in the top-most scope,
         * which must be a temporary scope.
         */

        debug!("schedule_clean_in_custom_scope(custom_scope={})",
               custom_scope.index);

        assert!(self.is_valid_custom_scope(custom_scope));

        let mut scopes = self.scopes.borrow_mut();
        let scope = &mut scopes.get()[custom_scope.index];
        scope.cleanups.push(cleanup);
        scope.clear_cached_exits();
    }

    fn needs_invoke(&self) -> bool {
        /*!
         * Returns true if there are pending cleanups that should
         * execute on failure.
         */

        let scopes = self.scopes.borrow();
        scopes.get().iter().rev().any(|s| s.needs_invoke())
    }

    fn get_landing_pad(&'a self) -> BasicBlockRef {
        /*!
         * Returns a basic block to branch to in the event of a failure.
         * This block will run the failure cleanups and eventually
         * invoke the LLVM `Resume` instruction.
         */

        let _icx = base::push_ctxt("get_landing_pad");

        debug!("get_landing_pad");

        let orig_scopes_len = self.scopes_len();
        assert!(orig_scopes_len > 0);

        // Remove any scopes that do not have cleanups on failure:
        let mut popped_scopes = opt_vec::Empty;
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
}

impl<'a> CleanupHelperMethods<'a> for FunctionContext<'a> {
    fn top_ast_scope(&self) -> Option<ast::NodeId> {
        /*!
         * Returns the id of the current top-most AST scope, if any.
         */
        let scopes = self.scopes.borrow();
        for scope in scopes.get().iter().rev() {
            match scope.kind {
                CustomScopeKind | LoopScopeKind(..) => {}
                AstScopeKind(i) => {
                    return Some(i);
                }
            }
        }
        None
    }

    fn top_nonempty_cleanup_scope(&self) -> Option<uint> {
        let scopes = self.scopes.borrow();
        scopes.get().iter().rev().position(|s| !s.cleanups.is_empty())
    }

    fn is_valid_to_pop_custom_scope(&self, custom_scope: CustomScopeIndex) -> bool {
        let scopes = self.scopes.borrow();
        self.is_valid_custom_scope(custom_scope) &&
            custom_scope.index == scopes.get().len() - 1
    }

    fn is_valid_custom_scope(&self, custom_scope: CustomScopeIndex) -> bool {
        let scopes = self.scopes.borrow();
        custom_scope.index < scopes.get().len() &&
            scopes.get()[custom_scope.index].kind.is_temp()
    }

    fn trans_scope_cleanups(&self, // cannot borrow self, will recurse
                            bcx: &'a Block<'a>,
                            scope: &CleanupScope) -> &'a Block<'a> {
        /*! Generates the cleanups for `scope` into `bcx` */

        let mut bcx = bcx;
        if !bcx.unreachable.get() {
            for cleanup in scope.cleanups.iter().rev() {
                bcx = cleanup.trans(bcx);
            }
        }
        bcx
    }

    fn scopes_len(&self) -> uint {
        let scopes = self.scopes.borrow();
        scopes.get().len()
    }

    fn push_scope(&self, scope: CleanupScope<'a>) {
        let mut scopes = self.scopes.borrow_mut();
        scopes.get().push(scope);
    }

    fn pop_scope(&self) -> CleanupScope<'a> {
        debug!("popping cleanup scope {}, {} scopes remaining",
               self.top_scope(|s| s.block_name("")),
               self.scopes_len() - 1);

        let mut scopes = self.scopes.borrow_mut();
        scopes.get().pop().unwrap()
    }

    fn top_scope<R>(&self, f: |&CleanupScope<'a>| -> R) -> R {
        let scopes = self.scopes.borrow();
        f(scopes.get().last().unwrap())
    }

    fn trans_cleanups_to_exit_scope(&'a self,
                                    label: EarlyExitLabel)
                                    -> BasicBlockRef {
        /*!
         * Used when the caller wishes to jump to an early exit, such
         * as a return, break, continue, or unwind. This function will
         * generate all cleanups between the top of the stack and the
         * exit `label` and return a basic block that the caller can
         * branch to.
         *
         * For example, if the current stack of cleanups were as follows:
         *
         *      AST 22
         *      Custom 1
         *      AST 23
         *      Loop 23
         *      Custom 2
         *      AST 24
         *
         * and the `label` specifies a break from `Loop 23`, then this
         * function would generate a series of basic blocks as follows:
         *
         *      Cleanup(AST 24) -> Cleanup(Custom 2) -> break_blk
         *
         * where `break_blk` is the block specified in `Loop 23` as
         * the target for breaks. The return value would be the first
         * basic block in that sequence (`Cleanup(AST 24)`). The
         * caller could then branch to `Cleanup(AST 24)` and it will
         * perform all cleanups and finally branch to the `break_blk`.
         */

        debug!("trans_cleanups_to_exit_scope label={:?} scopes={}",
               label, self.scopes_len());

        let orig_scopes_len = self.scopes_len();
        let mut prev_llbb;
        let mut popped_scopes = opt_vec::Empty;

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
                    UnwindExit => {
                        // Generate a block that will `Resume`.
                        let prev_bcx = self.new_block(true, "resume", None);
                        let personality = self.personality.get().expect(
                            "create_landing_pad() should have set this");
                        build::Resume(prev_bcx,
                                      build::Load(prev_bcx, personality));
                        prev_llbb = prev_bcx.llbb;
                        break;
                    }

                    ReturnExit => {
                        prev_llbb = self.get_llreturn();
                        break;
                    }

                    LoopExit(id, _) => {
                        self.ccx.tcx.sess.bug(format!(
                                "cannot exit from scope {:?}, \
                                not in scope", id));
                    }
                }
            }

            // Check if we have already cached the unwinding of this
            // scope for this label. If so, we can stop popping scopes
            // and branch to the cached label, since it contains the
            // cleanups for any subsequent scopes.
            match self.top_scope(|s| s.cached_early_exit(label)) {
                Some(cleanup_block) => {
                    prev_llbb = cleanup_block;
                    break;
                }
                None => { }
            }

            // Pop off the scope, since we will be generating
            // unwinding code for it. If we are searching for a loop exit,
            // and this scope is that loop, then stop popping and set
            // `prev_llbb` to the appropriate exit block from the loop.
            popped_scopes.push(self.pop_scope());
            let scope = popped_scopes.last().unwrap();
            match label {
                UnwindExit | ReturnExit => { }
                LoopExit(id, exit) => {
                    match scope.kind.early_exit_block(id, exit) {
                        Some(exitllbb) => {
                            prev_llbb = exitllbb;
                            break;
                        }

                        None => { }
                    }
                }
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
        while !popped_scopes.is_empty() {
            let mut scope = popped_scopes.pop().unwrap();

            if scope.cleanups.iter().any(|c| cleanup_is_suitable_for(*c, label))
            {
                let name = scope.block_name("clean");
                debug!("generating cleanups for {}", name);
                let bcx_in = self.new_block(label.is_unwind(), name, None);
                let mut bcx_out = bcx_in;
                for cleanup in scope.cleanups.iter().rev() {
                    if cleanup_is_suitable_for(*cleanup, label) {
                        bcx_out = cleanup.trans(bcx_out);
                    }
                }
                build::Br(bcx_out, prev_llbb);
                prev_llbb = bcx_in.llbb;
            } else {
                debug!("no suitable cleanups in {}",
                       scope.block_name("clean"));
            }

            scope.add_cached_early_exit(label, prev_llbb);
            self.push_scope(scope);
        }

        debug!("trans_cleanups_to_exit_scope: prev_llbb={}", prev_llbb);

        assert_eq!(self.scopes_len(), orig_scopes_len);
        prev_llbb
    }

    fn get_or_create_landing_pad(&'a self) -> BasicBlockRef {
        /*!
         * Creates a landing pad for the top scope, if one does not
         * exist.  The landing pad will perform all cleanups necessary
         * for an unwind and then `resume` to continue error
         * propagation:
         *
         *     landing_pad -> ... cleanups ... -> [resume]
         *
         * (The cleanups and resume instruction are created by
         * `trans_cleanups_to_exit_scope()`, not in this function
         * itself.)
         */

        let pad_bcx;

        debug!("get_or_create_landing_pad");

        // Check if a landing pad block exists; if not, create one.
        {
            let mut scopes = self.scopes.borrow_mut();
            let last_scope = scopes.get().mut_last().unwrap();
            match last_scope.cached_landing_pad {
                Some(llbb) => { return llbb; }
                None => {
                    let name = last_scope.block_name("unwind");
                    pad_bcx = self.new_block(true, name, None);
                    last_scope.cached_landing_pad = Some(pad_bcx.llbb);
                }
            }
        }

        // The landing pad return type (the type being propagated). Not sure what
        // this represents but it's determined by the personality function and
        // this is what the EH proposal example uses.
        let llretty = Type::struct_([Type::i8p(), Type::i32()], false);

        // The exception handling personality function.
        let def_id = common::langcall(pad_bcx, None, "", EhPersonalityLangItem);
        let llpersonality = callee::trans_fn_ref(pad_bcx, def_id, 0, false);

        // The only landing pad clause will be 'cleanup'
        let llretval = build::LandingPad(pad_bcx, llretty, llpersonality, 1u);

        // The landing pad block is a cleanup
        build::SetCleanup(pad_bcx, llretval);

        // We store the retval in a function-central alloca, so that calls to
        // Resume can find it.
        match self.personality.get() {
            Some(addr) => {
                build::Store(pad_bcx, llretval, addr);
            }
            None => {
                let addr = base::alloca(pad_bcx, common::val_ty(llretval), "");
                self.personality.set(Some(addr));
                build::Store(pad_bcx, llretval, addr);
            }
        }

        // Generate the cleanup block and branch to it.
        let cleanup_llbb = self.trans_cleanups_to_exit_scope(UnwindExit);
        build::Br(pad_bcx, cleanup_llbb);

        return pad_bcx.llbb;
    }
}

impl<'a> CleanupScope<'a> {
    fn new(kind: CleanupScopeKind<'a>) -> CleanupScope<'a> {
        CleanupScope {
            kind: kind,
            cleanups: opt_vec::Empty,
            cached_early_exits: opt_vec::Empty,
            cached_landing_pad: None,
        }
    }

    fn clear_cached_exits(&mut self) {
        self.cached_early_exits = opt_vec::Empty;
        self.cached_landing_pad = None;
    }

    fn cached_early_exit(&self,
                         label: EarlyExitLabel)
                         -> Option<BasicBlockRef> {
        self.cached_early_exits.iter().
            find(|e| e.label == label).
            map(|e| e.cleanup_block)
    }

    fn add_cached_early_exit(&mut self,
                             label: EarlyExitLabel,
                             blk: BasicBlockRef) {
        self.cached_early_exits.push(
            CachedEarlyExit { label: label,
                              cleanup_block: blk });
    }

    fn needs_invoke(&self) -> bool {
        /*! True if this scope has cleanups for use during unwinding */

        self.cached_landing_pad.is_some() ||
            self.cleanups.iter().any(|c| c.clean_on_unwind())
    }

    fn block_name(&self, prefix: &str) -> ~str {
        /*!
         * Returns a suitable name to use for the basic block that
         * handles this cleanup scope
         */

        match self.kind {
            CustomScopeKind => format!("{}_custom_", prefix),
            AstScopeKind(id) => format!("{}_ast_{}_", prefix, id),
            LoopScopeKind(id, _) => format!("{}_loop_{}_", prefix, id),
        }
    }
}

impl<'a> CleanupScopeKind<'a> {
    fn is_temp(&self) -> bool {
        match *self {
            CustomScopeKind => true,
            LoopScopeKind(..) | AstScopeKind(..) => false,
        }
    }

    fn is_ast_with_id(&self, id: ast::NodeId) -> bool {
        match *self {
            CustomScopeKind | LoopScopeKind(..) => false,
            AstScopeKind(i) => i == id
        }
    }

    fn is_loop_with_id(&self, id: ast::NodeId) -> bool {
        match *self {
            CustomScopeKind | AstScopeKind(..) => false,
            LoopScopeKind(i, _) => i == id
        }
    }

    fn early_exit_block(&self,
                        id: ast::NodeId,
                        exit: uint) -> Option<BasicBlockRef> {
        /*!
         * If this is a loop scope with id `id`, return the early
         * exit block `exit`, else `None`
         */

        match *self {
            LoopScopeKind(i, ref exits) if id == i => Some(exits[exit].llbb),
            _ => None,
        }
    }
}

impl EarlyExitLabel {
    fn is_unwind(&self) -> bool {
        match *self {
            UnwindExit => true,
            _ => false
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Cleanup types

pub struct DropValue {
    is_immediate: bool,
    on_unwind: bool,
    val: ValueRef,
    ty: ty::t,
}

impl Cleanup for DropValue {
    fn clean_on_unwind(&self) -> bool {
        self.on_unwind
    }

    fn trans<'a>(&self, bcx: &'a Block<'a>) -> &'a Block<'a> {
        if self.is_immediate {
            glue::drop_ty_immediate(bcx, self.val, self.ty)
        } else {
            glue::drop_ty(bcx, self.val, self.ty)
        }
    }
}

pub struct FreeValue {
    ptr: ValueRef,
    heap: common::heap,
}

impl Cleanup for FreeValue {
    fn clean_on_unwind(&self) -> bool {
        true
    }

    fn trans<'a>(&self, bcx: &'a Block<'a>) -> &'a Block<'a> {
        match self.heap {
            common::heap_managed => {
                glue::trans_free(bcx, self.ptr)
            }
            common::heap_exchange | common::heap_exchange_closure => {
                glue::trans_exchange_free(bcx, self.ptr)
            }
        }
    }
}

pub fn temporary_scope(tcx: ty::ctxt,
                       id: ast::NodeId)
                       -> ScopeId {
    match tcx.region_maps.temporary_scope(id) {
        Some(scope) => {
            let r = AstScope(scope);
            debug!("temporary_scope({}) = {:?}", id, r);
            r
        }
        None => {
            tcx.sess.bug(format!("no temporary scope available for expr {}", id))
        }
    }
}

pub fn var_scope(tcx: ty::ctxt,
                 id: ast::NodeId)
                 -> ScopeId {
    let r = AstScope(tcx.region_maps.var_scope(id));
    debug!("var_scope({}) = {:?}", id, r);
    r
}

fn cleanup_is_suitable_for(c: &Cleanup,
                           label: EarlyExitLabel) -> bool {
    !label.is_unwind() || c.clean_on_unwind()
}

///////////////////////////////////////////////////////////////////////////
// These traits just exist to put the methods into this file.

pub trait CleanupMethods<'a> {
    fn push_ast_cleanup_scope(&self, id: ast::NodeId);
    fn push_loop_cleanup_scope(&self,
                                   id: ast::NodeId,
                                   exits: [&'a Block<'a>, ..EXIT_MAX]);
    fn push_custom_cleanup_scope(&self) -> CustomScopeIndex;
    fn pop_and_trans_ast_cleanup_scope(&self,
                                              bcx: &'a Block<'a>,
                                              cleanup_scope: ast::NodeId)
                                              -> &'a Block<'a>;
    fn pop_loop_cleanup_scope(&self,
                              cleanup_scope: ast::NodeId);
    fn pop_custom_cleanup_scope(&self,
                                custom_scope: CustomScopeIndex);
    fn pop_and_trans_custom_cleanup_scope(&self,
                                          bcx: &'a Block<'a>,
                                          custom_scope: CustomScopeIndex)
                                          -> &'a Block<'a>;
    fn top_loop_scope(&self) -> ast::NodeId;
    fn normal_exit_block(&'a self,
                         cleanup_scope: ast::NodeId,
                         exit: uint) -> BasicBlockRef;
    fn return_exit_block(&'a self) -> BasicBlockRef;
    fn schedule_drop_mem(&self,
                         cleanup_scope: ScopeId,
                         val: ValueRef,
                         ty: ty::t);
    fn schedule_drop_immediate(&self,
                               cleanup_scope: ScopeId,
                               val: ValueRef,
                               ty: ty::t);
    fn schedule_free_value(&self,
                           cleanup_scope: ScopeId,
                           val: ValueRef,
                           heap: common::heap);
    fn schedule_clean(&self,
                      cleanup_scope: ScopeId,
                      cleanup: ~Cleanup);
    fn schedule_clean_in_ast_scope(&self,
                                   cleanup_scope: ast::NodeId,
                                   cleanup: ~Cleanup);
    fn schedule_clean_in_custom_scope(&self,
                                    custom_scope: CustomScopeIndex,
                                    cleanup: ~Cleanup);
    fn needs_invoke(&self) -> bool;
    fn get_landing_pad(&'a self) -> BasicBlockRef;
}

trait CleanupHelperMethods<'a> {
    fn top_ast_scope(&self) -> Option<ast::NodeId>;
    fn top_nonempty_cleanup_scope(&self) -> Option<uint>;
    fn is_valid_to_pop_custom_scope(&self, custom_scope: CustomScopeIndex) -> bool;
    fn is_valid_custom_scope(&self, custom_scope: CustomScopeIndex) -> bool;
    fn trans_scope_cleanups(&self,
                            bcx: &'a Block<'a>,
                            scope: &CleanupScope<'a>) -> &'a Block<'a>;
    fn trans_cleanups_to_exit_scope(&'a self,
                                    label: EarlyExitLabel)
                                    -> BasicBlockRef;
    fn get_or_create_landing_pad(&'a self) -> BasicBlockRef;
    fn scopes_len(&self) -> uint;
    fn push_scope(&self, scope: CleanupScope<'a>);
    fn pop_scope(&self) -> CleanupScope<'a>;
    fn top_scope<R>(&self, f: |&CleanupScope<'a>| -> R) -> R;
}
