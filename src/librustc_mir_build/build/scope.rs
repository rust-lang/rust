/*!
Managing the scope stack. The scopes are tied to lexical scopes, so as
we descend the HAIR, we push a scope on the stack, build its
contents, and then pop it off. Every scope is named by a
`region::Scope`.

### SEME Regions

When pushing a new scope, we record the current point in the graph (a
basic block); this marks the entry to the scope. We then generate more
stuff in the control-flow graph. Whenever the scope is exited, either
via a `break` or `return` or just by fallthrough, that marks an exit
from the scope. Each lexical scope thus corresponds to a single-entry,
multiple-exit (SEME) region in the control-flow graph.

For now, we keep a mapping from each `region::Scope` to its
corresponding SEME region for later reference (see caveat in next
paragraph). This is because region scopes are tied to
them. Eventually, when we shift to non-lexical lifetimes, there should
be no need to remember this mapping.

### Not so SEME Regions

In the course of building matches, it sometimes happens that certain code
(namely guards) gets executed multiple times. This means that the scope lexical
scope may in fact correspond to multiple, disjoint SEME regions. So in fact our
mapping is from one scope to a vector of SEME regions.

Also in matches, the scopes assigned to arms are not even SEME regions! Each
arm has a single region with one entry for each pattern. We manually
manipulate the scheduled drops in this scope to avoid dropping things multiple
times, although drop elaboration would clean this up for value drops.

### Drops

The primary purpose for scopes is to insert drops: while building
the contents, we also accumulate places that need to be dropped upon
exit from each scope. This is done by calling `schedule_drop`. Once a
drop is scheduled, whenever we branch out we will insert drops of all
those places onto the outgoing edge. Note that we don't know the full
set of scheduled drops up front, and so whenever we exit from the
scope we only drop the values scheduled thus far. For example, consider
the scope S corresponding to this loop:

```
# let cond = true;
loop {
    let x = ..;
    if cond { break; }
    let y = ..;
}
```

When processing the `let x`, we will add one drop to the scope for
`x`.  The break will then insert a drop for `x`. When we process `let
y`, we will add another drop (in fact, to a subscope, but let's ignore
that for now); any later drops would also drop `y`.

### Early exit

There are numerous "normal" ways to early exit a scope: `break`,
`continue`, `return` (panics are handled separately). Whenever an
early exit occurs, the method `exit_scope` is called. It is given the
current point in execution where the early exit occurs, as well as the
scope you want to branch to (note that all early exits from to some
other enclosing scope). `exit_scope` will record this exit point and
also add all drops.

Panics are handled in a similar fashion, except that a panic always
returns out to the `DIVERGE_BLOCK`. To trigger a panic, simply call
`panic(p)` with the current point `p`. Or else you can call
`diverge_cleanup`, which will produce a block that you can branch to
which does the appropriate cleanup and then diverges. `panic(p)`
simply calls `diverge_cleanup()` and adds an edge from `p` to the
result.

### Loop scopes

In addition to the normal scope stack, we track a loop scope stack
that contains only loops. It tracks where a `break` and `continue`
should go to.

*/

use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder, CFG};
use crate::hair::{Expr, ExprRef, LintLevel};
use rustc::middle::region;
use rustc::mir::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::GeneratorKind;
use rustc_index::vec::{Idx, IndexVec};
use rustc_span::{Span, DUMMY_SP};
use std::collections::hash_map::Entry;
use std::mem;

#[derive(Debug)]
pub struct Scopes<'tcx> {
    scopes: Vec<Scope>,
    /// The current set of breakable scopes. See module comment for more details.
    breakable_scopes: Vec<BreakableScope<'tcx>>,

    /// Drops that need to be done on unwind paths. See the comment on
    /// [DropTree] for more details.
    unwind_drops: DropTree,

    /// Drops that need to be done on paths to the `GeneratorDrop` terminator.
    generator_drops: DropTree,
    // TODO: what's this?
    // cached_unwind_drop: DropIdx,
}

#[derive(Debug)]
struct Scope {
    /// The source scope this scope was created in.
    source_scope: SourceScope,

    /// the region span of this scope within source code.
    region_scope: region::Scope,

    /// the span of that region_scope
    region_scope_span: Span,

    /// set of places to drop when exiting this scope. This starts
    /// out empty but grows as variables are declared during the
    /// building process. This is a stack, so we always drop from the
    /// end of the vector (top of the stack) first.
    drops: Vec<DropData>,

    moved_locals: Vec<Local>,
}

#[derive(Clone, Copy, Debug)]
struct DropData {
    /// The `Span` where drop obligation was incurred (typically where place was
    /// declared)
    span: Span,

    /// local to drop
    local: Local,

    /// Whether this is a value Drop or a StorageDead.
    kind: DropKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DropKind {
    Value,
    Storage,
}

#[derive(Debug)]
struct BreakableScope<'tcx> {
    /// Region scope of the loop
    region_scope: region::Scope,
    /// The destination of the loop/block expression itself (i.e., where to put
    /// the result of a `break` or `return` expression)
    break_destination: Place<'tcx>,
    /// Drops that happen on the
    drops: DropTree,
}

/// The target of an expression that breaks out of a scope
#[derive(Clone, Copy, Debug)]
crate enum BreakableTarget {
    Continue(region::Scope),
    Break(region::Scope),
    Return,
}

rustc_index::newtype_index! {
    struct DropIdx { .. }
}

const ROOT_NODE: DropIdx = DropIdx::from_u32_const(0);
const CONTINUE_NODE: DropIdx = DropIdx::from_u32_const(1);

/// A tree of drops that we have deferred lowering.
// TODO say some more.
#[derive(Debug)]
struct DropTree {
    /// The next item to drop, if there is one.
    // TODO actual comment
    drops: IndexVec<DropIdx, (DropData, DropIdx)>,
    /// Map for finding the inverse of the `next_drop` relation:
    ///
    /// `previous_drops[(next_drop[i], drops[i].local, drops[i].kind] == i`
    previous_drops: FxHashMap<(DropIdx, Local, DropKind), DropIdx>,
    /// Edges into the `DropTree` that need to be added once it's lowered.
    entry_points: Vec<(DropIdx, BasicBlock)>,
    /// The number of root nodes in the tree.
    num_roots: DropIdx,
}

impl Scope {
    /// Given a span and this scope's source scope, make a SourceInfo.
    fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo { span, scope: self.source_scope }
    }

    /// Whether there's anything to do for the cleanup path, that is,
    /// when unwinding through this scope. This includes destructors,
    /// but not StorageDead statements, which don't get emitted at all
    /// for unwinding, for several reasons:
    ///  * clang doesn't emit llvm.lifetime.end for C++ unwinding
    ///  * LLVM's memory dependency analysis can't handle it atm
    ///  * polluting the cleanup MIR with StorageDead creates
    ///    landing pads even though there's no actual destructors
    ///  * freeing up stack space has no effect during unwinding
    /// Note that for generators we do emit StorageDeads, for the
    /// use of optimizations in the MIR generator transform.
    fn needs_cleanup(&self) -> bool {
        self.drops.iter().any(|drop| match drop.kind {
            DropKind::Value => true,
            DropKind::Storage => false,
        })
    }
}

impl DropTree {
    fn new(num_roots: usize) -> Self {
        let fake_data = DropData { span: DUMMY_SP, local: Local::MAX, kind: DropKind::Storage };
        let drop_idx = DropIdx::MAX;
        let drops = IndexVec::from_elem_n((fake_data, drop_idx), num_roots);
        Self {
            drops,
            num_roots: DropIdx::from_usize(num_roots),
            entry_points: Vec::new(),
            previous_drops: FxHashMap::default(),
        }
    }

    fn add_drop(&mut self, drop: DropData, next: DropIdx) -> DropIdx {
        let drops = &mut self.drops;
        *self
            .previous_drops
            .entry((next, drop.local, drop.kind))
            .or_insert_with(|| drops.push((drop, next)))
    }

    fn add_entry(&mut self, from: BasicBlock, to: DropIdx) {
        self.entry_points.push((to, from));
    }
}

impl<'tcx> Scopes<'tcx> {
    pub(crate) fn new(is_generator: bool) -> Self {
        Self {
            scopes: Vec::new(),
            breakable_scopes: Vec::new(),
            unwind_drops: DropTree::new(1),
            generator_drops: DropTree::new(is_generator as usize),
        }
    }

    fn push_scope(&mut self, region_scope: (region::Scope, SourceInfo), vis_scope: SourceScope) {
        debug!("push_scope({:?})", region_scope);
        self.scopes.push(Scope {
            source_scope: vis_scope,
            region_scope: region_scope.0,
            region_scope_span: region_scope.1.span,
            drops: vec![],
            moved_locals: vec![],
        });
    }

    fn pop_scope(&mut self, region_scope: (region::Scope, SourceInfo)) -> Scope {
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.region_scope, region_scope.0);
        scope
    }

    fn scope_index(&self, region_scope: region::Scope, span: Span) -> usize {
        self.scopes
            .iter()
            .rposition(|scope| scope.region_scope == region_scope)
            .unwrap_or_else(|| span_bug!(span, "region_scope {:?} does not enclose", region_scope))
    }

    fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Scope> + '_ {
        self.scopes.iter_mut().rev()
    }

    /// Returns the topmost active scope, which is known to be alive until
    /// the next scope expression.
    fn topmost(&self) -> region::Scope {
        self.scopes.last().expect("topmost_scope: no scopes present").region_scope
    }

    //    fn source_info(&self, index: usize, span: Span) -> SourceInfo {
    //        self.scopes[self.len() - index].source_info(span)
    //    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    // Adding and removing scopes
    // ==========================
    //  Start a breakable scope, which tracks where `continue`, `break` and
    //  `return` should branch to.
    crate fn in_breakable_scope<F>(
        &mut self,
        loop_block: Option<BasicBlock>,
        break_destination: Place<'tcx>,
        span: Span,
        f: F,
    ) -> BlockAnd<()>
    where
        F: FnOnce(&mut Builder<'a, 'tcx>) -> Option<BlockAnd<()>>,
    {
        let region_scope = self.scopes.topmost();
        let scope = BreakableScope {
            region_scope,
            break_destination,
            drops: DropTree::new(1 + loop_block.is_some() as usize),
        };
        self.scopes.breakable_scopes.push(scope);
        let normal_exit_block = f(self);
        let breakable_scope = self.scopes.breakable_scopes.pop().unwrap();
        assert!(breakable_scope.region_scope == region_scope);
        let break_block = self.build_exit_tree(breakable_scope.drops, loop_block);
        match (normal_exit_block, break_block) {
            (Some(block), None) | (None, Some(block)) => block,
            (None, None) => self.cfg.start_new_block().unit(),
            (Some(normal_block), Some(exit_block)) => {
                let target = self.cfg.start_new_block();
                let source_info = self.source_info(span);
                self.cfg.terminate(
                    unpack!(normal_block),
                    source_info,
                    TerminatorKind::Goto { target },
                );
                self.cfg.terminate(
                    unpack!(exit_block),
                    source_info,
                    TerminatorKind::Goto { target },
                );
                target.unit()
            }
        }
    }

    crate fn in_opt_scope<F, R>(
        &mut self,
        opt_scope: Option<(region::Scope, SourceInfo)>,
        f: F,
    ) -> BlockAnd<R>
    where
        F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>,
    {
        debug!("in_opt_scope(opt_scope={:?})", opt_scope);
        if let Some(region_scope) = opt_scope {
            self.push_scope(region_scope);
        }
        let mut block;
        let rv = unpack!(block = f(self));
        if let Some(region_scope) = opt_scope {
            unpack!(block = self.pop_scope(region_scope, block));
        }
        debug!("in_scope: exiting opt_scope={:?} block={:?}", opt_scope, block);
        block.and(rv)
    }

    /// Convenience wrapper that pushes a scope and then executes `f`
    /// to build its contents, popping the scope afterwards.
    crate fn in_scope<F, R>(
        &mut self,
        region_scope: (region::Scope, SourceInfo),
        lint_level: LintLevel,
        f: F,
    ) -> BlockAnd<R>
    where
        F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>,
    {
        debug!("in_scope(region_scope={:?})", region_scope);
        let source_scope = self.source_scope;
        let tcx = self.hir.tcx();
        if let LintLevel::Explicit(current_hir_id) = lint_level {
            // Use `maybe_lint_level_root_bounded` with `root_lint_level` as a bound
            // to avoid adding Hir dependences on our parents.
            // We estimate the true lint roots here to avoid creating a lot of source scopes.

            let parent_root = tcx.maybe_lint_level_root_bounded(
                self.source_scopes[source_scope].local_data.as_ref().assert_crate_local().lint_root,
                self.hir.root_lint_level,
            );
            let current_root =
                tcx.maybe_lint_level_root_bounded(current_hir_id, self.hir.root_lint_level);

            if parent_root != current_root {
                self.source_scope = self.new_source_scope(
                    region_scope.1.span,
                    LintLevel::Explicit(current_root),
                    None,
                );
            }
        }
        self.push_scope(region_scope);
        let mut block;
        let rv = unpack!(block = f(self));
        unpack!(block = self.pop_scope(region_scope, block));
        self.source_scope = source_scope;
        debug!("in_scope: exiting region_scope={:?} block={:?}", region_scope, block);
        block.and(rv)
    }

    /// Push a scope onto the stack. You can then build code in this
    /// scope and call `pop_scope` afterwards. Note that these two
    /// calls must be paired; using `in_scope` as a convenience
    /// wrapper maybe preferable.
    crate fn push_scope(&mut self, region_scope: (region::Scope, SourceInfo)) {
        self.scopes.push_scope(region_scope, self.source_scope);
    }

    /// Pops a scope, which should have region scope `region_scope`,
    /// adding any drops onto the end of `block` that are needed.
    /// This must match 1-to-1 with `push_scope`.
    crate fn pop_scope(
        &mut self,
        region_scope: (region::Scope, SourceInfo),
        mut block: BasicBlock,
    ) -> BlockAnd<()> {
        debug!("pop_scope({:?}, {:?})", region_scope, block);

        block = self.leave_top_scope(block);

        self.scopes.pop_scope(region_scope);

        block.unit()
    }

    /// Sets up the drops for breaking from `block` to `target`.
    crate fn break_scope(
        &mut self,
        mut block: BasicBlock,
        value: Option<ExprRef<'tcx>>,
        scope: BreakableTarget,
        source_info: SourceInfo,
    ) -> BlockAnd<()> {
        let span = source_info.span;

        let get_scope_index = |scope: region::Scope| {
            // find the loop-scope by its `region::Scope`.
            self.scopes
                .breakable_scopes
                .iter()
                .rposition(|breakable_scope| breakable_scope.region_scope == scope)
                .unwrap_or_else(|| span_bug!(span, "no enclosing breakable scope found"))
        };
        let (break_index, destination) = match scope {
            BreakableTarget::Return => {
                let scope = &self.scopes.breakable_scopes[0];
                if scope.break_destination != Place::return_place() {
                    span_bug!(span, "`return` in item with no return scope");
                }
                (0, Some(scope.break_destination.clone()))
            }
            BreakableTarget::Break(scope) => {
                let break_index = get_scope_index(scope);
                (
                    break_index,
                    Some(self.scopes.breakable_scopes[break_index].break_destination.clone()),
                )
            }
            BreakableTarget::Continue(scope) => {
                let break_index = get_scope_index(scope);
                (break_index, None)
            }
        };

        if let Some(ref destination) = destination {
            if let Some(value) = value {
                debug!("stmt_expr Break val block_context.push(SubExpr)");
                self.block_context.push(BlockFrame::SubExpr);
                unpack!(block = self.into(destination, block, value));
                self.block_context.pop();
            } else {
                self.cfg.push_assign_unit(block, source_info, destination)
            }
        } else {
            assert!(value.is_none(), "`return` and `break` should have a destination");
        }

        let region_scope = self.scopes.breakable_scopes[break_index].region_scope;
        let scope_index = self.scopes.scope_index(region_scope, span);
        let exited_scopes = &self.scopes.scopes[scope_index + 1..];
        let scope_drops = exited_scopes.iter().flat_map(|scope| &scope.drops);

        let drops = &mut self.scopes.breakable_scopes[break_index].drops;
        let mut drop_idx = DropIdx::from_u32(destination.is_none() as u32);
        for drop in scope_drops {
            drop_idx = drops.add_drop(*drop, drop_idx);
        }
        drops.add_entry(block, drop_idx);
        // TODO: explain this hack!
        self.cfg.terminate(block, source_info, TerminatorKind::Resume);

        self.cfg.start_new_block().unit()
    }

    // TODO: use in pop_top_scope.
    crate fn exit_top_scope(
        &mut self,
        mut block: BasicBlock,
        target: BasicBlock,
        source_info: SourceInfo,
    ) {
        block = self.leave_top_scope(block);
        self.cfg.terminate(block, source_info, TerminatorKind::Goto { target });
    }

    fn leave_top_scope(&mut self, block: BasicBlock) -> BasicBlock {
        // If we are emitting a `drop` statement, we need to have the cached
        // diverge cleanup pads ready in case that drop panics.
        let scope = self.scopes.scopes.last().expect("exit_top_scope called with no scopes");
        let is_generator = self.generator_kind.is_some();
        let needs_cleanup = scope.needs_cleanup();

        let unwind_to = if needs_cleanup {
            let mut drops = self
                .scopes
                .scopes
                .iter()
                .flat_map(|scope| &scope.drops)
                .filter(|drop| is_generator || drop.kind == DropKind::Value);
            let mut next_drop = ROOT_NODE;
            let mut drop_info = drops.next().unwrap();
            for previous_drop_info in drops {
                next_drop = self.scopes.unwind_drops.add_drop(*drop_info, next_drop);
                drop_info = previous_drop_info;
            }
            next_drop
        } else {
            DropIdx::MAX
        };
        unpack!(build_scope_drops(
            &mut self.cfg,
            &mut self.scopes.unwind_drops,
            scope,
            block,
            unwind_to,
            is_generator && needs_cleanup,
            self.arg_count,
        ))
    }

    /// Sets up a path that performs all required cleanup for dropping a generator.
    ///
    /// This path terminates in GeneratorDrop. Returns the start of the path.
    /// None indicates there’s no cleanup to do at this point.
    crate fn generator_drop_cleanup(&mut self, yield_block: BasicBlock) {
        let drops = self.scopes.scopes.iter().flat_map(|scope| &scope.drops);
        let mut next_drop = ROOT_NODE;
        for drop in drops {
            next_drop = self.scopes.generator_drops.add_drop(*drop, next_drop);
        }
        self.scopes.generator_drops.add_entry(yield_block, next_drop);
    }

    /// Creates a new source scope, nested in the current one.
    crate fn new_source_scope(
        &mut self,
        span: Span,
        lint_level: LintLevel,
        safety: Option<Safety>,
    ) -> SourceScope {
        let parent = self.source_scope;
        debug!(
            "new_source_scope({:?}, {:?}, {:?}) - parent({:?})={:?}",
            span,
            lint_level,
            safety,
            parent,
            self.source_scopes.get(parent)
        );
        let scope_local_data = SourceScopeLocalData {
            lint_root: if let LintLevel::Explicit(lint_root) = lint_level {
                lint_root
            } else {
                self.source_scopes[parent].local_data.as_ref().assert_crate_local().lint_root
            },
            safety: safety.unwrap_or_else(|| {
                self.source_scopes[parent].local_data.as_ref().assert_crate_local().safety
            }),
        };
        self.source_scopes.push(SourceScopeData {
            span,
            parent_scope: Some(parent),
            local_data: ClearCrossCrate::Set(scope_local_data),
        })
    }

    /// Given a span and the current source scope, make a SourceInfo.
    crate fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo { span, scope: self.source_scope }
    }

    // Finding scopes
    // ==============
    /// Returns the scope that we should use as the lifetime of an
    /// operand. Basically, an operand must live until it is consumed.
    /// This is similar to, but not quite the same as, the temporary
    /// scope (which can be larger or smaller).
    ///
    /// Consider:
    ///
    ///     let x = foo(bar(X, Y));
    ///
    /// We wish to pop the storage for X and Y after `bar()` is
    /// called, not after the whole `let` is completed.
    ///
    /// As another example, if the second argument diverges:
    ///
    ///     foo(Box::new(2), panic!())
    ///
    /// We would allocate the box but then free it on the unwinding
    /// path; we would also emit a free on the 'success' path from
    /// panic, but that will turn out to be removed as dead-code.
    ///
    /// When building statics/constants, returns `None` since
    /// intermediate values do not have to be dropped in that case.
    crate fn local_scope(&self) -> Option<region::Scope> {
        match self.hir.body_owner_kind {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) =>
            // No need to free storage in this context.
            {
                None
            }
            hir::BodyOwnerKind::Closure | hir::BodyOwnerKind::Fn => Some(self.scopes.topmost()),
        }
    }

    // Scheduling drops
    // ================
    crate fn schedule_drop_storage_and_value(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        local: Local,
    ) {
        self.schedule_drop(span, region_scope, local, DropKind::Storage);
        self.schedule_drop(span, region_scope, local, DropKind::Value);
    }

    /// Indicates that `place` should be dropped on exit from `region_scope`.
    ///
    /// When called with `DropKind::Storage`, `place` shouldn't be the return
    /// place, or a function parameter.
    crate fn schedule_drop(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        local: Local,
        drop_kind: DropKind,
    ) {
        // TODO: add back in caching.
        let _needs_drop = match drop_kind {
            DropKind::Value => {
                if !self.hir.needs_drop(self.local_decls[local].ty) {
                    return;
                }
                true
            }
            DropKind::Storage => {
                if local.index() <= self.arg_count {
                    span_bug!(
                        span,
                        "`schedule_drop` called with local {:?} and arg_count {}",
                        local,
                        self.arg_count,
                    )
                }
                false
            }
        };

        let scope = self
            .scopes
            .iter_mut()
            .find(|scope| scope.region_scope == region_scope)
            .unwrap_or_else(|| {
                span_bug!(span, "region scope {:?} not in scope to drop {:?}", region_scope, local);
            });

        let region_scope_span = region_scope.span(self.hir.tcx(), &self.hir.region_scope_tree);
        // Attribute scope exit drops to scope's closing brace.
        let scope_end = self.hir.tcx().sess.source_map().end_point(region_scope_span);

        scope.drops.push(DropData { span: scope_end, local, kind: drop_kind });
    }

    /// Indicates that the "local operand" stored in `local` is
    /// *moved* at some point during execution (see `local_scope` for
    /// more information about what a "local operand" is -- in short,
    /// it's an intermediate operand created as part of preparing some
    /// MIR instruction). We use this information to suppress
    /// redundant drops on the non-unwind paths. This results in less
    /// MIR, but also avoids spurious borrow check errors
    /// (c.f. #64391).
    ///
    /// Example: when compiling the call to `foo` here:
    ///
    /// ```rust
    /// foo(bar(), ...)
    /// ```
    ///
    /// we would evaluate `bar()` to an operand `_X`. We would also
    /// schedule `_X` to be dropped when the expression scope for
    /// `foo(bar())` is exited. This is relevant, for example, if the
    /// later arguments should unwind (it would ensure that `_X` gets
    /// dropped). However, if no unwind occurs, then `_X` will be
    /// unconditionally consumed by the `call`:
    ///
    /// ```
    /// bb {
    ///   ...
    ///   _R = CALL(foo, _X, ...)
    /// }
    /// ```
    ///
    /// However, `_X` is still registered to be dropped, and so if we
    /// do nothing else, we would generate a `DROP(_X)` that occurs
    /// after the call. This will later be optimized out by the
    /// drop-elaboation code, but in the meantime it can lead to
    /// spurious borrow-check errors -- the problem, ironically, is
    /// not the `DROP(_X)` itself, but the (spurious) unwind pathways
    /// that it creates. See #64391 for an example.
    crate fn record_operands_moved(&mut self, operands: &[Operand<'tcx>]) {
        let scope = match self.local_scope() {
            None => {
                // if there is no local scope, operands won't be dropped anyway
                return;
            }

            Some(local_scope) => self
                .scopes
                .iter_mut()
                .find(|scope| scope.region_scope == local_scope)
                .unwrap_or_else(|| bug!("scope {:?} not found in scope list!", local_scope)),
        };

        // look for moves of a local variable, like `MOVE(_X)`
        let locals_moved = operands.iter().flat_map(|operand| match operand {
            Operand::Copy(_) | Operand::Constant(_) => None,
            Operand::Move(place) => place.as_local(),
        });

        for local in locals_moved {
            // check if we have a Drop for this operand and -- if so
            // -- add it to the list of moved operands. Note that this
            // local might not have been an operand created for this
            // call, it could come from other places too.
            if scope.drops.iter().any(|drop| drop.local == local && drop.kind == DropKind::Value) {
                scope.moved_locals.push(local);
            }
        }
    }

    // Other
    // =====
    /// Branch based on a boolean condition.
    ///
    /// This is a special case because the temporary for the condition needs to
    /// be dropped on both the true and the false arm.
    crate fn test_bool(
        &mut self,
        mut block: BasicBlock,
        condition: Expr<'tcx>,
        source_info: SourceInfo,
    ) -> (BasicBlock, BasicBlock) {
        let cond = unpack!(block = self.as_local_operand(block, condition));
        let true_block = self.cfg.start_new_block();
        let false_block = self.cfg.start_new_block();
        let term = TerminatorKind::if_(self.hir.tcx(), cond.clone(), true_block, false_block);
        self.cfg.terminate(block, source_info, term);

        match cond {
            // Don't try to drop a constant
            Operand::Constant(_) => (),
            // If constants and statics, we don't generate StorageLive for this
            // temporary, so don't try to generate StorageDead for it either.
            _ if self.local_scope().is_none() => (),
            Operand::Copy(place) | Operand::Move(place) => {
                if let Some(cond_temp) = place.as_local() {
                    // Manually drop the condition on both branches.
                    let top_scope = self.scopes.scopes.last_mut().unwrap();
                    let top_drop_data = top_scope.drops.pop().unwrap();

                    match top_drop_data.kind {
                        DropKind::Value { .. } => {
                            bug!("Drop scheduled on top of condition variable")
                        }
                        DropKind::Storage => {
                            let source_info = top_scope.source_info(top_drop_data.span);
                            let local = top_drop_data.local;
                            assert_eq!(local, cond_temp, "Drop scheduled on top of condition");
                            self.cfg.push(
                                true_block,
                                Statement { source_info, kind: StatementKind::StorageDead(local) },
                            );
                            self.cfg.push(
                                false_block,
                                Statement { source_info, kind: StatementKind::StorageDead(local) },
                            );
                        }
                    }
                } else {
                    bug!("Expected as_local_operand to produce a temporary");
                }
            }
        }

        (true_block, false_block)
    }

    fn diverge_cleanup(&mut self) -> DropIdx {
        let is_generator = self.is_generator;
        let drops = self
            .scopes
            .scopes
            .iter()
            .flat_map(|scope| &scope.drops)
            .filter(|drop| is_generator || drop.kind == DropKind::Value);
        let mut next_drop = ROOT_NODE;
        for drop in drops {
            next_drop = self.scopes.unwind_drops.add_drop(*drop, next_drop);
        }
        next_drop
    }

    /// Prepares to create a path that performs all required cleanup for
    /// unwinding.
    ///
    /// This path terminates in Resume. The path isn't created until after all
    /// of the non-unwind paths in this item have been lowered.
    crate fn diverge_from(&mut self, start: BasicBlock) {
        let next_drop = self.diverge_cleanup();
        self.scopes.unwind_drops.add_entry(start, next_drop);
    }

    /// Utility function for *non*-scope code to build their own drops
    crate fn build_drop_and_replace(
        &mut self,
        block: BasicBlock,
        span: Span,
        location: Place<'tcx>,
        value: Operand<'tcx>,
    ) -> BlockAnd<()> {
        let source_info = self.source_info(span);
        let next_target = self.cfg.start_new_block();
        self.diverge_from(block);
        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::DropAndReplace { location, value, target: next_target, unwind: None },
        );
        next_target.unit()
    }

    /// Creates an Assert terminator and return the success block.
    /// If the boolean condition operand is not the expected value,
    /// a runtime panic will be caused with the given message.
    crate fn assert(
        &mut self,
        block: BasicBlock,
        cond: Operand<'tcx>,
        expected: bool,
        msg: AssertMessage<'tcx>,
        span: Span,
    ) -> BasicBlock {
        let source_info = self.source_info(span);

        let success_block = self.cfg.start_new_block();
        self.diverge_from(block);

        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::Assert { cond, expected, msg, target: success_block, cleanup: None },
        );

        success_block
    }

    // `match` arm scopes
    // ==================
    /// Unschedules any drops in the top scope.
    ///
    /// This is only needed for `match` arm scopes, because they have one
    /// entrance per pattern, but only one exit.
    crate fn clear_top_scope(&mut self, region_scope: region::Scope) {
        let top_scope = self.scopes.scopes.last_mut().unwrap();

        assert_eq!(top_scope.region_scope, region_scope);

        top_scope.drops.clear();
    }
}

/// Builds drops for pop_scope and exit_scope.
fn build_scope_drops<'tcx>(
    cfg: &mut CFG<'tcx>,
    unwind_drops: &mut DropTree,
    scope: &Scope,
    mut block: BasicBlock,
    mut unwind_to: DropIdx,
    storage_dead_on_unwind: bool,
    arg_count: usize,
) -> BlockAnd<()> {
    debug!("build_scope_drops({:?} -> {:?})", block, scope);

    // Build up the drops in evaluation order. The end result will
    // look like:
    //
    // [SDs, drops[n]] --..> [SDs, drop[1]] -> [SDs, drop[0]] -> [[SDs]]
    //               |                    |                 |
    //               :                    |                 |
    //                                    V                 V
    // [drop[n]] -...-> [drop[1]] ------> [drop[0]] ------> [last_unwind_to]
    //
    // The horizontal arrows represent the execution path when the drops return
    // successfully. The downwards arrows represent the execution path when the
    // drops panic (panicking while unwinding will abort, so there's no need for
    // another set of arrows).
    //
    // For generators, we unwind from a drop on a local to its StorageDead
    // statement. For other functions we don't worry about StorageDead. The
    // drops for the unwind path should have already been generated by
    // `diverge_cleanup_gen`.

    for drop_data in scope.drops.iter().rev() {
        let source_info = scope.source_info(drop_data.span);
        let local = drop_data.local;

        match drop_data.kind {
            DropKind::Value => {
                // If the operand has been moved, and we are not on an unwind
                // path, then don't generate the drop. (We only take this into
                // account for non-unwind paths so as not to disturb the
                // caching mechanism.)
                if scope.moved_locals.iter().any(|&o| o == local) {
                    unwind_to = unwind_drops.drops[unwind_to].1;
                    continue;
                }

                unwind_drops.entry_points.push((unwind_to, block));

                let next = cfg.start_new_block();
                cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Drop { location: local.into(), target: next, unwind: None },
                );
                block = next;
            }
            DropKind::Storage => {
                if storage_dead_on_unwind {
                    unwind_to = unwind_drops.drops[unwind_to].1;
                }
                // Only temps and vars need their storage dead.
                assert!(local.index() > arg_count);
                cfg.push(block, Statement { source_info, kind: StatementKind::StorageDead(local) });
            }
        }
    }
    block.unit()
}

trait DropTreeBuilder<'tcx> {
    fn make_block(cfg: &mut CFG<'tcx>) -> BasicBlock;
    fn add_entry(cfg: &mut CFG<'tcx>, from: BasicBlock, to: BasicBlock);
}
impl<'a, 'tcx: 'a> Builder<'a, 'tcx> {
    fn build_exit_tree(
        &mut self,
        mut drops: DropTree,
        continue_block: Option<BasicBlock>,
    ) -> Option<BlockAnd<()>> {
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        if continue_block.is_some() {
            blocks[CONTINUE_NODE] = continue_block;
            debug_assert_eq!(drops.num_roots, DropIdx::new(2));
        } else {
            debug_assert_eq!(drops.num_roots, CONTINUE_NODE);
        }
        build_drop_tree::<ExitScopes>(&mut self.cfg, &mut drops, &mut blocks);
        if drops.drops.iter().any(|(drop, _)| drop.kind == DropKind::Value) {
            let unwind_target = self.diverge_cleanup();
            let num_roots = drops.num_roots.index();
            let mut unwind_indices = IndexVec::from_elem_n(unwind_target, num_roots);
            for (drop_idx, drop_data) in drops.drops.iter_enumerated().skip(num_roots) {
                match drop_data.0.kind {
                    DropKind::Storage => {
                        if self.is_generator {
                            let unwind_drop = self
                                .scopes
                                .unwind_drops
                                .add_drop(drop_data.0, unwind_indices[drop_data.1]);
                            unwind_indices.push(unwind_drop);
                        } else {
                            unwind_indices.push(unwind_indices[drop_data.1]);
                        }
                    }
                    DropKind::Value => {
                        let unwind_drop = self
                            .scopes
                            .unwind_drops
                            .add_drop(drop_data.0, unwind_indices[drop_data.1]);
                        self.scopes
                            .unwind_drops
                            .add_entry(blocks[drop_idx].unwrap(), unwind_indices[drop_data.1]);
                        unwind_indices.push(unwind_drop);
                    }
                }
            }
        }
        blocks[ROOT_NODE].map(BasicBlock::unit)
    }

    crate fn build_drop_trees(&mut self, should_abort: bool) {
        if self.is_generator {
            self.build_generator_drop_trees(should_abort);
        } else {
            Self::build_unwind_tree(
                &mut self.cfg,
                &mut self.scopes.unwind_drops,
                self.fn_span,
                should_abort,
            );
        }
    }

    fn build_generator_drop_trees(&mut self, should_abort: bool) {
        // Build the drop tree for dropping the generator while it's suspended.
        let drops = &mut self.scopes.generator_drops;
        let cfg = &mut self.cfg;
        let fn_span = self.fn_span;
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        build_drop_tree::<GeneratorDrop>(cfg, drops, &mut blocks);
        if let Some(root_block) = blocks[ROOT_NODE] {
            cfg.terminate(
                root_block,
                SourceInfo { scope: OUTERMOST_SOURCE_SCOPE, span: fn_span },
                TerminatorKind::GeneratorDrop,
            );
        }

        // Build the drop tree for unwinding in the normal control flow paths.
        let resume_block =
            Self::build_unwind_tree(cfg, &mut self.scopes.unwind_drops, fn_span, should_abort);

        // Build the drop tree for unwinding when dropping a suspended
        // generator.
        //
        // This is a different tree to the standard unwind paths here to
        // prevent drop elaboration from creating drop flags that would have
        // to be captured by the generator. I'm not sure how important this
        // optimization is, but it is here.
        for (drop_idx, drop_data) in drops.drops.iter_enumerated() {
            if let DropKind::Value = drop_data.0.kind {
                drops.entry_points.push((drop_data.1, blocks[drop_idx].unwrap()));
            }
        }
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        blocks[ROOT_NODE] = resume_block;
        build_drop_tree::<Unwind>(cfg, drops, &mut blocks);
        if let (None, Some(new_resume_block)) = (resume_block, blocks[ROOT_NODE]) {
            let terminator =
                if should_abort { TerminatorKind::Abort } else { TerminatorKind::Resume };
            cfg.terminate(
                new_resume_block,
                SourceInfo { scope: OUTERMOST_SOURCE_SCOPE, span: fn_span },
                terminator,
            );
        }
    }

    fn build_unwind_tree(
        cfg: &mut CFG<'tcx>,
        drops: &mut DropTree,
        fn_span: Span,
        should_abort: bool,
    ) -> Option<BasicBlock> {
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        build_drop_tree::<Unwind>(cfg, drops, &mut blocks);
        if let Some(resume_block) = blocks[ROOT_NODE] {
            let terminator =
                if should_abort { TerminatorKind::Abort } else { TerminatorKind::Resume };
            cfg.terminate(
                resume_block,
                SourceInfo { scope: OUTERMOST_SOURCE_SCOPE, span: fn_span },
                terminator,
            );
            Some(resume_block)
        } else {
            None
        }
    }
}

fn source_info(span: Span) -> SourceInfo {
    SourceInfo { span, scope: OUTERMOST_SOURCE_SCOPE }
}

fn build_drop_tree<'tcx, T: DropTreeBuilder<'tcx>>(
    cfg: &mut CFG<'tcx>,
    drops: &mut DropTree,
    blocks: &mut IndexVec<DropIdx, Option<BasicBlock>>,
) {
    debug!("build_drop_tree(drops = {:#?})", drops);
    // TODO: Some comment about this.
    #[derive(Clone, Copy)]
    enum NeedsBlock {
        NoPredecessor,
        CanShare(DropIdx),
        NeedsOwn,
    }

    // TODO: Split this into two functions.

    // If a drop has multiple predecessors, they need to be in separate blocks
    // so that they can both banch to the current drop.
    let mut needs_block = IndexVec::from_elem(NeedsBlock::NoPredecessor, &drops.drops);
    for root_idx in (ROOT_NODE..drops.num_roots).skip(1) {
        needs_block[root_idx] = NeedsBlock::NeedsOwn;
    }

    let entry_points = &mut drops.entry_points;
    entry_points.sort();

    for (drop_idx, drop_data) in drops.drops.iter_enumerated().rev() {
        if entry_points.last().map_or(false, |entry_point| entry_point.0 == drop_idx) {
            let block = *blocks[drop_idx].get_or_insert_with(|| T::make_block(cfg));
            needs_block[drop_idx] = NeedsBlock::NeedsOwn;
            while entry_points.last().map_or(false, |entry_point| entry_point.0 == drop_idx) {
                let entry_block = entry_points.pop().unwrap().1;
                T::add_entry(cfg, entry_block, block);
            }
        }
        match needs_block[drop_idx] {
            NeedsBlock::NoPredecessor => continue,
            NeedsBlock::NeedsOwn => {
                blocks[drop_idx].get_or_insert_with(|| T::make_block(cfg));
            }
            NeedsBlock::CanShare(pred) => {
                blocks[drop_idx] = blocks[pred];
            }
        }
        if let DropKind::Value = drop_data.0.kind {
            needs_block[drop_data.1] = NeedsBlock::NeedsOwn;
        } else {
            if drop_idx >= drops.num_roots {
                match &mut needs_block[drop_data.1] {
                    pred @ NeedsBlock::NoPredecessor => *pred = NeedsBlock::CanShare(drop_idx),
                    pred @ NeedsBlock::CanShare(_) => *pred = NeedsBlock::NeedsOwn,
                    NeedsBlock::NeedsOwn => (),
                }
            }
        }
    }
    assert!(entry_points.is_empty());
    debug!("build_drop_tree: blocks = {:#?}", blocks);

    for (drop_idx, drop_data) in drops.drops.iter_enumerated().rev() {
        if let NeedsBlock::NoPredecessor = needs_block[drop_idx] {
            continue;
        }
        match drop_data.0.kind {
            DropKind::Value => {
                let terminator = TerminatorKind::Drop {
                    target: blocks[drop_data.1].unwrap(),
                    // TODO: The caller will register this if needed.
                    unwind: None,
                    location: drop_data.0.local.into(),
                };
                cfg.terminate(blocks[drop_idx].unwrap(), source_info(drop_data.0.span), terminator);
            }
            // Root nodes don't correspond to a drop.
            DropKind::Storage if drop_idx < drops.num_roots => {}
            DropKind::Storage => {
                let block = blocks[drop_idx].unwrap();
                let stmt = Statement {
                    source_info: source_info(drop_data.0.span),
                    kind: StatementKind::StorageDead(drop_data.0.local),
                };
                cfg.push(block, stmt);
                let target = blocks[drop_data.1].unwrap();
                if target != block {
                    let terminator = TerminatorKind::Goto { target };
                    cfg.terminate(block, source_info(drop_data.0.span), terminator);
                }
            }
        }
    }
}

struct ExitScopes;

impl<'tcx> DropTreeBuilder<'tcx> for ExitScopes {
    fn make_block(cfg: &mut CFG<'tcx>) -> BasicBlock {
        cfg.start_new_block()
    }
    fn add_entry(cfg: &mut CFG<'tcx>, from: BasicBlock, to: BasicBlock) {
        cfg.block_data_mut(from).terminator_mut().kind = TerminatorKind::Goto { target: to };
    }
}

struct GeneratorDrop;

impl<'tcx> DropTreeBuilder<'tcx> for GeneratorDrop {
    fn make_block(cfg: &mut CFG<'tcx>) -> BasicBlock {
        cfg.start_new_block()
    }
    fn add_entry(cfg: &mut CFG<'tcx>, from: BasicBlock, to: BasicBlock) {
        let kind = &mut cfg.block_data_mut(from).terminator_mut().kind;
        if let TerminatorKind::Yield { drop, .. } = kind {
            *drop = Some(to);
        };
    }
}

struct Unwind;

impl<'tcx> DropTreeBuilder<'tcx> for Unwind {
    fn make_block(cfg: &mut CFG<'tcx>) -> BasicBlock {
        cfg.start_new_cleanup_block()
    }
    fn add_entry(cfg: &mut CFG<'tcx>, from: BasicBlock, to: BasicBlock) {
        let term = &mut cfg.block_data_mut(from).terminator_mut().kind;
        match term {
            TerminatorKind::Drop { unwind, .. }
            | TerminatorKind::DropAndReplace { unwind, .. }
            | TerminatorKind::FalseUnwind { unwind, .. }
            | TerminatorKind::Call { cleanup: unwind, .. }
            | TerminatorKind::Assert { cleanup: unwind, .. } => {
                *unwind = Some(to);
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::FalseEdges { .. } => bug!("cannot unwind from {:?}", term),
        }
    }
}
