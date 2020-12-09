/*!
Managing the scope stack. The scopes are tied to lexical scopes, so as
we descend the THIR, we push a scope on the stack, build its
contents, and then pop it off. Every scope is named by a
`region::Scope`.

### SEME Regions

When pushing a new [Scope], we record the current point in the graph (a
basic block); this marks the entry to the scope. We then generate more
stuff in the control-flow graph. Whenever the scope is exited, either
via a `break` or `return` or just by fallthrough, that marks an exit
from the scope. Each lexical scope thus corresponds to a single-entry,
multiple-exit (SEME) region in the control-flow graph.

For now, we record the `region::Scope` to each SEME region for later reference
(see caveat in next paragraph). This is because destruction scopes are tied to
them. This may change in the future so that MIR lowering determines its own
destruction scopes.

### Not so SEME Regions

In the course of building matches, it sometimes happens that certain code
(namely guards) gets executed multiple times. This means that the scope lexical
scope may in fact correspond to multiple, disjoint SEME regions. So in fact our
mapping is from one scope to a vector of SEME regions. Since the SEME regions
are disjoint, the mapping is still one-to-one for the set of SEME regions that
we're currently in.

Also in matches, the scopes assigned to arms are not always even SEME regions!
Each arm has a single region with one entry for each pattern. We manually
manipulate the scheduled drops in this scope to avoid dropping things multiple
times.

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
early exit occurs, the method `break_scope` is called. It is given the
current point in execution where the early exit occurs, as well as the
scope you want to branch to (note that all early exits from to some
other enclosing scope). `break_scope` will record the set of drops currently
scheduled in a [DropTree]. Later, before `in_breakable_scope` exits, the drops
will be added to the CFG.

Panics are handled in a similar fashion, except that the drops are added to the
MIR once the rest of the function has finished being lowered. If a terminator
can panic, call `diverge_from(block)` with the block containing the terminator
`block`.

### Breakable scopes

In addition to the normal scope stack, we track a loop scope stack
that contains only loops and breakable blocks. It tracks where a `break`,
`continue` or `return` should go to.

*/

use crate::build::matches::{ArmHasGuard, Candidate};
use crate::build::{BlockAnd, BlockAndExtension, BlockFrame, Builder, CFG};
use crate::thir::{Arm, Expr, ExprRef, LintLevel};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_index::vec::IndexVec;
use rustc_middle::middle::region;
use rustc_middle::mir::*;
use rustc_span::{Span, DUMMY_SP};

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

    /// The drop index that will drop everything in and below this scope on an
    /// unwind path.
    cached_unwind_block: Option<DropIdx>,

    /// The drop index that will drop everything in and below this scope on a
    /// generator drop path.
    cached_generator_drop_block: Option<DropIdx>,
}

#[derive(Clone, Copy, Debug)]
struct DropData {
    /// The `Span` where drop obligation was incurred (typically where place was
    /// declared)
    source_info: SourceInfo,

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
    /// The scope that the destination should have its drop scheduled in.
    destination_scope: Option<region::Scope>,
    /// Drops that happen on the `break`/`return` path.
    break_drops: DropTree,
    /// Drops that happen on the `continue` path.
    continue_drops: Option<DropTree>,
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

const ROOT_NODE: DropIdx = DropIdx::from_u32(0);

/// A tree of drops that we have deferred lowering. It's used for:
///
/// * Drops on unwind paths
/// * Drops on generator drop paths (when a suspended generator is dropped)
/// * Drops on return and loop exit paths
///
/// Once no more nodes could be added to the tree, we lower it to MIR in one go
/// in `build_mir`.
#[derive(Debug)]
struct DropTree {
    /// Drops in the tree.
    drops: IndexVec<DropIdx, (DropData, DropIdx)>,
    /// Map for finding the inverse of the `next_drop` relation:
    ///
    /// `previous_drops[(drops[i].1, drops[i].0.local, drops[i].0.kind)] == i`
    previous_drops: FxHashMap<(DropIdx, Local, DropKind), DropIdx>,
    /// Edges into the `DropTree` that need to be added once it's lowered.
    entry_points: Vec<(DropIdx, BasicBlock)>,
}

impl Scope {
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

    fn invalidate_cache(&mut self) {
        self.cached_unwind_block = None;
        self.cached_generator_drop_block = None;
    }
}

/// A trait that determined how [DropTree] creates its blocks and
/// links to any entry nodes.
trait DropTreeBuilder<'tcx> {
    /// Create a new block for the tree. This should call either
    /// `cfg.start_new_block()` or `cfg.start_new_cleanup_block()`.
    fn make_block(cfg: &mut CFG<'tcx>) -> BasicBlock;

    /// Links a block outside the drop tree, `from`, to the block `to` inside
    /// the drop tree.
    fn add_entry(cfg: &mut CFG<'tcx>, from: BasicBlock, to: BasicBlock);
}

impl DropTree {
    fn new() -> Self {
        // The root node of the tree doesn't represent a drop, but instead
        // represents the block in the tree that should be jumped to once all
        // of the required drops have been performed.
        let fake_source_info = SourceInfo::outermost(DUMMY_SP);
        let fake_data =
            DropData { source_info: fake_source_info, local: Local::MAX, kind: DropKind::Storage };
        let drop_idx = DropIdx::MAX;
        let drops = IndexVec::from_elem_n((fake_data, drop_idx), 1);
        Self { drops, entry_points: Vec::new(), previous_drops: FxHashMap::default() }
    }

    fn add_drop(&mut self, drop: DropData, next: DropIdx) -> DropIdx {
        let drops = &mut self.drops;
        *self
            .previous_drops
            .entry((next, drop.local, drop.kind))
            .or_insert_with(|| drops.push((drop, next)))
    }

    fn add_entry(&mut self, from: BasicBlock, to: DropIdx) {
        debug_assert!(to < self.drops.next_index());
        self.entry_points.push((to, from));
    }

    /// Builds the MIR for a given drop tree.
    ///
    /// `blocks` should have the same length as `self.drops`, and may have its
    /// first value set to some already existing block.
    fn build_mir<'tcx, T: DropTreeBuilder<'tcx>>(
        &mut self,
        cfg: &mut CFG<'tcx>,
        blocks: &mut IndexVec<DropIdx, Option<BasicBlock>>,
    ) {
        debug!("DropTree::build_mir(drops = {:#?})", self);
        assert_eq!(blocks.len(), self.drops.len());

        self.assign_blocks::<T>(cfg, blocks);
        self.link_blocks(cfg, blocks)
    }

    /// Assign blocks for all of the drops in the drop tree that need them.
    fn assign_blocks<'tcx, T: DropTreeBuilder<'tcx>>(
        &mut self,
        cfg: &mut CFG<'tcx>,
        blocks: &mut IndexVec<DropIdx, Option<BasicBlock>>,
    ) {
        // StorageDead statements can share blocks with each other and also with
        // a Drop terminator. We iterate through the drops to find which drops
        // need their own block.
        #[derive(Clone, Copy)]
        enum Block {
            // This drop is unreachable
            None,
            // This drop is only reachable through the `StorageDead` with the
            // specified index.
            Shares(DropIdx),
            // This drop has more than one way of being reached, or it is
            // branched to from outside the tree, or its predecessor is a
            // `Value` drop.
            Own,
        }

        let mut needs_block = IndexVec::from_elem(Block::None, &self.drops);
        if blocks[ROOT_NODE].is_some() {
            // In some cases (such as drops for `continue`) the root node
            // already has a block. In this case, make sure that we don't
            // override it.
            needs_block[ROOT_NODE] = Block::Own;
        }

        // Sort so that we only need to check the last value.
        let entry_points = &mut self.entry_points;
        entry_points.sort();

        for (drop_idx, drop_data) in self.drops.iter_enumerated().rev() {
            if entry_points.last().map_or(false, |entry_point| entry_point.0 == drop_idx) {
                let block = *blocks[drop_idx].get_or_insert_with(|| T::make_block(cfg));
                needs_block[drop_idx] = Block::Own;
                while entry_points.last().map_or(false, |entry_point| entry_point.0 == drop_idx) {
                    let entry_block = entry_points.pop().unwrap().1;
                    T::add_entry(cfg, entry_block, block);
                }
            }
            match needs_block[drop_idx] {
                Block::None => continue,
                Block::Own => {
                    blocks[drop_idx].get_or_insert_with(|| T::make_block(cfg));
                }
                Block::Shares(pred) => {
                    blocks[drop_idx] = blocks[pred];
                }
            }
            if let DropKind::Value = drop_data.0.kind {
                needs_block[drop_data.1] = Block::Own;
            } else if drop_idx != ROOT_NODE {
                match &mut needs_block[drop_data.1] {
                    pred @ Block::None => *pred = Block::Shares(drop_idx),
                    pred @ Block::Shares(_) => *pred = Block::Own,
                    Block::Own => (),
                }
            }
        }

        debug!("assign_blocks: blocks = {:#?}", blocks);
        assert!(entry_points.is_empty());
    }

    fn link_blocks<'tcx>(
        &self,
        cfg: &mut CFG<'tcx>,
        blocks: &IndexVec<DropIdx, Option<BasicBlock>>,
    ) {
        for (drop_idx, drop_data) in self.drops.iter_enumerated().rev() {
            let block = if let Some(block) = blocks[drop_idx] {
                block
            } else {
                continue;
            };
            match drop_data.0.kind {
                DropKind::Value => {
                    let terminator = TerminatorKind::Drop {
                        target: blocks[drop_data.1].unwrap(),
                        // The caller will handle this if needed.
                        unwind: None,
                        place: drop_data.0.local.into(),
                    };
                    cfg.terminate(block, drop_data.0.source_info, terminator);
                }
                // Root nodes don't correspond to a drop.
                DropKind::Storage if drop_idx == ROOT_NODE => {}
                DropKind::Storage => {
                    let stmt = Statement {
                        source_info: drop_data.0.source_info,
                        kind: StatementKind::StorageDead(drop_data.0.local),
                    };
                    cfg.push(block, stmt);
                    let target = blocks[drop_data.1].unwrap();
                    if target != block {
                        // Diagnostics don't use this `Span` but debuginfo
                        // might. Since we don't want breakpoints to be placed
                        // here, especially when this is on an unwind path, we
                        // use `DUMMY_SP`.
                        let source_info = SourceInfo { span: DUMMY_SP, ..drop_data.0.source_info };
                        let terminator = TerminatorKind::Goto { target };
                        cfg.terminate(block, source_info, terminator);
                    }
                }
            }
        }
    }
}

impl<'tcx> Scopes<'tcx> {
    pub(crate) fn new() -> Self {
        Self {
            scopes: Vec::new(),
            breakable_scopes: Vec::new(),
            unwind_drops: DropTree::new(),
            generator_drops: DropTree::new(),
        }
    }

    fn push_scope(&mut self, region_scope: (region::Scope, SourceInfo), vis_scope: SourceScope) {
        debug!("push_scope({:?})", region_scope);
        self.scopes.push(Scope {
            source_scope: vis_scope,
            region_scope: region_scope.0,
            region_scope_span: region_scope.1.span,
            drops: vec![],
            cached_unwind_block: None,
            cached_generator_drop_block: None,
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

    /// Returns the topmost active scope, which is known to be alive until
    /// the next scope expression.
    fn topmost(&self) -> region::Scope {
        self.scopes.last().expect("topmost_scope: no scopes present").region_scope
    }
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
        destination_scope: Option<region::Scope>,
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
            destination_scope,
            break_drops: DropTree::new(),
            continue_drops: loop_block.map(|_| DropTree::new()),
        };
        let continue_block = loop_block.map(|block| (block, self.diverge_cleanup()));
        self.scopes.breakable_scopes.push(scope);
        let normal_exit_block = f(self);
        let breakable_scope = self.scopes.breakable_scopes.pop().unwrap();
        assert!(breakable_scope.region_scope == region_scope);
        if let Some(drops) = breakable_scope.continue_drops {
            self.build_exit_tree(drops, continue_block);
        }
        let break_block = self.build_exit_tree(breakable_scope.break_drops, None);
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
        target: BreakableTarget,
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
        let (break_index, destination, dest_scope) = match target {
            BreakableTarget::Return => {
                let scope = &self.scopes.breakable_scopes[0];
                if scope.break_destination != Place::return_place() {
                    span_bug!(span, "`return` in item with no return scope");
                }
                (0, Some(scope.break_destination), scope.destination_scope)
            }
            BreakableTarget::Break(scope) => {
                let break_index = get_scope_index(scope);
                let scope = &self.scopes.breakable_scopes[break_index];
                (break_index, Some(scope.break_destination), scope.destination_scope)
            }
            BreakableTarget::Continue(scope) => {
                let break_index = get_scope_index(scope);
                (break_index, None, None)
            }
        };

        if let Some(destination) = destination {
            if let Some(value) = value {
                debug!("stmt_expr Break val block_context.push(SubExpr)");
                self.block_context.push(BlockFrame::SubExpr);
                unpack!(block = self.into(destination, dest_scope, block, value));
                dest_scope
                    .map(|scope| self.unschedule_drop(scope, destination.as_local().unwrap()));
                self.block_context.pop();
            } else {
                self.cfg.push_assign_unit(block, source_info, destination, self.hir.tcx())
            }
        } else {
            assert!(value.is_none(), "`return` and `break` should have a destination");
        }

        let region_scope = self.scopes.breakable_scopes[break_index].region_scope;
        let scope_index = self.scopes.scope_index(region_scope, span);
        let drops = if destination.is_some() {
            &mut self.scopes.breakable_scopes[break_index].break_drops
        } else {
            self.scopes.breakable_scopes[break_index].continue_drops.as_mut().unwrap()
        };
        let mut drop_idx = ROOT_NODE;
        for scope in &self.scopes.scopes[scope_index + 1..] {
            for drop in &scope.drops {
                drop_idx = drops.add_drop(*drop, drop_idx);
            }
        }
        drops.add_entry(block, drop_idx);

        // `build_drop_tree` doesn't have access to our source_info, so we
        // create a dummy terminator now. `TerminatorKind::Resume` is used
        // because MIR type checking will panic if it hasn't been overwritten.
        self.cfg.terminate(block, source_info, TerminatorKind::Resume);

        self.cfg.start_new_block().unit()
    }

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
        let needs_cleanup = self.scopes.scopes.last().map_or(false, |scope| scope.needs_cleanup());
        let is_generator = self.generator_kind.is_some();
        let unwind_to = if needs_cleanup { self.diverge_cleanup() } else { DropIdx::MAX };

        let scope = self.scopes.scopes.last().expect("leave_top_scope called with no scopes");
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
            inlined: None,
            inlined_parent_scope: None,
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
    crate fn local_scope(&self) -> region::Scope {
        self.scopes.topmost()
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
        let needs_drop = match drop_kind {
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

        // When building drops, we try to cache chains of drops to reduce the
        // number of `DropTree::add_drop` calls. This, however, means that
        // whenever we add a drop into a scope which already had some entries
        // in the drop tree built (and thus, cached) for it, we must invalidate
        // all caches which might branch into the scope which had a drop just
        // added to it. This is necessary, because otherwise some other code
        // might use the cache to branch into already built chain of drops,
        // essentially ignoring the newly added drop.
        //
        // For example consider there’s two scopes with a drop in each. These
        // are built and thus the caches are filled:
        //
        // +--------------------------------------------------------+
        // | +---------------------------------+                    |
        // | | +--------+     +-------------+  |  +---------------+ |
        // | | | return | <-+ | drop(outer) | <-+ |  drop(middle) | |
        // | | +--------+     +-------------+  |  +---------------+ |
        // | +------------|outer_scope cache|--+                    |
        // +------------------------------|middle_scope cache|------+
        //
        // Now, a new, inner-most scope is added along with a new drop into
        // both inner-most and outer-most scopes:
        //
        // +------------------------------------------------------------+
        // | +----------------------------------+                       |
        // | | +--------+      +-------------+  |   +---------------+   | +-------------+
        // | | | return | <+   | drop(new)   | <-+  |  drop(middle) | <--+| drop(inner) |
        // | | +--------+  |   | drop(outer) |  |   +---------------+   | +-------------+
        // | |             +-+ +-------------+  |                       |
        // | +---|invalid outer_scope cache|----+                       |
        // +----=----------------|invalid middle_scope cache|-----------+
        //
        // If, when adding `drop(new)` we do not invalidate the cached blocks for both
        // outer_scope and middle_scope, then, when building drops for the inner (right-most)
        // scope, the old, cached blocks, without `drop(new)` will get used, producing the
        // wrong results.
        //
        // Note that this code iterates scopes from the inner-most to the outer-most,
        // invalidating caches of each scope visited. This way bare minimum of the
        // caches gets invalidated. i.e., if a new drop is added into the middle scope, the
        // cache of outer scope stays intact.
        //
        // Since we only cache drops for the unwind path and the generator drop
        // path, we only need to invalidate the cache for drops that happen on
        // the unwind or generator drop paths. This means that for
        // non-generators we don't need to invalidate caches for `DropKind::Storage`.
        let invalidate_caches = needs_drop || self.generator_kind.is_some();
        for scope in self.scopes.scopes.iter_mut().rev() {
            if invalidate_caches {
                scope.invalidate_cache();
            }

            if scope.region_scope == region_scope {
                let region_scope_span =
                    region_scope.span(self.hir.tcx(), &self.hir.region_scope_tree);
                // Attribute scope exit drops to scope's closing brace.
                let scope_end = self.hir.tcx().sess.source_map().end_point(region_scope_span);

                scope.drops.push(DropData {
                    source_info: SourceInfo { span: scope_end, scope: scope.source_scope },
                    local,
                    kind: drop_kind,
                });

                return;
            }
        }

        span_bug!(span, "region scope {:?} not in scope to drop {:?}", region_scope, local);
    }

    /// Unschedule a drop. Used for `break`, `return` and `match` expressions,
    /// where `record_operands_moved` is not powerful enough.
    ///
    /// The given local is expected to have a value drop scheduled in the given
    /// scope and for that drop to be the most recent thing scheduled in that
    /// scope.
    fn unschedule_drop(&mut self, region_scope: region::Scope, local: Local) {
        if !self.hir.needs_drop(self.local_decls[local].ty) {
            return;
        }
        for scope in self.scopes.scopes.iter_mut().rev() {
            scope.invalidate_cache();

            if scope.region_scope == region_scope {
                let drop = scope.drops.pop();

                match drop {
                    Some(DropData { local: removed_local, kind: DropKind::Value, .. })
                        if removed_local == local =>
                    {
                        return;
                    }
                    _ => bug!(
                        "found wrong drop, expected value drop of {:?}, found {:?}",
                        local,
                        drop,
                    ),
                }
            }
        }

        bug!("region scope {:?} not in scope to unschedule drop of {:?}", region_scope, local);
    }

    /// Indicates that the "local operands" stored in `local` is
    /// *moved* at some point during execution (see `local_scope` for
    /// more information about what a "local operand" is -- in short,
    /// it's an intermediate operand created as part of preparing some
    /// MIR instruction). We use this information to suppress
    /// redundant drops. This results in less MIR, but also avoids spurious
    /// borrow check errors (c.f. #64391).
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
        let local_scope = self.local_scope();
        let scope = self.scopes.scopes.last_mut().unwrap();

        assert_eq!(
            scope.region_scope, local_scope,
            "local scope is not the topmost scope!",
        );

        // look for moves of a local variable, like `MOVE(_X)`
        let locals_moved = operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Copy(_) | Operand::Constant(_) => None,
                Operand::Move(place) => place.as_local(),
            })
            .collect::<FxHashSet<_>>();

        // Remove the drops for the moved operands.
        scope
            .drops
            .retain(|drop| drop.kind == DropKind::Storage || !locals_moved.contains(&drop.local));
        scope.invalidate_cache();
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
            Operand::Copy(place) | Operand::Move(place) => {
                if let Some(cond_temp) = place.as_local() {
                    // Manually drop the condition on both branches.
                    let top_scope = self.scopes.scopes.last_mut().unwrap();
                    let top_drop_data = top_scope.drops.pop().unwrap();
                    if self.generator_kind.is_some() {
                        top_scope.invalidate_cache();
                    }

                    match top_drop_data.kind {
                        DropKind::Value { .. } => {
                            bug!("Drop scheduled on top of condition variable")
                        }
                        DropKind::Storage => {
                            let source_info = top_drop_data.source_info;
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

    /// Returns the [DropIdx] for the innermost drop if the function unwound at
    /// this point. The `DropIdx` will be created if it doesn't already exist.
    fn diverge_cleanup(&mut self) -> DropIdx {
        let is_generator = self.generator_kind.is_some();
        let (uncached_scope, mut cached_drop) = self
            .scopes
            .scopes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(scope_idx, scope)| {
                scope.cached_unwind_block.map(|cached_block| (scope_idx + 1, cached_block))
            })
            .unwrap_or((0, ROOT_NODE));

        for scope in &mut self.scopes.scopes[uncached_scope..] {
            for drop in &scope.drops {
                if is_generator || drop.kind == DropKind::Value {
                    cached_drop = self.scopes.unwind_drops.add_drop(*drop, cached_drop);
                }
            }
            scope.cached_unwind_block = Some(cached_drop);
        }

        cached_drop
    }

    /// Prepares to create a path that performs all required cleanup for a
    /// terminator that can unwind at the given basic block.
    ///
    /// This path terminates in Resume. The path isn't created until after all
    /// of the non-unwind paths in this item have been lowered.
    crate fn diverge_from(&mut self, start: BasicBlock) {
        debug_assert!(
            matches!(
                self.cfg.block_data(start).terminator().kind,
                TerminatorKind::Assert { .. }
                | TerminatorKind::Call {..}
                | TerminatorKind::DropAndReplace { .. }
                | TerminatorKind::FalseUnwind { .. }
            ),
            "diverge_from called on block with terminator that cannot unwind."
        );

        let next_drop = self.diverge_cleanup();
        self.scopes.unwind_drops.add_entry(start, next_drop);
    }

    /// Sets up a path that performs all required cleanup for dropping a
    /// generator, starting from the given block that ends in
    /// [TerminatorKind::Yield].
    ///
    /// This path terminates in GeneratorDrop.
    crate fn generator_drop_cleanup(&mut self, yield_block: BasicBlock) {
        debug_assert!(
            matches!(
                self.cfg.block_data(yield_block).terminator().kind,
                TerminatorKind::Yield { .. }
            ),
            "generator_drop_cleanup called on block with non-yield terminator."
        );
        let (uncached_scope, mut cached_drop) = self
            .scopes
            .scopes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(scope_idx, scope)| {
                scope.cached_generator_drop_block.map(|cached_block| (scope_idx + 1, cached_block))
            })
            .unwrap_or((0, ROOT_NODE));

        for scope in &mut self.scopes.scopes[uncached_scope..] {
            for drop in &scope.drops {
                cached_drop = self.scopes.generator_drops.add_drop(*drop, cached_drop);
            }
            scope.cached_generator_drop_block = Some(cached_drop);
        }

        self.scopes.generator_drops.add_entry(yield_block, cached_drop);
    }

    /// Utility function for *non*-scope code to build their own drops
    crate fn build_drop_and_replace(
        &mut self,
        block: BasicBlock,
        span: Span,
        place: Place<'tcx>,
        value: Operand<'tcx>,
    ) -> BlockAnd<()> {
        let source_info = self.source_info(span);
        let next_target = self.cfg.start_new_block();

        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::DropAndReplace { place, value, target: next_target, unwind: None },
        );
        self.diverge_from(block);

        next_target.unit()
    }

    /// Creates an `Assert` terminator and return the success block.
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

        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::Assert { cond, expected, msg, target: success_block, cleanup: None },
        );
        self.diverge_from(block);

        success_block
    }

    /// Lower the arms and guards of a match.
    ///
    /// The decision tree should have already been created (by
    /// [Builder::lower_match_tree]).
    ///
    /// This is this module, and not in `build::matches` because we have to do
    /// some careful scope manipulation to have the drop of the destination be
    /// scheduled at the end of each arm and then cleared for the next arm.
    crate fn lower_match_arms(
        &mut self,
        destination: Place<'tcx>,
        destination_scope: Option<region::Scope>,
        scrutinee_place: Place<'tcx>,
        scrutinee_span: Span,
        arm_candidates: Vec<(&'_ Arm<'tcx>, Candidate<'_, 'tcx>)>,
        outer_source_info: SourceInfo,
        fake_borrow_temps: Vec<(Place<'tcx>, Local)>,
    ) -> BlockAnd<()> {
        if arm_candidates.is_empty() {
            // If there are no arms to schedule drops, then we have to do it
            // manually.
            if let Some(scope) = destination_scope {
                self.schedule_drop(
                    outer_source_info.span,
                    scope,
                    destination.as_local().unwrap(),
                    DropKind::Value,
                );
            }
            return self.cfg.start_new_block().unit();
        }
        let mut first_arm = true;
        let arm_end_blocks: Vec<_> = arm_candidates
            .into_iter()
            .map(|(arm, candidate)| {
                debug!("lowering arm {:?}\ncandidate = {:?}", arm, candidate);

                if first_arm {
                    first_arm = false;
                } else if let Some(scope) = destination_scope {
                    self.unschedule_drop(scope, destination.as_local().unwrap());
                }

                let arm_source_info = self.source_info(arm.span);
                let arm_scope = (arm.scope, arm_source_info);
                self.in_scope(arm_scope, arm.lint_level, |this| {
                    let body = this.hir.mirror(arm.body.clone());
                    let scope = this.declare_bindings(
                        None,
                        arm.span,
                        &arm.pattern,
                        ArmHasGuard(arm.guard.is_some()),
                        Some((Some(&scrutinee_place), scrutinee_span)),
                    );

                    let arm_block = this.bind_pattern(
                        outer_source_info,
                        candidate,
                        arm.guard.as_ref(),
                        &fake_borrow_temps,
                        scrutinee_span,
                        Some(arm.scope),
                    );

                    if let Some(source_scope) = scope {
                        this.source_scope = source_scope;
                    }

                    this.into(destination, destination_scope, arm_block, body)
                })
            })
            .collect();

        // all the arm blocks will rejoin here
        let end_block = self.cfg.start_new_block();

        for arm_block in arm_end_blocks {
            self.cfg.goto(unpack!(arm_block), outer_source_info, end_block);
        }

        self.source_scope = outer_source_info.scope;

        end_block.unit()
    }

    /// Unschedules any drops in the top scope.
    ///
    /// This is only needed for `match` arm scopes, because they have one
    /// entrance per pattern, but only one exit.
    pub(super) fn clear_top_scope(&mut self, region_scope: region::Scope) {
        let top_scope = self.scopes.scopes.last_mut().unwrap();

        assert_eq!(top_scope.region_scope, region_scope);

        top_scope.drops.clear();
        top_scope.invalidate_cache();
    }

    /// Unschedules the drop of the return place.
    ///
    /// If the return type of a function requires drop, then we schedule it
    /// in the outermost scope so that it's dropped if there's a panic while
    /// we drop any local variables. But we don't want to drop it if we
    /// return normally.
    crate fn unschedule_return_place_drop(&mut self) {
        assert_eq!(self.scopes.scopes.len(), 1);
        assert!(self.scopes.scopes[0].drops.len() <= 1);
        self.scopes.scopes[0].drops.clear();
    }
}

/// Builds drops for `pop_scope` and `leave_top_scope`.
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
        let source_info = drop_data.source_info;
        let local = drop_data.local;

        match drop_data.kind {
            DropKind::Value => {
                // `unwind_to` should drop the value that we're about to
                // schedule. If dropping this value panics, then we continue
                // with the *next* value on the unwind path.
                debug_assert_eq!(unwind_drops.drops[unwind_to].0.local, drop_data.local);
                debug_assert_eq!(unwind_drops.drops[unwind_to].0.kind, drop_data.kind);
                unwind_to = unwind_drops.drops[unwind_to].1;

                unwind_drops.add_entry(block, unwind_to);

                let next = cfg.start_new_block();
                cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Drop { place: local.into(), target: next, unwind: None },
                );
                block = next;
            }
            DropKind::Storage => {
                if storage_dead_on_unwind {
                    debug_assert_eq!(unwind_drops.drops[unwind_to].0.local, drop_data.local);
                    debug_assert_eq!(unwind_drops.drops[unwind_to].0.kind, drop_data.kind);
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

impl<'a, 'tcx: 'a> Builder<'a, 'tcx> {
    /// Build a drop tree for a breakable scope.
    ///
    /// If `continue_block` is `Some`, then the tree is for `continue` inside a
    /// loop. Otherwise this is for `break` or `return`. The `DropIdx` is the
    /// next drop in the case that the drop tree unwinds. This is needed
    /// because the drop of the break destination has already been scheduled
    /// but it hasn't been initialized on the `continue` paths.
    fn build_exit_tree(
        &mut self,
        mut drops: DropTree,
        continue_block: Option<(BasicBlock, DropIdx)>,
    ) -> Option<BlockAnd<()>> {
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        blocks[ROOT_NODE] = continue_block.map(|(block, _)| block);

        drops.build_mir::<ExitScopes>(&mut self.cfg, &mut blocks);

        // Link the exit drop tree to unwind drop tree.
        if drops.drops.iter().any(|(drop, _)| drop.kind == DropKind::Value) {
            let unwind_target = continue_block
                .map_or_else(|| self.diverge_cleanup(), |(_, unwind_target)| unwind_target);
            let mut unwind_indices = IndexVec::from_elem_n(unwind_target, 1);
            for (drop_idx, drop_data) in drops.drops.iter_enumerated().skip(1) {
                match drop_data.0.kind {
                    DropKind::Storage => {
                        if self.generator_kind.is_some() {
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

    /// Build the unwind and generator drop trees.
    crate fn build_drop_trees(&mut self, should_abort: bool) {
        if self.generator_kind.is_some() {
            self.build_generator_drop_trees(should_abort);
        } else {
            Self::build_unwind_tree(
                &mut self.cfg,
                &mut self.scopes.unwind_drops,
                self.fn_span,
                should_abort,
                &mut None,
            );
        }
    }

    fn build_generator_drop_trees(&mut self, should_abort: bool) {
        // Build the drop tree for dropping the generator while it's suspended.
        let drops = &mut self.scopes.generator_drops;
        let cfg = &mut self.cfg;
        let fn_span = self.fn_span;
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        drops.build_mir::<GeneratorDrop>(cfg, &mut blocks);
        if let Some(root_block) = blocks[ROOT_NODE] {
            cfg.terminate(
                root_block,
                SourceInfo::outermost(fn_span),
                TerminatorKind::GeneratorDrop,
            );
        }

        // Build the drop tree for unwinding in the normal control flow paths.
        let resume_block = &mut None;
        let unwind_drops = &mut self.scopes.unwind_drops;
        Self::build_unwind_tree(cfg, unwind_drops, fn_span, should_abort, resume_block);

        // Build the drop tree for unwinding when dropping a suspended
        // generator.
        //
        // This is a different tree to the standard unwind paths here to
        // prevent drop elaboration from creating drop flags that would have
        // to be captured by the generator. I'm not sure how important this
        // optimization is, but it is here.
        for (drop_idx, drop_data) in drops.drops.iter_enumerated() {
            if let DropKind::Value = drop_data.0.kind {
                debug_assert!(drop_data.1 < drops.drops.next_index());
                drops.entry_points.push((drop_data.1, blocks[drop_idx].unwrap()));
            }
        }
        Self::build_unwind_tree(cfg, drops, fn_span, should_abort, resume_block);
    }

    fn build_unwind_tree(
        cfg: &mut CFG<'tcx>,
        drops: &mut DropTree,
        fn_span: Span,
        should_abort: bool,
        resume_block: &mut Option<BasicBlock>,
    ) {
        let mut blocks = IndexVec::from_elem(None, &drops.drops);
        blocks[ROOT_NODE] = *resume_block;
        drops.build_mir::<Unwind>(cfg, &mut blocks);
        if let (None, Some(resume)) = (*resume_block, blocks[ROOT_NODE]) {
            // `TerminatorKind::Abort` is used for `#[unwind(aborts)]`
            // functions.
            let terminator =
                if should_abort { TerminatorKind::Abort } else { TerminatorKind::Resume };

            cfg.terminate(resume, SourceInfo::outermost(fn_span), terminator);

            *resume_block = blocks[ROOT_NODE];
        }
    }
}

// DropTreeBuilder implementations.

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
        let term = cfg.block_data_mut(from).terminator_mut();
        if let TerminatorKind::Yield { ref mut drop, .. } = term.kind {
            *drop = Some(to);
        } else {
            span_bug!(
                term.source_info.span,
                "cannot enter generator drop tree from {:?}",
                term.kind
            )
        }
    }
}

struct Unwind;

impl<'tcx> DropTreeBuilder<'tcx> for Unwind {
    fn make_block(cfg: &mut CFG<'tcx>) -> BasicBlock {
        cfg.start_new_cleanup_block()
    }
    fn add_entry(cfg: &mut CFG<'tcx>, from: BasicBlock, to: BasicBlock) {
        let term = &mut cfg.block_data_mut(from).terminator_mut();
        match &mut term.kind {
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
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::InlineAsm { .. } => {
                span_bug!(term.source_info.span, "cannot unwind from {:?}", term.kind)
            }
        }
    }
}
