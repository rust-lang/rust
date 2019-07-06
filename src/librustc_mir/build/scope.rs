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
use rustc::ty::Ty;
use rustc::hir;
use rustc::mir::*;
use syntax_pos::{DUMMY_SP, Span};
use rustc_data_structures::fx::FxHashMap;
use std::collections::hash_map::Entry;
use std::mem;

#[derive(Debug)]
struct Scope {
    /// The source scope this scope was created in.
    source_scope: SourceScope,

    /// the region span of this scope within source code.
    region_scope: region::Scope,

    /// the span of that region_scope
    region_scope_span: Span,

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
    needs_cleanup: bool,

    /// set of places to drop when exiting this scope. This starts
    /// out empty but grows as variables are declared during the
    /// building process. This is a stack, so we always drop from the
    /// end of the vector (top of the stack) first.
    drops: Vec<DropData>,

    /// The cache for drop chain on “normal” exit into a particular BasicBlock.
    cached_exits: FxHashMap<(BasicBlock, region::Scope), BasicBlock>,

    /// The cache for drop chain on "generator drop" exit.
    cached_generator_drop: Option<BasicBlock>,

    /// The cache for drop chain on "unwind" exit.
    cached_unwind: CachedBlock,
}

#[derive(Debug, Default)]
pub struct Scopes<'tcx> {
    scopes: Vec<Scope>,
    /// The current set of breakable scopes. See module comment for more details.
    breakable_scopes: Vec<BreakableScope<'tcx>>,
}

#[derive(Debug)]
struct DropData {
    /// span where drop obligation was incurred (typically where place was declared)
    span: Span,

    /// local to drop
    local: Local,

    /// Whether this is a value Drop or a StorageDead.
    kind: DropKind,

    /// The cached blocks for unwinds.
    cached_block: CachedBlock,
}

#[derive(Debug, Default, Clone, Copy)]
struct CachedBlock {
    /// The cached block for the cleanups-on-diverge path. This block
    /// contains code to run the current drop and all the preceding
    /// drops (i.e., those having lower index in Drop’s Scope drop
    /// array)
    unwind: Option<BasicBlock>,

    /// The cached block for unwinds during cleanups-on-generator-drop path
    ///
    /// This is split from the standard unwind path here to prevent drop
    /// elaboration from creating drop flags that would have to be captured
    /// by the generator. I'm not sure how important this optimization is,
    /// but it is here.
    generator_drop: Option<BasicBlock>,
}

#[derive(Debug)]
pub(crate) enum DropKind {
    Value,
    Storage,
}

#[derive(Clone, Debug)]
struct BreakableScope<'tcx> {
    /// Region scope of the loop
    region_scope: region::Scope,
    /// Where the body of the loop begins. `None` if block
    continue_block: Option<BasicBlock>,
    /// Block to branch into when the loop or block terminates (either by being `break`-en out
    /// from, or by having its condition to become false)
    break_block: BasicBlock,
    /// The destination of the loop/block expression itself (i.e., where to put the result of a
    /// `break` expression)
    break_destination: Place<'tcx>,
}

/// The target of an expression that breaks out of a scope
#[derive(Clone, Copy, Debug)]
pub enum BreakableTarget {
    Continue(region::Scope),
    Break(region::Scope),
    Return,
}

impl CachedBlock {
    fn invalidate(&mut self) {
        self.generator_drop = None;
        self.unwind = None;
    }

    fn get(&self, generator_drop: bool) -> Option<BasicBlock> {
        if generator_drop {
            self.generator_drop
        } else {
            self.unwind
        }
    }

    fn ref_mut(&mut self, generator_drop: bool) -> &mut Option<BasicBlock> {
        if generator_drop {
            &mut self.generator_drop
        } else {
            &mut self.unwind
        }
    }
}

impl Scope {
    /// Invalidates all the cached blocks in the scope.
    ///
    /// Should always be run for all inner scopes when a drop is pushed into some scope enclosing a
    /// larger extent of code.
    ///
    /// `storage_only` controls whether to invalidate only drop paths that run `StorageDead`.
    /// `this_scope_only` controls whether to invalidate only drop paths that refer to the current
    /// top-of-scope (as opposed to dependent scopes).
    fn invalidate_cache(&mut self, storage_only: bool, is_generator: bool, this_scope_only: bool) {
        // FIXME: maybe do shared caching of `cached_exits` etc. to handle functions
        // with lots of `try!`?

        // cached exits drop storage and refer to the top-of-scope
        self.cached_exits.clear();

        // the current generator drop and unwind refer to top-of-scope
        self.cached_generator_drop = None;

        let ignore_unwinds = storage_only && !is_generator;
        if !ignore_unwinds {
            self.cached_unwind.invalidate();
        }

        if !ignore_unwinds && !this_scope_only {
            for drop_data in &mut self.drops {
                drop_data.cached_block.invalidate();
            }
        }
    }

    /// Given a span and this scope's source scope, make a SourceInfo.
    fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            span,
            scope: self.source_scope
        }
    }
}

impl<'tcx> Scopes<'tcx> {
    fn len(&self) -> usize {
        self.scopes.len()
    }

    fn push_scope(&mut self, region_scope: (region::Scope, SourceInfo), vis_scope: SourceScope) {
        debug!("push_scope({:?})", region_scope);
        self.scopes.push(Scope {
            source_scope: vis_scope,
            region_scope: region_scope.0,
            region_scope_span: region_scope.1.span,
            needs_cleanup: false,
            drops: vec![],
            cached_generator_drop: None,
            cached_exits: Default::default(),
            cached_unwind: CachedBlock::default(),
        });
    }

    fn pop_scope(
        &mut self,
        region_scope: (region::Scope, SourceInfo),
    ) -> (Scope, Option<BasicBlock>) {
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.region_scope, region_scope.0);
        let unwind_to = self.scopes.last()
            .and_then(|next_scope| next_scope.cached_unwind.get(false));
        (scope, unwind_to)
    }

    fn may_panic(&self, scope_count: usize) -> bool {
        let len = self.len();
        self.scopes[(len - scope_count)..].iter().any(|s| s.needs_cleanup)
    }

    /// Finds the breakable scope for a given label. This is used for
    /// resolving `return`, `break` and `continue`.
    fn find_breakable_scope(
        &self,
        span: Span,
        target: BreakableTarget,
    ) -> (BasicBlock, region::Scope, Option<Place<'tcx>>) {
        let get_scope = |scope: region::Scope| {
            // find the loop-scope by its `region::Scope`.
            self.breakable_scopes.iter()
                .rfind(|breakable_scope| breakable_scope.region_scope == scope)
                .unwrap_or_else(|| span_bug!(span, "no enclosing breakable scope found"))
        };
        match target {
            BreakableTarget::Return => {
                let scope = &self.breakable_scopes[0];
                if scope.break_destination != Place::RETURN_PLACE {
                    span_bug!(span, "`return` in item with no return scope");
                }
                (scope.break_block, scope.region_scope, Some(scope.break_destination.clone()))
            }
            BreakableTarget::Break(scope) => {
                let scope = get_scope(scope);
                (scope.break_block, scope.region_scope, Some(scope.break_destination.clone()))
            }
            BreakableTarget::Continue(scope) => {
                let scope = get_scope(scope);
                let continue_block = scope.continue_block
                    .unwrap_or_else(|| span_bug!(span, "missing `continue` block"));
                (continue_block, scope.region_scope, None)
            }
        }
    }

    fn num_scopes_above(&self, region_scope: region::Scope, span: Span) -> usize {
        let scope_count = self.scopes.iter().rev()
            .position(|scope| scope.region_scope == region_scope)
            .unwrap_or_else(|| {
                span_bug!(span, "region_scope {:?} does not enclose", region_scope)
            });
        let len = self.len();
        assert!(scope_count < len, "should not use `exit_scope` to pop ALL scopes");
        scope_count
    }

    fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item=&mut Scope> + '_ {
        self.scopes.iter_mut().rev()
    }

    fn top_scopes(&mut self, count: usize) -> impl DoubleEndedIterator<Item=&mut Scope> + '_ {
        let len = self.len();
        self.scopes[len - count..].iter_mut()
    }

    /// Returns the topmost active scope, which is known to be alive until
    /// the next scope expression.
    pub(super) fn topmost(&self) -> region::Scope {
        self.scopes.last().expect("topmost_scope: no scopes present").region_scope
    }

    fn source_info(&self, index: usize, span: Span) -> SourceInfo {
        self.scopes[self.len() - index].source_info(span)
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    // Adding and removing scopes
    // ==========================
    //  Start a breakable scope, which tracks where `continue`, `break` and
    //  `return` should branch to.
    pub fn in_breakable_scope<F, R>(&mut self,
                                    loop_block: Option<BasicBlock>,
                                    break_block: BasicBlock,
                                    break_destination: Place<'tcx>,
                                    f: F) -> R
        where F: FnOnce(&mut Builder<'a, 'tcx>) -> R
    {
        let region_scope = self.scopes.topmost();
        let scope = BreakableScope {
            region_scope,
            continue_block: loop_block,
            break_block,
            break_destination,
        };
        self.scopes.breakable_scopes.push(scope);
        let res = f(self);
        let breakable_scope = self.scopes.breakable_scopes.pop().unwrap();
        assert!(breakable_scope.region_scope == region_scope);
        res
    }

    pub fn in_opt_scope<F, R>(&mut self,
                              opt_scope: Option<(region::Scope, SourceInfo)>,
                              f: F)
                              -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_opt_scope(opt_scope={:?})", opt_scope);
        if let Some(region_scope) = opt_scope { self.push_scope(region_scope); }
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
    pub fn in_scope<F, R>(&mut self,
                          region_scope: (region::Scope, SourceInfo),
                          lint_level: LintLevel,
                          f: F)
                          -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_scope(region_scope={:?})", region_scope);
        let source_scope = self.source_scope;
        let tcx = self.hir.tcx();
        if let LintLevel::Explicit(current_hir_id) = lint_level {
            // Use `maybe_lint_level_root_bounded` with `root_lint_level` as a bound
            // to avoid adding Hir dependences on our parents.
            // We estimate the true lint roots here to avoid creating a lot of source scopes.

            let parent_root = tcx.maybe_lint_level_root_bounded(
                self.source_scope_local_data[source_scope].lint_root,
                self.hir.root_lint_level,
            );
            let current_root = tcx.maybe_lint_level_root_bounded(
                current_hir_id,
                self.hir.root_lint_level
            );

            if parent_root != current_root {
                self.source_scope = self.new_source_scope(
                    region_scope.1.span,
                    LintLevel::Explicit(current_root),
                    None
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
    pub fn push_scope(&mut self, region_scope: (region::Scope, SourceInfo)) {
        self.scopes.push_scope(region_scope, self.source_scope);
    }

    /// Pops a scope, which should have region scope `region_scope`,
    /// adding any drops onto the end of `block` that are needed.
    /// This must match 1-to-1 with `push_scope`.
    pub fn pop_scope(&mut self,
                     region_scope: (region::Scope, SourceInfo),
                     mut block: BasicBlock)
                     -> BlockAnd<()> {
        debug!("pop_scope({:?}, {:?})", region_scope, block);
        // If we are emitting a `drop` statement, we need to have the cached
        // diverge cleanup pads ready in case that drop panics.
        if self.scopes.may_panic(1) {
            self.diverge_cleanup();
        }
        let (scope, unwind_to) = self.scopes.pop_scope(region_scope);
        let unwind_to = unwind_to.unwrap_or_else(|| self.resume_block());

        unpack!(block = build_scope_drops(
            &mut self.cfg,
            self.is_generator,
            &scope,
            block,
            unwind_to,
            self.arg_count,
            false,
        ));

        block.unit()
    }

    pub fn break_scope(
        &mut self,
        mut block: BasicBlock,
        value: Option<ExprRef<'tcx>>,
        scope: BreakableTarget,
        source_info: SourceInfo,
    ) -> BlockAnd<()> {
        let (mut target_block, region_scope, destination)
            = self.scopes.find_breakable_scope(source_info.span, scope);
        if let BreakableTarget::Return = scope {
            // We call this now, rather than when we start lowering the
            // function so that the return block doesn't precede the entire
            // rest of the CFG. Some passes and LLVM prefer blocks to be in
            // approximately CFG order.
            target_block = self.return_block();
        }
        if let Some(destination) = destination {
            if let Some(value) = value {
                debug!("stmt_expr Break val block_context.push(SubExpr)");
                self.block_context.push(BlockFrame::SubExpr);
                unpack!(block = self.into(&destination, block, value));
                self.block_context.pop();
            } else {
                self.cfg.push_assign_unit(block, source_info, &destination)
            }
        } else {
            assert!(value.is_none(), "`return` and `break` should have a destination");
        }
        self.exit_scope(source_info.span, region_scope, block, target_block);
        self.cfg.start_new_block().unit()
    }

    /// Branch out of `block` to `target`, exiting all scopes up to
    /// and including `region_scope`. This will insert whatever drops are
    /// needed. See module comment for details.
    pub fn exit_scope(&mut self,
                      span: Span,
                      region_scope: region::Scope,
                      mut block: BasicBlock,
                      target: BasicBlock) {
        debug!("exit_scope(region_scope={:?}, block={:?}, target={:?})",
               region_scope, block, target);
        let scope_count = self.scopes.num_scopes_above(region_scope, span);

        // If we are emitting a `drop` statement, we need to have the cached
        // diverge cleanup pads ready in case that drop panics.
        let may_panic = self.scopes.may_panic(scope_count);
        if may_panic {
            self.diverge_cleanup();
        }

        let mut scopes = self.scopes.top_scopes(scope_count + 1).rev();
        let mut scope = scopes.next().unwrap();
        for next_scope in scopes {
            if scope.drops.is_empty() {
                scope = next_scope;
                continue;
            }
            let source_info = scope.source_info(span);
            block = match scope.cached_exits.entry((target, region_scope)) {
                Entry::Occupied(e) => {
                    self.cfg.terminate(block, source_info,
                                    TerminatorKind::Goto { target: *e.get() });
                    return;
                }
                Entry::Vacant(v) => {
                    let b = self.cfg.start_new_block();
                    self.cfg.terminate(block, source_info,
                                    TerminatorKind::Goto { target: b });
                    v.insert(b);
                    b
                }
            };

            let unwind_to = next_scope.cached_unwind.get(false).unwrap_or_else(|| {
                debug_assert!(!may_panic, "cached block not present?");
                START_BLOCK
            });

            unpack!(block = build_scope_drops(
                &mut self.cfg,
                self.is_generator,
                scope,
                block,
                unwind_to,
                self.arg_count,
                false,
            ));

            scope = next_scope;
        }

        let source_info = self.scopes.source_info(scope_count, span);
        self.cfg.terminate(block, source_info, TerminatorKind::Goto { target });
    }

    /// Creates a path that performs all required cleanup for dropping a generator.
    ///
    /// This path terminates in GeneratorDrop. Returns the start of the path.
    /// None indicates there’s no cleanup to do at this point.
    pub fn generator_drop_cleanup(&mut self) -> Option<BasicBlock> {
        // Fill in the cache for unwinds
        self.diverge_cleanup_gen(true);

        let src_info = self.scopes.source_info(self.scopes.len(), self.fn_span);
        let resume_block = self.resume_block();
        let mut scopes = self.scopes.iter_mut().peekable();
        let mut block = self.cfg.start_new_block();
        let result = block;

        while let Some(scope) = scopes.next() {
            block = if let Some(b) = scope.cached_generator_drop {
                self.cfg.terminate(block, src_info,
                                   TerminatorKind::Goto { target: b });
                return Some(result);
            } else {
                let b = self.cfg.start_new_block();
                scope.cached_generator_drop = Some(b);
                self.cfg.terminate(block, src_info,
                                   TerminatorKind::Goto { target: b });
                b
            };

            let unwind_to = scopes.peek().as_ref().map(|scope| {
                scope.cached_unwind.get(true).unwrap_or_else(|| {
                    span_bug!(src_info.span, "cached block not present?")
                })
            }).unwrap_or(resume_block);

            unpack!(block = build_scope_drops(
                &mut self.cfg,
                self.is_generator,
                scope,
                block,
                unwind_to,
                self.arg_count,
                true,
            ));
        }

        self.cfg.terminate(block, src_info, TerminatorKind::GeneratorDrop);

        Some(result)
    }

    /// Creates a new source scope, nested in the current one.
    pub fn new_source_scope(&mut self,
                            span: Span,
                            lint_level: LintLevel,
                            safety: Option<Safety>) -> SourceScope {
        let parent = self.source_scope;
        debug!("new_source_scope({:?}, {:?}, {:?}) - parent({:?})={:?}",
               span, lint_level, safety,
               parent, self.source_scope_local_data.get(parent));
        let scope = self.source_scopes.push(SourceScopeData {
            span,
            parent_scope: Some(parent),
        });
        let scope_local_data = SourceScopeLocalData {
            lint_root: if let LintLevel::Explicit(lint_root) = lint_level {
                lint_root
            } else {
                self.source_scope_local_data[parent].lint_root
            },
            safety: safety.unwrap_or_else(|| {
                self.source_scope_local_data[parent].safety
            })
        };
        self.source_scope_local_data.push(scope_local_data);
        scope
    }

    /// Given a span and the current source scope, make a SourceInfo.
    pub fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            span,
            scope: self.source_scope
        }
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
    pub fn local_scope(&self) -> Option<region::Scope> {
        match self.hir.body_owner_kind {
            hir::BodyOwnerKind::Const |
            hir::BodyOwnerKind::Static(_) =>
                // No need to free storage in this context.
                None,
            hir::BodyOwnerKind::Closure |
            hir::BodyOwnerKind::Fn =>
                Some(self.scopes.topmost()),
        }
    }

    // Schedule an abort block - this is used for some ABIs that cannot unwind
    pub fn schedule_abort(&mut self) -> BasicBlock {
        let source_info = self.scopes.source_info(self.scopes.len(), self.fn_span);
        let abortblk = self.cfg.start_new_cleanup_block();
        self.cfg.terminate(abortblk, source_info, TerminatorKind::Abort);
        self.cached_resume_block = Some(abortblk);
        abortblk
    }

    // Scheduling drops
    // ================
    pub fn schedule_drop_storage_and_value(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        local: Local,
        place_ty: Ty<'tcx>,
    ) {
        self.schedule_drop(span, region_scope, local, place_ty, DropKind::Storage);
        self.schedule_drop(span, region_scope, local, place_ty, DropKind::Value);
    }

    /// Indicates that `place` should be dropped on exit from
    /// `region_scope`.
    ///
    /// When called with `DropKind::Storage`, `place` should be a local
    /// with an index higher than the current `self.arg_count`.
    pub fn schedule_drop(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        local: Local,
        place_ty: Ty<'tcx>,
        drop_kind: DropKind,
    ) {
        let needs_drop = self.hir.needs_drop(place_ty);
        match drop_kind {
            DropKind::Value => if !needs_drop { return },
            DropKind::Storage => {
                if local.index() <= self.arg_count {
                    span_bug!(
                        span, "`schedule_drop` called with local {:?} and arg_count {}",
                        local,
                        self.arg_count,
                    )
                }
            }
        }

        for scope in self.scopes.iter_mut() {
            let this_scope = scope.region_scope == region_scope;
            // When building drops, we try to cache chains of drops in such a way so these drops
            // could be reused by the drops which would branch into the cached (already built)
            // blocks.  This, however, means that whenever we add a drop into a scope which already
            // had some blocks built (and thus, cached) for it, we must invalidate all caches which
            // might branch into the scope which had a drop just added to it. This is necessary,
            // because otherwise some other code might use the cache to branch into already built
            // chain of drops, essentially ignoring the newly added drop.
            //
            // For example consider there’s two scopes with a drop in each. These are built and
            // thus the caches are filled:
            //
            // +--------------------------------------------------------+
            // | +---------------------------------+                    |
            // | | +--------+     +-------------+  |  +---------------+ |
            // | | | return | <-+ | drop(outer) | <-+ |  drop(middle) | |
            // | | +--------+     +-------------+  |  +---------------+ |
            // | +------------|outer_scope cache|--+                    |
            // +------------------------------|middle_scope cache|------+
            //
            // Now, a new, inner-most scope is added along with a new drop into both inner-most and
            // outer-most scopes:
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
            // The cache and its invalidation for unwind branch is somewhat special. The cache is
            // per-drop, rather than per scope, which has a several different implications. Adding
            // a new drop into a scope will not invalidate cached blocks of the prior drops in the
            // scope. That is true, because none of the already existing drops will have an edge
            // into a block with the newly added drop.
            //
            // Note that this code iterates scopes from the inner-most to the outer-most,
            // invalidating caches of each scope visited. This way bare minimum of the
            // caches gets invalidated. i.e., if a new drop is added into the middle scope, the
            // cache of outer scope stays intact.
            scope.invalidate_cache(!needs_drop, self.is_generator, this_scope);
            if this_scope {
                if let DropKind::Value = drop_kind {
                    scope.needs_cleanup = true;
                }

                let region_scope_span = region_scope.span(self.hir.tcx(),
                                                          &self.hir.region_scope_tree);
                // Attribute scope exit drops to scope's closing brace.
                let scope_end = self.hir.tcx().sess.source_map().end_point(region_scope_span);

                scope.drops.push(DropData {
                    span: scope_end,
                    local,
                    kind: drop_kind,
                    cached_block: CachedBlock::default(),
                });
                return;
            }
        }
        span_bug!(span, "region scope {:?} not in scope to drop {:?}", region_scope, local);
    }

    // Other
    // =====
    /// Branch based on a boolean condition.
    ///
    /// This is a special case because the temporary for the condition needs to
    /// be dropped on both the true and the false arm.
    pub fn test_bool(
        &mut self,
        mut block: BasicBlock,
        condition: Expr<'tcx>,
        source_info: SourceInfo,
    ) -> (BasicBlock, BasicBlock) {
        let cond = unpack!(block = self.as_local_operand(block, condition));
        let true_block = self.cfg.start_new_block();
        let false_block = self.cfg.start_new_block();
        let term = TerminatorKind::if_(
            self.hir.tcx(),
            cond.clone(),
            true_block,
            false_block,
        );
        self.cfg.terminate(block, source_info, term);

        match cond {
            // Don't try to drop a constant
            Operand::Constant(_) => (),
            // If constants and statics, we don't generate StorageLive for this
            // temporary, so don't try to generate StorageDead for it either.
            _ if self.local_scope().is_none() => (),
            Operand::Copy(Place::Base(PlaceBase::Local(cond_temp)))
            | Operand::Move(Place::Base(PlaceBase::Local(cond_temp))) => {
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
                            Statement {
                                source_info,
                                kind: StatementKind::StorageDead(local)
                            },
                        );
                        self.cfg.push(
                            false_block,
                            Statement {
                                source_info,
                                kind: StatementKind::StorageDead(local)
                            },
                        );
                    }
                }

                top_scope.invalidate_cache(true, self.is_generator, true);
            }
            _ => bug!("Expected as_local_operand to produce a temporary"),
        }

        (true_block, false_block)
    }

    /// Creates a path that performs all required cleanup for unwinding.
    ///
    /// This path terminates in Resume. Returns the start of the path.
    /// See module comment for more details.
    pub fn diverge_cleanup(&mut self) -> BasicBlock {
        self.diverge_cleanup_gen(false)
    }

    fn resume_block(&mut self) -> BasicBlock {
        if let Some(target) = self.cached_resume_block {
            target
        } else {
            let resumeblk = self.cfg.start_new_cleanup_block();
            self.cfg.terminate(resumeblk,
                               SourceInfo {
                                   scope: OUTERMOST_SOURCE_SCOPE,
                                   span: self.fn_span
                               },
                               TerminatorKind::Resume);
            self.cached_resume_block = Some(resumeblk);
            resumeblk
        }
    }

    fn diverge_cleanup_gen(&mut self, generator_drop: bool) -> BasicBlock {
        // Build up the drops in **reverse** order. The end result will
        // look like:
        //
        //    scopes[n] -> scopes[n-1] -> ... -> scopes[0]
        //
        // However, we build this in **reverse order**. That is, we
        // process scopes[0], then scopes[1], etc, pointing each one at
        // the result generates from the one before. Along the way, we
        // store caches. If everything is cached, we'll just walk right
        // to left reading the cached results but never created anything.

        // Find the last cached block
        debug!("diverge_cleanup_gen(self.scopes = {:?})", self.scopes);
        let cached_cleanup = self.scopes.iter_mut().enumerate()
            .find_map(|(idx, ref scope)| {
                let cached_block = scope.cached_unwind.get(generator_drop)?;
                Some((cached_block, idx))
            });
        let (mut target, first_uncached) = cached_cleanup
            .unwrap_or_else(|| (self.resume_block(), self.scopes.len()));

        for scope in self.scopes.top_scopes(first_uncached) {
            target = build_diverge_scope(&mut self.cfg, scope.region_scope_span,
                                         scope, target, generator_drop, self.is_generator);
        }

        target
    }

    /// Utility function for *non*-scope code to build their own drops
    pub fn build_drop_and_replace(&mut self,
                                  block: BasicBlock,
                                  span: Span,
                                  location: Place<'tcx>,
                                  value: Operand<'tcx>) -> BlockAnd<()> {
        let source_info = self.source_info(span);
        let next_target = self.cfg.start_new_block();
        let diverge_target = self.diverge_cleanup();
        self.cfg.terminate(block, source_info,
                           TerminatorKind::DropAndReplace {
                               location,
                               value,
                               target: next_target,
                               unwind: Some(diverge_target),
                           });
        next_target.unit()
    }

    /// Creates an Assert terminator and return the success block.
    /// If the boolean condition operand is not the expected value,
    /// a runtime panic will be caused with the given message.
    pub fn assert(&mut self, block: BasicBlock,
                  cond: Operand<'tcx>,
                  expected: bool,
                  msg: AssertMessage<'tcx>,
                  span: Span)
                  -> BasicBlock {
        let source_info = self.source_info(span);

        let success_block = self.cfg.start_new_block();
        let cleanup = self.diverge_cleanup();

        self.cfg.terminate(block, source_info,
                           TerminatorKind::Assert {
                               cond,
                               expected,
                               msg,
                               target: success_block,
                               cleanup: Some(cleanup),
                           });

        success_block
    }

    // `match` arm scopes
    // ==================
    /// Unschedules any drops in the top scope.
    ///
    /// This is only needed for `match` arm scopes, because they have one
    /// entrance per pattern, but only one exit.
    pub(crate) fn clear_top_scope(&mut self, region_scope: region::Scope) {
        let top_scope = self.scopes.scopes.last_mut().unwrap();

        assert_eq!(top_scope.region_scope, region_scope);

        top_scope.drops.clear();
        top_scope.invalidate_cache(false, self.is_generator, true);
    }
}

/// Builds drops for pop_scope and exit_scope.
fn build_scope_drops<'tcx>(
    cfg: &mut CFG<'tcx>,
    is_generator: bool,
    scope: &Scope,
    mut block: BasicBlock,
    last_unwind_to: BasicBlock,
    arg_count: usize,
    generator_drop: bool,
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

    for drop_idx in (0..scope.drops.len()).rev() {
        let drop_data = &scope.drops[drop_idx];
        let source_info = scope.source_info(drop_data.span);
        let local = drop_data.local;
        match drop_data.kind {
            DropKind::Value => {
                let unwind_to = get_unwind_to(scope, is_generator, drop_idx, generator_drop)
                    .unwrap_or(last_unwind_to);

                let next = cfg.start_new_block();
                cfg.terminate(block, source_info, TerminatorKind::Drop {
                    location: local.into(),
                    target: next,
                    unwind: Some(unwind_to)
                });
                block = next;
            }
            DropKind::Storage => {
                // Only temps and vars need their storage dead.
                assert!(local.index() > arg_count);
                cfg.push(block, Statement {
                    source_info,
                    kind: StatementKind::StorageDead(local)
                });
            }
        }
    }
    block.unit()
}

fn get_unwind_to(
    scope: &Scope,
    is_generator: bool,
    unwind_from: usize,
    generator_drop: bool,
) -> Option<BasicBlock> {
    for drop_idx in (0..unwind_from).rev() {
        let drop_data = &scope.drops[drop_idx];
        match (is_generator, &drop_data.kind) {
            (true, DropKind::Storage) => {
                return Some(drop_data.cached_block.get(generator_drop).unwrap_or_else(|| {
                    span_bug!(drop_data.span, "cached block not present for {:?}", drop_data)
                }));
            }
            (false, DropKind::Value) => {
                return Some(drop_data.cached_block.get(generator_drop).unwrap_or_else(|| {
                    span_bug!(drop_data.span, "cached block not present for {:?}", drop_data)
                }));
            }
            _ => (),
        }
    }
    None
}

fn build_diverge_scope<'tcx>(cfg: &mut CFG<'tcx>,
                             span: Span,
                             scope: &mut Scope,
                             mut target: BasicBlock,
                             generator_drop: bool,
                             is_generator: bool)
                             -> BasicBlock
{
    // Build up the drops in **reverse** order. The end result will
    // look like:
    //
    //    [drops[n]] -...-> [drops[0]] -> [target]
    //
    // The code in this function reads from right to left. At each
    // point, we check for cached blocks representing the
    // remainder. If everything is cached, we'll just walk right to
    // left reading the cached results but never create anything.

    let source_scope = scope.source_scope;
    let source_info = |span| SourceInfo {
        span,
        scope: source_scope
    };

    // We keep track of StorageDead statements to prepend to our current block
    // and store them here, in reverse order.
    let mut storage_deads = vec![];

    let mut target_built_by_us = false;

    // Build up the drops. Here we iterate the vector in
    // *forward* order, so that we generate drops[0] first (right to
    // left in diagram above).
    debug!("build_diverge_scope({:?})", scope.drops);
    for (j, drop_data) in scope.drops.iter_mut().enumerate() {
        debug!("build_diverge_scope drop_data[{}]: {:?}", j, drop_data);
        // Only full value drops are emitted in the diverging path,
        // not StorageDead, except in the case of generators.
        //
        // Note: This may not actually be what we desire (are we
        // "freeing" stack storage as we unwind, or merely observing a
        // frozen stack)? In particular, the intent may have been to
        // match the behavior of clang, but on inspection eddyb says
        // this is not what clang does.
        match drop_data.kind {
            DropKind::Storage if is_generator => {
                storage_deads.push(Statement {
                    source_info: source_info(drop_data.span),
                    kind: StatementKind::StorageDead(drop_data.local)
                });
                if !target_built_by_us {
                    // We cannot add statements to an existing block, so we create a new
                    // block for our StorageDead statements.
                    let block = cfg.start_new_cleanup_block();
                    let source_info = SourceInfo { span: DUMMY_SP, scope: source_scope };
                    cfg.terminate(block, source_info,
                                    TerminatorKind::Goto { target: target });
                    target = block;
                    target_built_by_us = true;
                }
                *drop_data.cached_block.ref_mut(generator_drop) = Some(target);
            }
            DropKind::Storage => {}
            DropKind::Value => {
                let cached_block = drop_data.cached_block.ref_mut(generator_drop);
                target = if let Some(cached_block) = *cached_block {
                    storage_deads.clear();
                    target_built_by_us = false;
                    cached_block
                } else {
                    push_storage_deads(cfg, target, &mut storage_deads);
                    let block = cfg.start_new_cleanup_block();
                    cfg.terminate(
                        block,
                        source_info(drop_data.span),
                        TerminatorKind::Drop {
                            location: drop_data.local.into(),
                            target,
                            unwind: None
                        },
                    );
                    *cached_block = Some(block);
                    target_built_by_us = true;
                    block
                };
            }
        };
    }
    push_storage_deads(cfg, target, &mut storage_deads);
    *scope.cached_unwind.ref_mut(generator_drop) = Some(target);

    assert!(storage_deads.is_empty());
    debug!("build_diverge_scope({:?}, {:?}) = {:?}", scope, span, target);

    target
}

fn push_storage_deads(cfg: &mut CFG<'tcx>,
                      target: BasicBlock,
                      storage_deads: &mut Vec<Statement<'tcx>>) {
    if storage_deads.is_empty() { return; }
    let statements = &mut cfg.block_data_mut(target).statements;
    storage_deads.reverse();
    debug!("push_storage_deads({:?}), storage_deads={:?}, statements={:?}",
           target, storage_deads, statements);
    storage_deads.append(statements);
    mem::swap(statements, storage_deads);
    assert!(storage_deads.is_empty());
}
