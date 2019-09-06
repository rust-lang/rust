use crate::build::scope::{BreakableScope, BreakableTarget, DropKind, DropData, ScopeInfo};
use crate::build::{BlockAnd, Builder};
use crate::hair::Expr;
use rustc::middle::region;
use rustc::mir::{BasicBlock, Local, Operand, Place, PlaceBase, SourceScope};
use rustc::mir::{SourceInfo, Statement, StatementKind, START_BLOCK, TerminatorKind};
use syntax_pos::Span;
use rustc_data_structures::fx::FxHashMap;
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
    cached_exits: FxHashMap<(BasicBlock, region::Scope), Option<BasicBlock>>,

    /// The cache for drop chain on "generator drop" exit.
    cached_generator_drop: Option<BasicBlock>,

    /// The cache for drop chain on "unwind" exit.
    cached_unwind: CachedBlock,
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

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CachedBlock {
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

impl CachedBlock {
    fn invalidate(&mut self) {
        self.generator_drop = None;
        self.unwind = None;
    }

    pub(super) fn get(&self, generator_drop: bool) -> Option<BasicBlock> {
        if generator_drop {
            self.generator_drop
        } else {
            self.unwind
        }
    }

    pub(super) fn ref_mut(&mut self, generator_drop: bool) -> &mut Option<BasicBlock> {
        if generator_drop {
            &mut self.generator_drop
        } else {
            &mut self.unwind
        }
    }
}

#[derive(Debug, Default)]
pub struct Scopes<'tcx> {
    scopes: Vec<Scope>,
    /// The current set of breakable scopes. See module comment for more details.
    pub(super) breakable_scopes: Vec<BreakableScope<'tcx>>,
}

impl<'tcx> Scopes<'tcx> {
    pub(super) fn len(&self) -> usize {
        self.scopes.len()
    }

    pub(super) fn push_scope(
        &mut self,
        region_scope: (region::Scope, SourceInfo),
        vis_scope: SourceScope,
    ) {
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

    pub(super) fn pop_scope(
        &mut self,
        region_scope: (region::Scope, SourceInfo),
    ) -> (Vec<DropData>, Option<BasicBlock>, SourceScope) {
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.region_scope, region_scope.0);
        let unwind_to = self.scopes.last()
            .and_then(|next_scope| next_scope.cached_unwind.get(false));
        (scope.drops, unwind_to, scope.source_scope)
    }

    pub(super) fn may_panic(&self, scope_count: usize) -> bool {
        let len = self.len();
        self.scopes[(len - scope_count)..].iter().any(|s| s.needs_cleanup)
    }

    /// Finds the breakable scope for a given label. This is used for
    /// resolving `return`, `break` and `continue`.
    pub(super) fn find_breakable_scope(
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
                if scope.break_destination != Place::return_place() {
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

    /// Get the number of scopes that are above the scope with the given
    /// [region::Scope] (exclusive).
    pub(super) fn num_scopes_above(&self, region_scope: region::Scope, span: Span) -> usize {
        let scope_count = self.scopes.iter().rev()
            .position(|scope| scope.region_scope == region_scope)
            .unwrap_or_else(|| {
                span_bug!(span, "region_scope {:?} does not enclose", region_scope)
            });
        let len = self.len();
        assert!(scope_count < len, "should not use `exit_scope` to pop ALL scopes");
        scope_count
    }

    /// Call the given action for each scope that doesn't already have a cached
    /// unwind block. Scopes are iterated going up the scope stack.
    pub(super) fn for_each_diverge_block<F>(
        &mut self,
        generator_drop: bool,
        resume_block: BasicBlock,
        mut action: F,
    ) -> BasicBlock
    where
        F: FnMut(SourceScope, &mut [DropData], &mut Option<BasicBlock>, BasicBlock) -> BasicBlock
    {
        let (mut target, start_at) = self.scopes.iter().enumerate().rev()
            .find_map(move |(idx, scope)| {
                let cached_block = scope.cached_unwind.get(generator_drop)?;
                Some((cached_block, idx + 1))
            }).unwrap_or((resume_block, 0));
        self.scopes[start_at..].iter_mut().for_each(|scope| {
            target = action(
                scope.source_scope,
                &mut *scope.drops,
                scope.cached_unwind.ref_mut(generator_drop),
                target,
            )
        });
        target
    }

    /// Iterate over the [ScopeInfo] for exiting to the given target.
    /// Scopes are iterated going down the scope stack.
    pub(super) fn exit_blocks(
        &mut self,
        target: (BasicBlock, region::Scope),
    ) -> impl Iterator<Item=ScopeInfo<'_>> {
        let mut scopes = self.scopes.iter_mut().rev();
        let top_scope = scopes.next().expect("Should have at least one scope");
        scopes.scan(top_scope, move |scope, next_scope| {
            let unwind_to = next_scope.cached_unwind.get(false).unwrap_or(START_BLOCK);
            let current_scope = mem::replace(scope, next_scope);
            Some(ScopeInfo {
                source_scope: current_scope.source_scope,
                drops: &*current_scope.drops,
                cached_block: current_scope.cached_exits.entry(target).or_insert(None),
                unwind_to,
            })
        })
    }

    /// Iterate over the [ScopeInfo] for dropping a suspended generator.
    /// Scopes are iterated going down the scope stack.
    pub(super) fn generator_drop_blocks(&mut self) -> impl Iterator<Item=ScopeInfo<'_>> {
        // We don't return a ScopeInfo for the outermost scope, so ensure that
        // it's empty.
        assert!(self.scopes[0].drops.is_empty());
        let mut scopes = self.scopes.iter_mut().rev();
        let top_scope = scopes.next().unwrap();

        scopes.scan(top_scope, move |scope, next_scope| {
            let unwind_to = next_scope.cached_unwind.get(true)
                .unwrap_or_else(|| bug!("cached block not present?"));
            let current_scope = mem::replace(scope, next_scope);
            Some(ScopeInfo {
                source_scope: current_scope.source_scope,
                drops: &*current_scope.drops,
                cached_block: &mut current_scope.cached_generator_drop,
                unwind_to,
            })
        })
    }

    pub(super) fn source_info(&self, index: usize, span: Span) -> SourceInfo {
        self.scopes[self.len() - index].source_info(span)
    }
}

impl<'tcx> Builder<'_, 'tcx> {
    /// Returns the topmost active scope, which is known to be alive until
    /// the next scope expression.
    crate fn topmost_scope(&self) -> region::Scope {
        self.scopes.scopes.last().expect("topmost_scope: no scopes present").region_scope
    }
    /// Indicates that `place` should be dropped on exit from
    /// `region_scope`.
    ///
    /// When called with `DropKind::Storage`, `place` should be a local
    /// with an index higher than the current `self.arg_count`.
    crate fn schedule_drop(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        local: Local,
        drop_kind: DropKind,
    ) {
        match drop_kind {
            DropKind::Value => if !self.hir.needs_drop(self.local_decls[local].ty) { return },
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

        let is_drop = match drop_kind {
            DropKind::Value => true,
            DropKind::Storage => false,
        };
        for scope in self.scopes.scopes.iter_mut().rev() {
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
            scope.invalidate_cache(!is_drop, self.is_generator, this_scope);
            if this_scope {
                if let DropKind::Value = drop_kind {
                    scope.needs_cleanup = true;
                }

                let region_scope_span = region_scope.span(
                    self.hir.tcx(),
                    &self.hir.region_scope_tree,
                );
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
        top_scope.invalidate_cache(false, self.is_generator, true);
    }

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
            Operand::Copy(Place {
                base: PlaceBase::Local(cond_temp),
                projection: box [],
            })
            | Operand::Move(Place {
                base: PlaceBase::Local(cond_temp),
                projection: box [],
            }) => {
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
}
