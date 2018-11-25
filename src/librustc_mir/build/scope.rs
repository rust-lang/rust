// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

There is one additional wrinkle, actually, that I wanted to hide from
you but duty compels me to mention. In the course of building
matches, it sometimes happen that certain code (namely guards) gets
executed multiple times. This means that the scope lexical scope may
in fact correspond to multiple, disjoint SEME regions. So in fact our
mapping is from one scope to a vector of SEME regions.

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

use build::{BlockAnd, BlockAndExtension, Builder, CFG};
use hair::LintLevel;
use rustc::middle::region;
use rustc::ty::Ty;
use rustc::hir;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::mir::*;
use syntax_pos::{Span};
use rustc_data_structures::fx::FxHashMap;
use std::collections::hash_map::Entry;

#[derive(Debug)]
pub struct Scope<'tcx> {
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
    needs_cleanup: bool,

    /// set of places to drop when exiting this scope. This starts
    /// out empty but grows as variables are declared during the
    /// building process. This is a stack, so we always drop from the
    /// end of the vector (top of the stack) first.
    drops: Vec<DropData<'tcx>>,

    /// The cache for drop chain on “normal” exit into a particular BasicBlock.
    cached_exits: FxHashMap<(BasicBlock, region::Scope), BasicBlock>,

    /// The cache for drop chain on "generator drop" exit.
    cached_generator_drop: Option<BasicBlock>,

    /// The cache for drop chain on "unwind" exit.
    cached_unwind: CachedBlock,
}

#[derive(Debug)]
struct DropData<'tcx> {
    /// span where drop obligation was incurred (typically where place was declared)
    span: Span,

    /// place to drop
    location: Place<'tcx>,

    /// Whether this is a value Drop or a StorageDead.
    kind: DropKind,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CachedBlock {
    /// The cached block for the cleanups-on-diverge path. This block
    /// contains code to run the current drop and all the preceding
    /// drops (i.e. those having lower index in Drop’s Scope drop
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
    Value {
        cached_block: CachedBlock,
    },
    Storage
}

#[derive(Clone, Debug)]
pub struct BreakableScope<'tcx> {
    /// Region scope of the loop
    pub region_scope: region::Scope,
    /// Where the body of the loop begins. `None` if block
    pub continue_block: Option<BasicBlock>,
    /// Block to branch into when the loop or block terminates (either by being `break`-en out
    /// from, or by having its condition to become false)
    pub break_block: BasicBlock,
    /// The destination of the loop/block expression itself (i.e. where to put the result of a
    /// `break` expression)
    pub break_destination: Place<'tcx>,
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

impl DropKind {
    fn may_panic(&self) -> bool {
        match *self {
            DropKind::Value { .. } => true,
            DropKind::Storage => false
        }
    }
}

impl<'tcx> Scope<'tcx> {
    /// Invalidate all the cached blocks in the scope.
    ///
    /// Should always be run for all inner scopes when a drop is pushed into some scope enclosing a
    /// larger extent of code.
    ///
    /// `storage_only` controls whether to invalidate only drop paths that run `StorageDead`.
    /// `this_scope_only` controls whether to invalidate only drop paths that refer to the current
    /// top-of-scope (as opposed to dependent scopes).
    fn invalidate_cache(&mut self, storage_only: bool, this_scope_only: bool) {
        // FIXME: maybe do shared caching of `cached_exits` etc. to handle functions
        // with lots of `try!`?

        // cached exits drop storage and refer to the top-of-scope
        self.cached_exits.clear();

        if !storage_only {
            // the current generator drop and unwind ignore
            // storage but refer to top-of-scope
            self.cached_generator_drop = None;
            self.cached_unwind.invalidate();
        }

        if !storage_only && !this_scope_only {
            for drop_data in &mut self.drops {
                if let DropKind::Value { ref mut cached_block } = drop_data.kind {
                    cached_block.invalidate();
                }
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

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    // Adding and removing scopes
    // ==========================
    /// Start a breakable scope, which tracks where `continue` and `break`
    /// should branch to. See module comment for more details.
    ///
    /// Returns the might_break attribute of the BreakableScope used.
    pub fn in_breakable_scope<F, R>(&mut self,
                                    loop_block: Option<BasicBlock>,
                                    break_block: BasicBlock,
                                    break_destination: Place<'tcx>,
                                    f: F) -> R
        where F: FnOnce(&mut Builder<'a, 'gcx, 'tcx>) -> R
    {
        let region_scope = self.topmost_scope();
        let scope = BreakableScope {
            region_scope,
            continue_block: loop_block,
            break_block,
            break_destination,
        };
        self.breakable_scopes.push(scope);
        let res = f(self);
        let breakable_scope = self.breakable_scopes.pop().unwrap();
        assert!(breakable_scope.region_scope == region_scope);
        res
    }

    pub fn in_opt_scope<F, R>(&mut self,
                              opt_scope: Option<(region::Scope, SourceInfo)>,
                              mut block: BasicBlock,
                              f: F)
                              -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'gcx, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_opt_scope(opt_scope={:?}, block={:?})", opt_scope, block);
        if let Some(region_scope) = opt_scope { self.push_scope(region_scope); }
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
                          mut block: BasicBlock,
                          f: F)
                          -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'gcx, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_scope(region_scope={:?}, block={:?})", region_scope, block);
        let source_scope = self.source_scope;
        let tcx = self.hir.tcx();
        if let LintLevel::Explicit(node_id) = lint_level {
            let same_lint_scopes = tcx.dep_graph.with_ignore(|| {
                let sets = tcx.lint_levels(LOCAL_CRATE);
                let parent_hir_id =
                    tcx.hir.definitions().node_to_hir_id(
                        self.source_scope_local_data[source_scope].lint_root
                    );
                let current_hir_id =
                    tcx.hir.definitions().node_to_hir_id(node_id);
                sets.lint_level_set(parent_hir_id) ==
                    sets.lint_level_set(current_hir_id)
            });

            if !same_lint_scopes {
                self.source_scope =
                    self.new_source_scope(region_scope.1.span, lint_level,
                                          None);
            }
        }
        self.push_scope(region_scope);
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
        debug!("push_scope({:?})", region_scope);
        let vis_scope = self.source_scope;
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
        let may_panic =
            self.scopes.last().unwrap().drops.iter().any(|s| s.kind.may_panic());
        if may_panic {
            self.diverge_cleanup();
        }
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.region_scope, region_scope.0);

        let unwind_to = self.scopes.last().and_then(|next_scope| {
            next_scope.cached_unwind.get(false)
        }).unwrap_or_else(|| self.resume_block());

        unpack!(block = build_scope_drops(
            &mut self.cfg,
            &scope,
            block,
            unwind_to,
            self.arg_count,
            false,
        ));

        block.unit()
    }


    /// Branch out of `block` to `target`, exiting all scopes up to
    /// and including `region_scope`.  This will insert whatever drops are
    /// needed. See module comment for details.
    pub fn exit_scope(&mut self,
                      span: Span,
                      region_scope: (region::Scope, SourceInfo),
                      mut block: BasicBlock,
                      target: BasicBlock) {
        debug!("exit_scope(region_scope={:?}, block={:?}, target={:?})",
               region_scope, block, target);
        let scope_count = 1 + self.scopes.iter().rev()
            .position(|scope| scope.region_scope == region_scope.0)
            .unwrap_or_else(|| {
                span_bug!(span, "region_scope {:?} does not enclose", region_scope)
            });
        let len = self.scopes.len();
        assert!(scope_count < len, "should not use `exit_scope` to pop ALL scopes");

        // If we are emitting a `drop` statement, we need to have the cached
        // diverge cleanup pads ready in case that drop panics.
        let may_panic = self.scopes[(len - scope_count)..].iter().any(|s| s.needs_cleanup);
        if may_panic {
            self.diverge_cleanup();
        }

        let mut scopes = self.scopes[(len - scope_count - 1)..].iter_mut().rev();
        let mut scope = scopes.next().unwrap();
        for next_scope in scopes {
            if scope.drops.is_empty() {
                scope = next_scope;
                continue;
            }
            let source_info = scope.source_info(span);
            block = match scope.cached_exits.entry((target, region_scope.0)) {
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
                scope,
                block,
                unwind_to,
                self.arg_count,
                false,
            ));

            scope = next_scope;
        }

        let scope = &self.scopes[len - scope_count];
        self.cfg.terminate(block, scope.source_info(span),
                           TerminatorKind::Goto { target });
    }

    /// Creates a path that performs all required cleanup for dropping a generator.
    ///
    /// This path terminates in GeneratorDrop. Returns the start of the path.
    /// None indicates there’s no cleanup to do at this point.
    pub fn generator_drop_cleanup(&mut self) -> Option<BasicBlock> {
        if !self.scopes.iter().any(|scope| scope.needs_cleanup) {
            return None;
        }

        // Fill in the cache for unwinds
        self.diverge_cleanup_gen(true);

        let src_info = self.scopes[0].source_info(self.fn_span);
        let resume_block = self.resume_block();
        let mut scopes = self.scopes.iter_mut().rev().peekable();
        let mut block = self.cfg.start_new_block();
        let result = block;

        while let Some(scope) = scopes.next() {
            if !scope.needs_cleanup {
                continue;
            }

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

    // Finding scopes
    // ==============
    /// Finds the breakable scope for a given label. This is used for
    /// resolving `break` and `continue`.
    pub fn find_breakable_scope(&self,
                                span: Span,
                                label: region::Scope)
                                -> &BreakableScope<'tcx> {
        // find the loop-scope with the correct id
        self.breakable_scopes.iter()
            .rev()
            .filter(|breakable_scope| breakable_scope.region_scope == label)
            .next()
            .unwrap_or_else(|| span_bug!(span, "no enclosing breakable scope found"))
    }

    /// Given a span and the current source scope, make a SourceInfo.
    pub fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            span,
            scope: self.source_scope
        }
    }

    /// Returns the `region::Scope` of the scope which should be exited by a
    /// return.
    pub fn region_scope_of_return_scope(&self) -> region::Scope {
        // The outermost scope (`scopes[0]`) will be the `CallSiteScope`.
        // We want `scopes[1]`, which is the `ParameterScope`.
        assert!(self.scopes.len() >= 2);
        assert!(match self.scopes[1].region_scope.data {
            region::ScopeData::Arguments => true,
            _ => false,
        });
        self.scopes[1].region_scope
    }

    /// Returns the topmost active scope, which is known to be alive until
    /// the next scope expression.
    pub fn topmost_scope(&self) -> region::Scope {
        self.scopes.last().expect("topmost_scope: no scopes present").region_scope
    }

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
            hir::BodyOwnerKind::Fn =>
                Some(self.topmost_scope()),
        }
    }

    // Schedule an abort block - this is used for some ABIs that cannot unwind
    pub fn schedule_abort(&mut self) -> BasicBlock {
        self.scopes[0].needs_cleanup = true;
        let abortblk = self.cfg.start_new_cleanup_block();
        let source_info = self.scopes[0].source_info(self.fn_span);
        self.cfg.terminate(abortblk, source_info, TerminatorKind::Abort);
        self.cached_resume_block = Some(abortblk);
        abortblk
    }

    pub fn schedule_drop_storage_and_value(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        place: &Place<'tcx>,
        place_ty: Ty<'tcx>,
    ) {
        self.schedule_drop(
            span, region_scope, place, place_ty,
            DropKind::Storage,
        );
        self.schedule_drop(
            span, region_scope, place, place_ty,
            DropKind::Value {
                cached_block: CachedBlock::default(),
            },
        );
    }

    // Scheduling drops
    // ================
    /// Indicates that `place` should be dropped on exit from
    /// `region_scope`.
    ///
    /// When called with `DropKind::Storage`, `place` should be a local
    /// with an index higher than the current `self.arg_count`.
    pub fn schedule_drop(
        &mut self,
        span: Span,
        region_scope: region::Scope,
        place: &Place<'tcx>,
        place_ty: Ty<'tcx>,
        drop_kind: DropKind,
    ) {
        let needs_drop = self.hir.needs_drop(place_ty);
        match drop_kind {
            DropKind::Value { .. } => if !needs_drop { return },
            DropKind::Storage => {
                match *place {
                    Place::Local(index) => if index.index() <= self.arg_count {
                        span_bug!(
                            span, "`schedule_drop` called with index {} and arg_count {}",
                            index.index(),
                            self.arg_count,
                        )
                    },
                    _ => span_bug!(
                        span, "`schedule_drop` called with non-`Local` place {:?}", place
                    ),
                }
            }
        }

        for scope in self.scopes.iter_mut().rev() {
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
            // caches gets invalidated. i.e. if a new drop is added into the middle scope, the
            // cache of outer scpoe stays intact.
            scope.invalidate_cache(!needs_drop, this_scope);
            if this_scope {
                if let DropKind::Value { .. } = drop_kind {
                    scope.needs_cleanup = true;
                }

                let region_scope_span = region_scope.span(self.hir.tcx(),
                                                          &self.hir.region_scope_tree);
                // Attribute scope exit drops to scope's closing brace.
                let scope_end = self.hir.tcx().sess.source_map().end_point(region_scope_span);

                scope.drops.push(DropData {
                    span: scope_end,
                    location: place.clone(),
                    kind: drop_kind
                });
                return;
            }
        }
        span_bug!(span, "region scope {:?} not in scope to drop {:?}", region_scope, place);
    }

    // Other
    // =====
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
        let (mut target, first_uncached) = if let Some(cached_index) = self.scopes.iter()
            .rposition(|scope| scope.cached_unwind.get(generator_drop).is_some()) {
            (self.scopes[cached_index].cached_unwind.get(generator_drop).unwrap(), cached_index + 1)
        } else {
            (self.resume_block(), 0)
        };

        for scope in self.scopes[first_uncached..].iter_mut() {
            target = build_diverge_scope(&mut self.cfg, scope.region_scope_span,
                                         scope, target, generator_drop);
        }

        target
    }

    /// Utility function for *non*-scope code to build their own drops
    pub fn build_drop(&mut self,
                      block: BasicBlock,
                      span: Span,
                      location: Place<'tcx>,
                      ty: Ty<'tcx>) -> BlockAnd<()> {
        if !self.hir.needs_drop(ty) {
            return block.unit();
        }
        let source_info = self.source_info(span);
        let next_target = self.cfg.start_new_block();
        let diverge_target = self.diverge_cleanup();
        self.cfg.terminate(block, source_info,
                           TerminatorKind::Drop {
                               location,
                               target: next_target,
                               unwind: Some(diverge_target),
                           });
        next_target.unit()
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

    /// Create an Assert terminator and return the success block.
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
}

/// Builds drops for pop_scope and exit_scope.
fn build_scope_drops<'tcx>(
    cfg: &mut CFG<'tcx>,
    scope: &Scope<'tcx>,
    mut block: BasicBlock,
    last_unwind_to: BasicBlock,
    arg_count: usize,
    generator_drop: bool,
) -> BlockAnd<()> {
    debug!("build_scope_drops({:?} -> {:?}", block, scope);

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
    // another set of arrows). The drops for the unwind path should have already
    // been generated by `diverge_cleanup_gen`.
    //
    // The code in this function reads from right to left.
    // Storage dead drops have to be done left to right (since we can only push
    // to the end of a Vec). So, we find the next drop and then call
    // push_storage_deads which will iterate backwards through them so that
    // they are added in the correct order.

    let mut unwind_blocks = scope.drops.iter().rev().filter_map(|drop_data| {
        if let DropKind::Value { cached_block } = drop_data.kind {
            Some(cached_block.get(generator_drop).unwrap_or_else(|| {
                span_bug!(drop_data.span, "cached block not present?")
            }))
        } else {
            None
        }
    });

    // When we unwind from a drop, we start cleaning up from the next one, so
    // we don't need this block.
    unwind_blocks.next();

    for drop_data in scope.drops.iter().rev() {
        let source_info = scope.source_info(drop_data.span);
        match drop_data.kind {
            DropKind::Value { .. } => {
                let unwind_to = unwind_blocks.next().unwrap_or(last_unwind_to);

                let next = cfg.start_new_block();
                cfg.terminate(block, source_info, TerminatorKind::Drop {
                    location: drop_data.location.clone(),
                    target: next,
                    unwind: Some(unwind_to)
                });
                block = next;
            }
            DropKind::Storage => {
                // We do not need to emit StorageDead for generator drops
                if generator_drop {
                    continue
                }

                // Drop the storage for both value and storage drops.
                // Only temps and vars need their storage dead.
                match drop_data.location {
                    Place::Local(index) if index.index() > arg_count => {
                        cfg.push(block, Statement {
                            source_info,
                            kind: StatementKind::StorageDead(index)
                        });
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    block.unit()
}

fn build_diverge_scope<'tcx>(cfg: &mut CFG<'tcx>,
                             span: Span,
                             scope: &mut Scope<'tcx>,
                             mut target: BasicBlock,
                             generator_drop: bool)
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

    // Next, build up the drops. Here we iterate the vector in
    // *forward* order, so that we generate drops[0] first (right to
    // left in diagram above).
    for (j, drop_data) in scope.drops.iter_mut().enumerate() {
        debug!("build_diverge_scope drop_data[{}]: {:?}", j, drop_data);
        // Only full value drops are emitted in the diverging path,
        // not StorageDead.
        //
        // Note: This may not actually be what we desire (are we
        // "freeing" stack storage as we unwind, or merely observing a
        // frozen stack)? In particular, the intent may have been to
        // match the behavior of clang, but on inspection eddyb says
        // this is not what clang does.
        let cached_block = match drop_data.kind {
            DropKind::Value { ref mut cached_block } => cached_block.ref_mut(generator_drop),
            DropKind::Storage => continue
        };
        target = if let Some(cached_block) = *cached_block {
            cached_block
        } else {
            let block = cfg.start_new_cleanup_block();
            cfg.terminate(block, source_info(drop_data.span),
                          TerminatorKind::Drop {
                              location: drop_data.location.clone(),
                              target,
                              unwind: None
                          });
            *cached_block = Some(block);
            block
        };
    }

    *scope.cached_unwind.ref_mut(generator_drop) = Some(target);

    debug!("build_diverge_scope({:?}, {:?}) = {:?}", scope, span, target);

    target
}
