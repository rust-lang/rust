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
use crate::hair::{ExprRef, LintLevel};
use rustc::middle::region;
use rustc::hir;
use rustc::mir::*;
use syntax_pos::{DUMMY_SP, Span};
use std::mem;

crate use stack::Scopes;
use stack::CachedBlock;

mod stack;

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
    /// Block to branch into when the loop or block terminates (either by being
    /// `break`-en out from, or by having its condition to become false)
    break_block: BasicBlock,
    /// The destination of the loop/block expression itself (i.e., where to put
    /// the result of a `break` expression)
    break_destination: Place<'tcx>,
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

/// The target of an expression that breaks out of a scope
#[derive(Clone, Copy, Debug)]
pub enum BreakableTarget {
    Continue(region::Scope),
    Break(region::Scope),
    Return,
}

/// A view of the information in a scope that's needed to generate a non-unwind
/// exit from that scope
#[derive(Debug)]
struct ScopeInfo<'a> {
    source_scope: SourceScope,
    drops: &'a [DropData],
    /// Cached block that will start by exiting this scope.
    cached_block: &'a mut Option<BasicBlock>,
    /// Cached block for unwind paths that starts at the next scope. This block
    /// should be branched to if any of the drops for this scope panic.
    unwind_to: BasicBlock,
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
        let region_scope = self.topmost_scope();
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
        let (drops, unwind_to, source_scope) = self.scopes.pop_scope(region_scope);
        let unwind_to = unwind_to.unwrap_or_else(|| self.resume_block());
        let cached_block = &mut Some(block);
        let scope = ScopeInfo { drops: &drops, source_scope, cached_block, unwind_to };

        unpack!(block = build_scope_drops(
            &mut self.cfg,
            self.is_generator,
            scope,
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

        let scopes = self.scopes.exit_blocks((target, region_scope)).take(scope_count);
        for scope in scopes {
            if scope.drops.is_empty() {
                continue;
            }
            let source_info = SourceInfo { scope: scope.source_scope, span };
            match *scope.cached_block {
                Some(e) => {
                    self.cfg.terminate(block, source_info, TerminatorKind::Goto { target: e });
                    return;
                }
                None => {
                    let b = self.cfg.start_new_block();
                    self.cfg.terminate(block, source_info, TerminatorKind::Goto { target: b });
                    *scope.cached_block = Some(b);
                }
            };

            unpack!(block = build_scope_drops(
                &mut self.cfg,
                self.is_generator,
                scope,
                self.arg_count,
                false,
            ));
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
        let scopes = self.scopes.generator_drop_blocks();
        let mut block = self.cfg.start_new_block();
        let result = block;

        for scope in scopes {
            if let Some(b) = *scope.cached_block {
                self.cfg.terminate(block, src_info, TerminatorKind::Goto { target: b });
                return Some(result);
            } else {
                let b = self.cfg.start_new_block();
                *scope.cached_block = Some(b);
                self.cfg.terminate(block, src_info, TerminatorKind::Goto { target: b });
            };

            unpack!(block = build_scope_drops(
                &mut self.cfg,
                self.is_generator,
                scope,
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
                Some(self.topmost_scope()),
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
    ) {
        self.schedule_drop(span, region_scope, local, DropKind::Storage);
        self.schedule_drop(span, region_scope, local, DropKind::Value);
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
        debug!("diverge_cleanup_gen(self.scopes = {:#?})", self.scopes);
        let resume_block = self.resume_block();
        let cfg = &mut self.cfg;
        let is_generator = self.is_generator;

        self.scopes.for_each_diverge_block(
            generator_drop,
            resume_block,
            |source_scope, drops, cached_unwind, mut target| {
                target = build_diverge_scope(
                    cfg,
                    source_scope,
                    drops,
                    target,
                    generator_drop,
                    is_generator,
                );
                *cached_unwind = Some(target);
                target
            }
        )
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
}

/// Builds drops for pop_scope and exit_scope.
fn build_scope_drops<'tcx>(
    cfg: &mut CFG<'tcx>,
    is_generator: bool,
    scope: ScopeInfo<'_>,
    arg_count: usize,
    generator_drop: bool,
) -> BlockAnd<()> {
    debug!("build_scope_drops({:?})", scope);

    let mut block = scope.cached_block.unwrap();

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

    let drops = scope.drops;
    for (drop_idx, drop_data) in drops.iter().enumerate().rev() {
        let source_info = SourceInfo { scope: scope.source_scope, span: drop_data.span };
        let local = drop_data.local;
        match drop_data.kind {
            DropKind::Value => {
                let unwind_to = get_unwind_to(&drops[..drop_idx], is_generator, generator_drop)
                    .unwrap_or(scope.unwind_to);

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
    drops: &[DropData],
    is_generator: bool,
    generator_drop: bool,
) -> Option<BasicBlock> {
    for drop_data in drops.iter().rev() {
        match (is_generator, &drop_data.kind) {
            (true, DropKind::Storage) | (false, DropKind::Value) => {
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
                             source_scope:  SourceScope,
                             drops: &mut [DropData],
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
    debug!("build_diverge_scope({:?})", drops);
    for (j, drop_data) in drops.iter_mut().enumerate() {
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

    assert!(storage_deads.is_empty());
    debug!("build_diverge_scope({:?}) = {:?}", drops, target);

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
