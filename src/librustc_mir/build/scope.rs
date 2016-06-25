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
we descend the HAIR, we push a scope on the stack, translate ite
contents, and then pop it off. Every scope is named by a
`CodeExtent`.

### SEME Regions

When pushing a new scope, we record the current point in the graph (a
basic block); this marks the entry to the scope. We then generate more
stuff in the control-flow graph. Whenever the scope is exited, either
via a `break` or `return` or just by fallthrough, that marks an exit
from the scope. Each lexical scope thus corresponds to a single-entry,
multiple-exit (SEME) region in the control-flow graph.

For now, we keep a mapping from each `CodeExtent` to its
corresponding SEME region for later reference (see caveat in next
paragraph). This is because region scopes are tied to
them. Eventually, when we shift to non-lexical lifetimes, three should
be no need to remember this mapping.

There is one additional wrinkle, actually, that I wanted to hide from
you but duty compels me to mention. In the course of translating
matches, it sometimes happen that certain code (namely guards) gets
executed multiple times. This means that the scope lexical scope may
in fact correspond to multiple, disjoint SEME regions. So in fact our
mapping is from one scope to a vector of SEME regions.

### Drops

The primary purpose for scopes is to insert drops: while translating
the contents, we also accumulate lvalues that need to be dropped upon
exit from each scope. This is done by calling `schedule_drop`. Once a
drop is scheduled, whenever we branch out we will insert drops of all
those lvalues onto the outgoing edge. Note that we don't know the full
set of scheduled drops up front, and so whenever we exit from the
scope we only drop the values scheduled thus far. For example, consider
the scope S corresponding to this loop:

```rust,ignore
loop {
    let x = ...;
    if cond { break; }
    let y = ...;
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
other enclosing scope). `exit_scope` will record thid exit point and
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

use build::{BlockAnd, BlockAndExtension, Builder, CFG, ScopeAuxiliary, ScopeId};
use rustc::middle::region::{CodeExtent, CodeExtentData};
use rustc::middle::lang_items;
use rustc::ty::subst::{Substs, Subst, VecPerParamSpace};
use rustc::ty::{Ty, TyCtxt};
use rustc::mir::repr::*;
use syntax_pos::Span;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::fnv::FnvHashMap;

pub struct Scope<'tcx> {
    /// the scope-id within the scope_auxiliary
    id: ScopeId,

    /// The visibility scope this scope was created in.
    visibility_scope: VisibilityScope,

    /// the extent of this scope within source code; also stored in
    /// `ScopeAuxiliary`, but kept here for convenience
    extent: CodeExtent,

    /// set of lvalues to drop when exiting this scope. This starts
    /// out empty but grows as variables are declared during the
    /// building process. This is a stack, so we always drop from the
    /// end of the vector (top of the stack) first.
    drops: Vec<DropData<'tcx>>,

    /// A scope may only have one associated free, because:
    ///
    /// 1. We require a `free` to only be scheduled in the scope of
    ///    `EXPR` in `box EXPR`;
    /// 2. It only makes sense to have it translated into the diverge-path.
    ///
    /// This kind of drop will be run *after* all the regular drops
    /// scheduled onto this scope, because drops may have dependencies
    /// on the allocated memory.
    ///
    /// This is expected to go away once `box EXPR` becomes a sugar
    /// for placement protocol and gets desugared in some earlier
    /// stage.
    free: Option<FreeData<'tcx>>,

    /// The cache for drop chain on “normal” exit into a particular BasicBlock.
    cached_exits: FnvHashMap<(BasicBlock, CodeExtent), BasicBlock>,
}

struct DropData<'tcx> {
    /// span where drop obligation was incurred (typically where lvalue was declared)
    span: Span,

    /// lvalue to drop
    location: Lvalue<'tcx>,

    /// The cached block for the cleanups-on-diverge path. This block
    /// contains code to run the current drop and all the preceding
    /// drops (i.e. those having lower index in Drop’s Scope drop
    /// array)
    cached_block: Option<BasicBlock>
}

struct FreeData<'tcx> {
    /// span where free obligation was incurred
    span: Span,

    /// Lvalue containing the allocated box.
    value: Lvalue<'tcx>,

    /// type of item for which the box was allocated for (i.e. the T in Box<T>).
    item_ty: Ty<'tcx>,

    /// The cached block containing code to run the free. The block will also execute all the drops
    /// in the scope.
    cached_block: Option<BasicBlock>
}

#[derive(Clone, Debug)]
pub struct LoopScope {
    /// Extent of the loop
    pub extent: CodeExtent,
    /// Where the body of the loop begins
    pub continue_block: BasicBlock,
    /// Block to branch into when the loop terminates (either by being `break`-en out from, or by
    /// having its condition to become false)
    pub break_block: BasicBlock,
    /// Indicates the reachability of the break_block for this loop
    pub might_break: bool
}

impl<'tcx> Scope<'tcx> {
    /// Invalidate all the cached blocks in the scope.
    ///
    /// Should always be run for all inner scopes when a drop is pushed into some scope enclosing a
    /// larger extent of code.
    fn invalidate_cache(&mut self) {
        self.cached_exits = FnvHashMap();
        for dropdata in &mut self.drops {
            dropdata.cached_block = None;
        }
        if let Some(ref mut freedata) = self.free {
            freedata.cached_block = None;
        }
    }

    /// Returns the cached entrypoint for diverging exit from this scope.
    ///
    /// Precondition: the caches must be fully filled (i.e. diverge_cleanup is called) in order for
    /// this method to work correctly.
    fn cached_block(&self) -> Option<BasicBlock> {
        if let Some(data) = self.drops.last() {
            Some(data.cached_block.expect("drop cache is not filled"))
        } else if let Some(ref data) = self.free {
            Some(data.cached_block.expect("free cache is not filled"))
        } else {
            None
        }
    }

    /// Given a span and this scope's visibility scope, make a SourceInfo.
    fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            span: span,
            scope: self.visibility_scope
        }
    }
}

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    // Adding and removing scopes
    // ==========================
    /// Start a loop scope, which tracks where `continue` and `break`
    /// should branch to. See module comment for more details.
    ///
    /// Returns the might_break attribute of the LoopScope used.
    pub fn in_loop_scope<F>(&mut self,
                               loop_block: BasicBlock,
                               break_block: BasicBlock,
                               f: F)
                               -> bool
        where F: FnOnce(&mut Builder<'a, 'gcx, 'tcx>)
    {
        let extent = self.extent_of_innermost_scope();
        let loop_scope = LoopScope {
            extent: extent.clone(),
            continue_block: loop_block,
            break_block: break_block,
            might_break: false
        };
        self.loop_scopes.push(loop_scope);
        f(self);
        let loop_scope = self.loop_scopes.pop().unwrap();
        assert!(loop_scope.extent == extent);
        loop_scope.might_break
    }

    /// Convenience wrapper that pushes a scope and then executes `f`
    /// to build its contents, popping the scope afterwards.
    pub fn in_scope<F, R>(&mut self, extent: CodeExtent, mut block: BasicBlock, f: F) -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'gcx, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_scope(extent={:?}, block={:?})", extent, block);
        self.push_scope(extent, block);
        let rv = unpack!(block = f(self));
        unpack!(block = self.pop_scope(extent, block));
        debug!("in_scope: exiting extent={:?} block={:?}", extent, block);
        block.and(rv)
    }

    /// Push a scope onto the stack. You can then build code in this
    /// scope and call `pop_scope` afterwards. Note that these two
    /// calls must be paired; using `in_scope` as a convenience
    /// wrapper maybe preferable.
    pub fn push_scope(&mut self, extent: CodeExtent, entry: BasicBlock) {
        debug!("push_scope({:?})", extent);
        let id = ScopeId::new(self.scope_auxiliary.len());
        let vis_scope = self.visibility_scope;
        self.scopes.push(Scope {
            id: id,
            visibility_scope: vis_scope,
            extent: extent,
            drops: vec![],
            free: None,
            cached_exits: FnvHashMap()
        });
        self.scope_auxiliary.push(ScopeAuxiliary {
            extent: extent,
            dom: self.cfg.current_location(entry),
            postdoms: vec![]
        });
    }

    /// Pops a scope, which should have extent `extent`, adding any
    /// drops onto the end of `block` that are needed.  This must
    /// match 1-to-1 with `push_scope`.
    pub fn pop_scope(&mut self,
                     extent: CodeExtent,
                     mut block: BasicBlock)
                     -> BlockAnd<()> {
        debug!("pop_scope({:?}, {:?})", extent, block);
        // We need to have `cached_block`s available for all the drops, so we call diverge_cleanup
        // to make sure all the `cached_block`s are filled in.
        self.diverge_cleanup();
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.extent, extent);
        unpack!(block = build_scope_drops(&mut self.cfg, &scope, &self.scopes, block));
        self.scope_auxiliary[scope.id]
            .postdoms
            .push(self.cfg.current_location(block));
        block.unit()
    }


    /// Branch out of `block` to `target`, exiting all scopes up to
    /// and including `extent`.  This will insert whatever drops are
    /// needed, as well as tracking this exit for the SEME region. See
    /// module comment for details.
    pub fn exit_scope(&mut self,
                      span: Span,
                      extent: CodeExtent,
                      mut block: BasicBlock,
                      target: BasicBlock) {
        debug!("exit_scope(extent={:?}, block={:?}, target={:?})", extent, block, target);
        let scope_count = 1 + self.scopes.iter().rev().position(|scope| scope.extent == extent)
                                                      .unwrap_or_else(||{
            span_bug!(span, "extent {:?} does not enclose", extent)
        });
        let len = self.scopes.len();
        assert!(scope_count < len, "should not use `exit_scope` to pop ALL scopes");
        let tmp = self.get_unit_temp();
        {
        let mut rest = &mut self.scopes[(len - scope_count)..];
        while let Some((scope, rest_)) = {rest}.split_last_mut() {
            rest = rest_;
            block = if let Some(&e) = scope.cached_exits.get(&(target, extent)) {
                self.cfg.terminate(block, scope.source_info(span),
                                   TerminatorKind::Goto { target: e });
                return;
            } else {
                let b = self.cfg.start_new_block();
                self.cfg.terminate(block, scope.source_info(span),
                                   TerminatorKind::Goto { target: b });
                scope.cached_exits.insert((target, extent), b);
                b
            };
            unpack!(block = build_scope_drops(&mut self.cfg, scope, rest, block));
            if let Some(ref free_data) = scope.free {
                let next = self.cfg.start_new_block();
                let free = build_free(self.hir.tcx(), &tmp, free_data, next);
                self.cfg.terminate(block, scope.source_info(span), free);
                block = next;
            }
            self.scope_auxiliary[scope.id]
                .postdoms
                .push(self.cfg.current_location(block));
        }
        }
        let scope = &self.scopes[len - scope_count];
        self.cfg.terminate(block, scope.source_info(span),
                           TerminatorKind::Goto { target: target });
    }

    /// Creates a new visibility scope, nested in the current one.
    pub fn new_visibility_scope(&mut self, span: Span) -> VisibilityScope {
        let parent = self.visibility_scope;
        let scope = VisibilityScope::new(self.visibility_scopes.len());
        self.visibility_scopes.push(VisibilityScopeData {
            span: span,
            parent_scope: Some(parent),
        });
        scope
    }

    // Finding scopes
    // ==============
    /// Finds the loop scope for a given label. This is used for
    /// resolving `break` and `continue`.
    pub fn find_loop_scope(&mut self,
                           span: Span,
                           label: Option<CodeExtent>)
                           -> &mut LoopScope {
        let loop_scopes = &mut self.loop_scopes;
        match label {
            None => {
                // no label? return the innermost loop scope
                loop_scopes.iter_mut().rev().next()
            }
            Some(label) => {
                // otherwise, find the loop-scope with the correct id
                loop_scopes.iter_mut()
                           .rev()
                           .filter(|loop_scope| loop_scope.extent == label)
                           .next()
            }
        }.unwrap_or_else(|| span_bug!(span, "no enclosing loop scope found?"))
    }

    /// Given a span and the current visibility scope, make a SourceInfo.
    pub fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            span: span,
            scope: self.visibility_scope
        }
    }

    pub fn extent_of_innermost_scope(&self) -> CodeExtent {
        self.scopes.last().map(|scope| scope.extent).unwrap()
    }

    /// Returns the extent of the scope which should be exited by a
    /// return.
    pub fn extent_of_return_scope(&self) -> CodeExtent {
        // The outermost scope (`scopes[0]`) will be the `CallSiteScope`.
        // We want `scopes[1]`, which is the `ParameterScope`.
        assert!(self.scopes.len() >= 2);
        assert!(match self.hir.tcx().region_maps.code_extent_data(self.scopes[1].extent) {
            CodeExtentData::ParameterScope { .. } => true,
            _ => false,
        });
        self.scopes[1].extent
    }

    // Scheduling drops
    // ================
    /// Indicates that `lvalue` should be dropped on exit from
    /// `extent`.
    pub fn schedule_drop(&mut self,
                         span: Span,
                         extent: CodeExtent,
                         lvalue: &Lvalue<'tcx>,
                         lvalue_ty: Ty<'tcx>) {
        if !self.hir.needs_drop(lvalue_ty) {
            return
        }
        for scope in self.scopes.iter_mut().rev() {
            if scope.extent == extent {
                // No need to invalidate any caches here. The just-scheduled drop will branch into
                // the drop that comes before it in the vector.
                scope.drops.push(DropData {
                    span: span,
                    location: lvalue.clone(),
                    cached_block: None
                });
                return;
            } else {
                // We must invalidate all the cached_blocks leading up to the scope we’re
                // looking for, because all of the blocks in the chain will become incorrect.
                scope.invalidate_cache()
            }
        }
        span_bug!(span, "extent {:?} not in scope to drop {:?}", extent, lvalue);
    }

    /// Schedule dropping of a not-yet-fully-initialised box.
    ///
    /// This cleanup will only be translated into unwind branch.
    /// The extent should be for the `EXPR` inside `box EXPR`.
    /// There may only be one “free” scheduled in any given scope.
    pub fn schedule_box_free(&mut self,
                             span: Span,
                             extent: CodeExtent,
                             value: &Lvalue<'tcx>,
                             item_ty: Ty<'tcx>) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.extent == extent {
                assert!(scope.free.is_none(), "scope already has a scheduled free!");
                // We also must invalidate the caches in the scope for which the free is scheduled
                // because the drops must branch into the free we schedule here.
                scope.invalidate_cache();
                scope.free = Some(FreeData {
                    span: span,
                    value: value.clone(),
                    item_ty: item_ty,
                    cached_block: None
                });
                return;
            } else {
                // We must invalidate all the cached_blocks leading up to the scope we’re looking
                // for, because otherwise some/most of the blocks in the chain will become
                // incorrect.
                scope.invalidate_cache();
            }
        }
        span_bug!(span, "extent {:?} not in scope to free {:?}", extent, value);
    }

    // Other
    // =====
    /// Creates a path that performs all required cleanup for unwinding.
    ///
    /// This path terminates in Resume. Returns the start of the path.
    /// See module comment for more details. None indicates there’s no
    /// cleanup to do at this point.
    pub fn diverge_cleanup(&mut self) -> Option<BasicBlock> {
        if self.scopes.iter().all(|scope| scope.drops.is_empty() && scope.free.is_none()) {
            return None;
        }
        assert!(!self.scopes.is_empty()); // or `all` above would be true

        let unit_temp = self.get_unit_temp();
        let Builder { ref mut hir, ref mut cfg, ref mut scopes,
                      ref mut cached_resume_block, .. } = *self;

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

        // To start, create the resume terminator.
        let mut target = if let Some(target) = *cached_resume_block {
            target
        } else {
            let resumeblk = cfg.start_new_cleanup_block();
            cfg.terminate(resumeblk,
                          scopes[0].source_info(self.fn_span),
                          TerminatorKind::Resume);
            *cached_resume_block = Some(resumeblk);
            resumeblk
        };

        for scope in scopes.iter_mut().filter(|s| !s.drops.is_empty() || s.free.is_some()) {
            target = build_diverge_scope(hir.tcx(), cfg, &unit_temp, scope, target);
        }
        Some(target)
    }

    /// Utility function for *non*-scope code to build their own drops
    pub fn build_drop(&mut self,
                      block: BasicBlock,
                      span: Span,
                      location: Lvalue<'tcx>,
                      ty: Ty<'tcx>) -> BlockAnd<()> {
        if !self.hir.needs_drop(ty) {
            return block.unit();
        }
        let source_info = self.source_info(span);
        let next_target = self.cfg.start_new_block();
        let diverge_target = self.diverge_cleanup();
        self.cfg.terminate(block, source_info,
                           TerminatorKind::Drop {
                               location: location,
                               target: next_target,
                               unwind: diverge_target,
                           });
        next_target.unit()
    }

    /// Utility function for *non*-scope code to build their own drops
    pub fn build_drop_and_replace(&mut self,
                                  block: BasicBlock,
                                  span: Span,
                                  location: Lvalue<'tcx>,
                                  value: Operand<'tcx>) -> BlockAnd<()> {
        let source_info = self.source_info(span);
        let next_target = self.cfg.start_new_block();
        let diverge_target = self.diverge_cleanup();
        self.cfg.terminate(block, source_info,
                           TerminatorKind::DropAndReplace {
                               location: location,
                               value: value,
                               target: next_target,
                               unwind: diverge_target,
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
                               cond: cond,
                               expected: expected,
                               msg: msg,
                               target: success_block,
                               cleanup: cleanup
                           });

        success_block
    }
}

/// Builds drops for pop_scope and exit_scope.
fn build_scope_drops<'tcx>(cfg: &mut CFG<'tcx>,
                           scope: &Scope<'tcx>,
                           earlier_scopes: &[Scope<'tcx>],
                           mut block: BasicBlock)
                           -> BlockAnd<()> {
    let mut iter = scope.drops.iter().rev().peekable();
    while let Some(drop_data) = iter.next() {
        // Try to find the next block with its cached block for us to diverge into in case the
        // drop panics.
        let on_diverge = iter.peek().iter().flat_map(|dd| dd.cached_block.into_iter()).next();
        // If there’s no `cached_block`s within current scope, we must look for one in the
        // enclosing scope.
        let on_diverge = on_diverge.or_else(||{
            earlier_scopes.iter().rev().flat_map(|s| s.cached_block()).next()
        });
        let next = cfg.start_new_block();
        cfg.terminate(block, scope.source_info(drop_data.span), TerminatorKind::Drop {
            location: drop_data.location.clone(),
            target: next,
            unwind: on_diverge
        });
        block = next;
    }
    block.unit()
}

fn build_diverge_scope<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                       cfg: &mut CFG<'tcx>,
                                       unit_temp: &Lvalue<'tcx>,
                                       scope: &mut Scope<'tcx>,
                                       mut target: BasicBlock)
                                       -> BasicBlock
{
    // Build up the drops in **reverse** order. The end result will
    // look like:
    //
    //    [drops[n]] -...-> [drops[0]] -> [Free] -> [target]
    //    |                                    |
    //    +------------------------------------+
    //     code for scope
    //
    // The code in this function reads from right to left. At each
    // point, we check for cached blocks representing the
    // remainder. If everything is cached, we'll just walk right to
    // left reading the cached results but never created anything.

    let visibility_scope = scope.visibility_scope;
    let source_info = |span| SourceInfo {
        span: span,
        scope: visibility_scope
    };

    // Next, build up any free.
    if let Some(ref mut free_data) = scope.free {
        target = if let Some(cached_block) = free_data.cached_block {
            cached_block
        } else {
            let into = cfg.start_new_cleanup_block();
            cfg.terminate(into, source_info(free_data.span),
                          build_free(tcx, unit_temp, free_data, target));
            free_data.cached_block = Some(into);
            into
        };
    }

    // Next, build up the drops. Here we iterate the vector in
    // *forward* order, so that we generate drops[0] first (right to
    // left in diagram above).
    for drop_data in &mut scope.drops {
        target = if let Some(cached_block) = drop_data.cached_block {
            cached_block
        } else {
            let block = cfg.start_new_cleanup_block();
            cfg.terminate(block, source_info(drop_data.span),
                          TerminatorKind::Drop {
                              location: drop_data.location.clone(),
                              target: target,
                              unwind: None
                          });
            drop_data.cached_block = Some(block);
            block
        };
    }

    target
}

fn build_free<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                              unit_temp: &Lvalue<'tcx>,
                              data: &FreeData<'tcx>,
                              target: BasicBlock)
                              -> TerminatorKind<'tcx> {
    let free_func = tcx.lang_items.require(lang_items::BoxFreeFnLangItem)
                       .unwrap_or_else(|e| tcx.sess.fatal(&e));
    let substs = tcx.mk_substs(Substs::new(
        VecPerParamSpace::new(vec![], vec![], vec![data.item_ty]),
        VecPerParamSpace::new(vec![], vec![], vec![])
    ));
    TerminatorKind::Call {
        func: Operand::Constant(Constant {
            span: data.span,
            ty: tcx.lookup_item_type(free_func).ty.subst(tcx, substs),
            literal: Literal::Item {
                def_id: free_func,
                substs: substs
            }
        }),
        args: vec![Operand::Consume(data.value.clone())],
        destination: Some((unit_temp.clone(), target)),
        cleanup: None
    }
}
