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

```
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

use build::{BlockAnd, BlockAndExtension, Builder, CFG};
use rustc::middle::region::CodeExtent;
use rustc::middle::lang_items;
use rustc::middle::subst::{Substs, Subst, VecPerParamSpace};
use rustc::middle::ty::{self, Ty};
use rustc::mir::repr::*;
use syntax::codemap::{Span, DUMMY_SP};
use syntax::parse::token::intern_and_get_ident;

pub struct Scope<'tcx> {
    extent: CodeExtent,
    drops: Vec<DropData<'tcx>>,
    // A scope may only have one associated free, because:
    // 1. We require a `free` to only be scheduled in the scope of `EXPR` in `box EXPR`;
    // 2. It only makes sense to have it translated into the diverge-path.
    //
    // This kind of drop will be run *after* all the regular drops scheduled onto this scope,
    // because drops may have dependencies on the allocated memory.
    //
    // This is expected to go away once `box EXPR` becomes a sugar for placement protocol and gets
    // desugared in some earlier stage.
    free: Option<FreeData<'tcx>>,
}

struct DropData<'tcx> {
    value: Lvalue<'tcx>,
    // NB: per-drop “cache” is necessary for the build_scope_drops function below.
    /// The cached block for the cleanups-on-diverge path. This block contains code to run the
    /// current drop and all the preceding drops (i.e. those having lower index in Drop’s
    /// Scope drop array)
    cached_block: Option<BasicBlock>
}

struct FreeData<'tcx> {
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
    pub break_block: BasicBlock, // where to go on a `break
    /// Indicates the reachability of the break_block for this loop
    pub might_break: bool
}

impl<'tcx> Scope<'tcx> {
    /// Invalidate all the cached blocks in the scope.
    ///
    /// Should always be run for all inner scopes when a drop is pushed into some scope enclosing a
    /// larger extent of code.
    fn invalidate_cache(&mut self) {
        for dropdata in &mut self.drops {
            dropdata.cached_block = None;
        }
        if let Some(ref mut freedata) = self.free {
            freedata.cached_block = None;
        }
    }

    /// Returns the cached block for this scope.
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
}

impl<'a,'tcx> Builder<'a,'tcx> {
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
        where F: FnOnce(&mut Builder<'a, 'tcx>)
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
        where F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_scope(extent={:?}, block={:?})", extent, block);
        self.push_scope(extent);
        let rv = unpack!(block = f(self));
        unpack!(block = self.pop_scope(extent, block));
        debug!("in_scope: exiting extent={:?} block={:?}", extent, block);
        block.and(rv)
    }

    /// Push a scope onto the stack. You can then build code in this
    /// scope and call `pop_scope` afterwards. Note that these two
    /// calls must be paired; using `in_scope` as a convenience
    /// wrapper maybe preferable.
    pub fn push_scope(&mut self, extent: CodeExtent) {
        debug!("push_scope({:?})", extent);
        self.scopes.push(Scope {
            extent: extent.clone(),
            drops: vec![],
            free: None
        });
    }

    /// Pops a scope, which should have extent `extent`, adding any
    /// drops onto the end of `block` that are needed.  This must
    /// match 1-to-1 with `push_scope`.
    pub fn pop_scope(&mut self, extent: CodeExtent, block: BasicBlock) -> BlockAnd<()> {
        debug!("pop_scope({:?}, {:?})", extent, block);
        // We need to have `cached_block`s available for all the drops, so we call diverge_cleanup
        // to make sure all the `cached_block`s are filled in.
        self.diverge_cleanup();
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.extent, extent);
        build_scope_drops(&mut self.cfg, &scope, &self.scopes[..], block)
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
        let scope_count = 1 + self.scopes.iter().rev().position(|scope| scope.extent == extent)
                                                      .unwrap_or_else(||{
            self.hir.span_bug(span, &format!("extent {:?} does not enclose", extent))
        });

        let tmp = self.get_unit_temp();
        for (idx, ref scope) in self.scopes.iter().enumerate().rev().take(scope_count) {
            unpack!(block = build_scope_drops(&mut self.cfg,
                                              scope,
                                              &self.scopes[..idx],
                                              block));
            if let Some(ref free_data) = scope.free {
                let next = self.cfg.start_new_block();
                let free = build_free(self.hir.tcx(), tmp.clone(), free_data, next);
                self.cfg.terminate(block, free);
                block = next;
            }
        }
        self.cfg.terminate(block, Terminator::Goto { target: target });
    }

    // Finding scopes
    // ==============
    /// Finds the loop scope for a given label. This is used for
    /// resolving `break` and `continue`.
    pub fn find_loop_scope(&mut self,
                           span: Span,
                           label: Option<CodeExtent>)
                           -> &mut LoopScope {
        let Builder { ref mut loop_scopes, ref mut hir, .. } = *self;
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
        }.unwrap_or_else(|| hir.span_bug(span, "no enclosing loop scope found?"))
    }

    pub fn extent_of_innermost_scope(&self) -> CodeExtent {
        self.scopes.last().map(|scope| scope.extent).unwrap()
    }

    pub fn extent_of_outermost_scope(&self) -> CodeExtent {
        self.scopes.first().map(|scope| scope.extent).unwrap()
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
                    value: lvalue.clone(),
                    cached_block: None
                });
                return;
            } else {
                // We must invalidate all the cached_blocks leading up to the scope we’re
                // looking for, because all of the blocks in the chain will become incorrect.
                scope.invalidate_cache()
            }
        }
        self.hir.span_bug(span,
                          &format!("extent {:?} not in scope to drop {:?}", extent, lvalue));
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
        self.hir.span_bug(span,
                          &format!("extent {:?} not in scope to free {:?}", extent, value));
    }

    // Other
    // =====
    /// Creates a path that performs all required cleanup for unwinding.
    ///
    /// This path terminates in Resume. Returns the start of the path.
    /// See module comment for more details. None indicates there’s no
    /// cleanup to do at this point.
    pub fn diverge_cleanup(&mut self) -> Option<BasicBlock> {
        if self.scopes.is_empty() {
            return None;
        }
        let unit_temp = self.get_unit_temp();
        let Builder { ref mut hir, ref mut cfg, ref mut scopes, .. } = *self;
        let mut next_block = None;

        // Given an array of scopes, we generate these from the outermost scope to the innermost
        // one. Thus for array [S0, S1, S2] with corresponding cleanup blocks [B0, B1, B2], we will
        // generate B0 <- B1 <- B2 in left-to-right order. Control flow of the generated blocks
        // always ends up at a block with the Resume terminator.
        for scope in scopes.iter_mut().filter(|s| !s.drops.is_empty() || s.free.is_some()) {
            next_block = Some(build_diverge_scope(hir.tcx(),
                                                  cfg,
                                                  unit_temp.clone(),
                                                  scope,
                                                  next_block));
        }
        scopes.iter().rev().flat_map(|x| x.cached_block()).next()
    }

    /// Utility function for *non*-scope code to build their own drops
    pub fn build_drop(&mut self, block: BasicBlock, value: Lvalue<'tcx>) -> BlockAnd<()> {
        let next_target = self.cfg.start_new_block();
        let diverge_target = self.diverge_cleanup();
        self.cfg.terminate(block, Terminator::Drop {
            value: value,
            target: next_target,
            unwind: diverge_target,
        });
        next_target.unit()
    }


    // Panicking
    // =========
    // FIXME: should be moved into their own module
    pub fn panic_bounds_check(&mut self,
                             block: BasicBlock,
                             index: Operand<'tcx>,
                             len: Operand<'tcx>,
                             span: Span) {
        // fn(&(filename: &'static str, line: u32), index: usize, length: usize) -> !
        let func = self.lang_function(lang_items::PanicBoundsCheckFnLangItem);
        let args = func.ty.fn_args();
        let ref_ty = args.skip_binder()[0];
        let (region, tup_ty) = if let ty::TyRef(region, tyandmut) = ref_ty.sty {
            (region, tyandmut.ty)
        } else {
            self.hir.span_bug(span, &format!("unexpected panic_bound_check type: {:?}", func.ty));
        };
        let (tuple, tuple_ref) = (self.temp(tup_ty), self.temp(ref_ty));
        let (file, line) = self.span_to_fileline_args(span);
        let elems = vec![Operand::Constant(file), Operand::Constant(line)];
        // FIXME: We should have this as a constant, rather than a stack variable (to not pollute
        // icache with cold branch code), however to achieve that we either have to rely on rvalue
        // promotion or have some way, in MIR, to create constants.
        self.cfg.push_assign(block, DUMMY_SP, &tuple, // tuple = (file_arg, line_arg);
                             Rvalue::Aggregate(AggregateKind::Tuple, elems));
        // FIXME: is this region really correct here?
        self.cfg.push_assign(block, DUMMY_SP, &tuple_ref, // tuple_ref = &tuple;
                             Rvalue::Ref(*region, BorrowKind::Unique, tuple));
        let cleanup = self.diverge_cleanup();
        self.cfg.terminate(block, Terminator::Call {
            func: Operand::Constant(func),
            args: vec![Operand::Consume(tuple_ref), index, len],
            destination: None,
            cleanup: cleanup,
        });
    }

    /// Create diverge cleanup and branch to it from `block`.
    pub fn panic(&mut self, block: BasicBlock, msg: &'static str, span: Span) {
        // fn(&(msg: &'static str filename: &'static str, line: u32)) -> !
        let func = self.lang_function(lang_items::PanicFnLangItem);
        let args = func.ty.fn_args();
        let ref_ty = args.skip_binder()[0];
        let (region, tup_ty) = if let ty::TyRef(region, tyandmut) = ref_ty.sty {
            (region, tyandmut.ty)
        } else {
            self.hir.span_bug(span, &format!("unexpected panic type: {:?}", func.ty));
        };
        let (tuple, tuple_ref) = (self.temp(tup_ty), self.temp(ref_ty));
        let (file, line) = self.span_to_fileline_args(span);
        let message = Constant {
            span: DUMMY_SP,
            ty: self.hir.tcx().mk_static_str(),
            literal: self.hir.str_literal(intern_and_get_ident(msg))
        };
        let elems = vec![Operand::Constant(message),
                         Operand::Constant(file),
                         Operand::Constant(line)];
        // FIXME: We should have this as a constant, rather than a stack variable (to not pollute
        // icache with cold branch code), however to achieve that we either have to rely on rvalue
        // promotion or have some way, in MIR, to create constants.
        self.cfg.push_assign(block, DUMMY_SP, &tuple, // tuple = (message_arg, file_arg, line_arg);
                             Rvalue::Aggregate(AggregateKind::Tuple, elems));
        // FIXME: is this region really correct here?
        self.cfg.push_assign(block, DUMMY_SP, &tuple_ref, // tuple_ref = &tuple;
                             Rvalue::Ref(*region, BorrowKind::Unique, tuple));
        let cleanup = self.diverge_cleanup();
        self.cfg.terminate(block, Terminator::Call {
            func: Operand::Constant(func),
            args: vec![Operand::Consume(tuple_ref)],
            cleanup: cleanup,
            destination: None,
        });
    }

    fn lang_function(&mut self, lang_item: lang_items::LangItem) -> Constant<'tcx> {
        let funcdid = match self.hir.tcx().lang_items.require(lang_item) {
            Ok(d) => d,
            Err(m) => {
                self.hir.tcx().sess.fatal(&*m)
            }
        };
        Constant {
            span: DUMMY_SP,
            ty: self.hir.tcx().lookup_item_type(funcdid).ty,
            literal: Literal::Item {
                def_id: funcdid,
                kind: ItemKind::Function,
                substs: self.hir.tcx().mk_substs(Substs::empty())
            }
        }
    }

    fn span_to_fileline_args(&mut self, span: Span) -> (Constant<'tcx>, Constant<'tcx>) {
        let span_lines = self.hir.tcx().sess.codemap().lookup_char_pos(span.lo);
        (Constant {
            span: DUMMY_SP,
            ty: self.hir.tcx().mk_static_str(),
            literal: self.hir.str_literal(intern_and_get_ident(&span_lines.file.name))
        }, Constant {
            span: DUMMY_SP,
            ty: self.hir.tcx().types.u32,
            literal: self.hir.usize_literal(span_lines.line)
        })
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
        cfg.terminate(block, Terminator::Drop {
            value: drop_data.value.clone(),
            target: next,
            unwind: on_diverge
        });
        block = next;
    }
    block.unit()
}

fn build_diverge_scope<'tcx>(tcx: &ty::ctxt<'tcx>,
                             cfg: &mut CFG<'tcx>,
                             unit_temp: Lvalue<'tcx>,
                             scope: &mut Scope<'tcx>,
                             target: Option<BasicBlock>)
                             -> BasicBlock {
    debug_assert!(!scope.drops.is_empty() || scope.free.is_some());

    // First, we build the drops, iterating the drops array in reverse. We do that so that as soon
    // as we find a `cached_block`, we know that we’re finished and don’t need to do anything else.
    let mut previous = None;
    let mut last_drop_block = None;
    for drop_data in scope.drops.iter_mut().rev() {
        if let Some(cached_block) = drop_data.cached_block {
            if let Some((previous_block, previous_value)) = previous {
                cfg.terminate(previous_block, Terminator::Drop {
                    value: previous_value,
                    target: cached_block,
                    unwind: None
                });
                return last_drop_block.unwrap();
            } else {
                return cached_block;
            }
        } else {
            let block = cfg.start_new_cleanup_block();
            drop_data.cached_block = Some(block);
            if let Some((previous_block, previous_value)) = previous {
                cfg.terminate(previous_block, Terminator::Drop {
                    value: previous_value,
                    target: block,
                    unwind: None
                });
            } else {
                last_drop_block = Some(block);
            }
            previous = Some((block, drop_data.value.clone()));
        }
    }

    // Prepare the end target for this chain.
    let mut target = target.unwrap_or_else(||{
        let b = cfg.start_new_cleanup_block();
        cfg.terminate(b, Terminator::Resume);
        b
    });

    // Then, build the free branching into the prepared target.
    if let Some(ref mut free_data) = scope.free {
        target = if let Some(cached_block) = free_data.cached_block {
            cached_block
        } else {
            let into = cfg.start_new_cleanup_block();
            cfg.terminate(into, build_free(tcx, unit_temp, free_data, target));
            free_data.cached_block = Some(into);
            into
        }
    };

    if let Some((previous_block, previous_value)) = previous {
        // Finally, branch into that just-built `target` from the `previous_block`.
        cfg.terminate(previous_block, Terminator::Drop {
            value: previous_value,
            target: target,
            unwind: None
        });
        last_drop_block.unwrap()
    } else {
        // If `previous.is_none()`, there were no drops in this scope – we return the
        // target.
        target
    }
}

fn build_free<'tcx>(tcx: &ty::ctxt<'tcx>,
                    unit_temp: Lvalue<'tcx>,
                    data: &FreeData<'tcx>,
                    target: BasicBlock) -> Terminator<'tcx> {
    let free_func = tcx.lang_items.box_free_fn()
                       .expect("box_free language item is missing");
    let substs = tcx.mk_substs(Substs::new(
        VecPerParamSpace::new(vec![], vec![], vec![data.item_ty]),
        VecPerParamSpace::new(vec![], vec![], vec![])
    ));
    Terminator::Call {
        func: Operand::Constant(Constant {
            span: data.span,
            ty: tcx.lookup_item_type(free_func).ty.subst(tcx, substs),
            literal: Literal::Item {
                def_id: free_func,
                kind: ItemKind::Function,
                substs: substs
            }
        }),
        args: vec![Operand::Consume(data.value.clone())],
        destination: Some((unit_temp, target)),
        cleanup: None
    }
}
