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
mapping os from one scope to a vector of SEME regions.

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

use build::{BlockAnd, BlockAndExtension, Builder};
use rustc::middle::region::CodeExtent;
use rustc::middle::ty::Ty;
use rustc::mir::repr::*;
use syntax::codemap::Span;

pub struct Scope<'tcx> {
    extent: CodeExtent,
    drops: Vec<(DropKind, Span, Lvalue<'tcx>)>,
    cached_block: Option<BasicBlock>,
}

#[derive(Clone, Debug)]
pub struct LoopScope {
    pub extent: CodeExtent, // extent of the loop
    pub continue_block: BasicBlock, // where to go on a `loop`
    pub break_block: BasicBlock, // where to go on a `break
}

impl<'a,'tcx> Builder<'a,'tcx> {
    /// Start a loop scope, which tracks where `continue` and `break`
    /// should branch to. See module comment for more details.
    pub fn in_loop_scope<F, R>(&mut self,
                               loop_block: BasicBlock,
                               break_block: BasicBlock,
                               f: F)
                               -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>
    {
        let extent = self.extent_of_innermost_scope();
        let loop_scope = LoopScope {
            extent: extent.clone(),
            continue_block: loop_block,
            break_block: break_block,
        };
        self.loop_scopes.push(loop_scope);
        let r = f(self);
        assert!(self.loop_scopes.pop().unwrap().extent == extent);
        r
    }

    /// Convenience wrapper that pushes a scope and then executes `f`
    /// to build its contents, popping the scope afterwards.
    pub fn in_scope<F, R>(&mut self, extent: CodeExtent, mut block: BasicBlock, f: F) -> BlockAnd<R>
        where F: FnOnce(&mut Builder<'a, 'tcx>) -> BlockAnd<R>
    {
        debug!("in_scope(extent={:?}, block={:?})", extent, block);
        self.push_scope(extent, block);
        let rv = unpack!(block = f(self));
        self.pop_scope(extent, block);
        debug!("in_scope: exiting extent={:?} block={:?}", extent, block);
        block.and(rv)
    }

    /// Push a scope onto the stack. You can then build code in this
    /// scope and call `pop_scope` afterwards. Note that these two
    /// calls must be paired; using `in_scope` as a convenience
    /// wrapper maybe preferable.
    pub fn push_scope(&mut self, extent: CodeExtent, block: BasicBlock) {
        debug!("push_scope({:?}, {:?})", extent, block);

        // push scope, execute `f`, then pop scope again
        self.scopes.push(Scope {
            extent: extent.clone(),
            drops: vec![],
            cached_block: None,
        });
    }

    /// Pops a scope, which should have extent `extent`, adding any
    /// drops onto the end of `block` that are needed.  This must
    /// match 1-to-1 with `push_scope`.
    pub fn pop_scope(&mut self, extent: CodeExtent, block: BasicBlock) {
        debug!("pop_scope({:?}, {:?})", extent, block);
        let scope = self.scopes.pop().unwrap();

        assert_eq!(scope.extent, extent);

        // add in any drops needed on the fallthrough path (any other
        // exiting paths, such as those that arise from `break`, will
        // have drops already)
        for (kind, span, lvalue) in scope.drops {
            self.cfg.push_drop(block, span, kind, &lvalue);
        }
    }


    /// Finds the loop scope for a given label. This is used for
    /// resolving `break` and `continue`.
    pub fn find_loop_scope(&mut self,
                           span: Span,
                           label: Option<CodeExtent>)
                           -> LoopScope {
        let loop_scope =
            match label {
                None => {
                    // no label? return the innermost loop scope
                    self.loop_scopes.iter()
                                    .rev()
                                    .next()
                }
                Some(label) => {
                    // otherwise, find the loop-scope with the correct id
                    self.loop_scopes.iter()
                                    .rev()
                                    .filter(|loop_scope| loop_scope.extent == label)
                                    .next()
                }
            };

        match loop_scope {
            Some(loop_scope) => loop_scope.clone(),
            None => self.hir.span_bug(span, "no enclosing loop scope found?"),
        }
    }

    /// Branch out of `block` to `target`, exiting all scopes up to
    /// and including `extent`.  This will insert whatever drops are
    /// needed, as well as tracking this exit for the SEME region. See
    /// module comment for details.
    pub fn exit_scope(&mut self,
                      span: Span,
                      extent: CodeExtent,
                      block: BasicBlock,
                      target: BasicBlock) {
        let popped_scopes =
            match self.scopes.iter().rev().position(|scope| scope.extent == extent) {
                Some(p) => p + 1,
                None => self.hir.span_bug(span, &format!("extent {:?} does not enclose",
                                                              extent)),
            };

        for scope in self.scopes.iter_mut().rev().take(popped_scopes) {
            for &(kind, drop_span, ref lvalue) in &scope.drops {
                self.cfg.push_drop(block, drop_span, kind, lvalue);
            }
        }

        self.cfg.terminate(block, Terminator::Goto { target: target });
    }

    /// Creates a path that performs all required cleanup for unwinding.
    ///
    /// This path terminates in Resume. Returns the start of the path.
    /// See module comment for more details. None indicates there’s no
    /// cleanup to do at this point.
    pub fn diverge_cleanup(&mut self) -> Option<BasicBlock> {
        if self.scopes.is_empty() {
            return None;
        }

        let mut terminator = Terminator::Resume;
        // Given an array of scopes, we generate these from the outermost scope to the innermost
        // one. Thus for array [S0, S1, S2] with corresponding cleanup blocks [B0, B1, B2], we will
        // generate B0 <- B1 <- B2 in left-to-right order. The outermost scope (B0) will always
        // terminate with a Resume terminator.
        for scope in self.scopes.iter_mut().filter(|s| !s.drops.is_empty()) {
            if let Some(b) = scope.cached_block {
                terminator = Terminator::Goto { target: b };
                continue;
            } else {
                let new_block = self.cfg.start_new_block();
                self.cfg.terminate(new_block, terminator);
                terminator = Terminator::Goto { target: new_block };
                for &(kind, span, ref lvalue) in scope.drops.iter().rev() {
                    self.cfg.push_drop(new_block, span, kind, lvalue);
                }
                scope.cached_block = Some(new_block);
            }
        }
        // Return the innermost cached block, most likely the one we just generated.
        // Note that if there are no cleanups in scope we return None.
        self.scopes.iter().rev().flat_map(|b| b.cached_block).next()
    }

    /// Create diverge cleanup and branch to it from `block`.
    pub fn panic(&mut self, block: BasicBlock) {
        // FIXME: panic terminator should also have conditional cleanup?
        let cleanup = self.diverge_cleanup().unwrap_or(DIVERGE_BLOCK);
        self.cfg.terminate(block, Terminator::Panic { target: cleanup });
    }

    /// Indicates that `lvalue` should be dropped on exit from
    /// `extent`.
    pub fn schedule_drop(&mut self,
                         span: Span,
                         extent: CodeExtent,
                         kind: DropKind,
                         lvalue: &Lvalue<'tcx>,
                         lvalue_ty: Ty<'tcx>) {
        if self.hir.needs_drop(lvalue_ty) {
            for scope in self.scopes.iter_mut().rev() {
                // We must invalidate all the cached_blocks leading up to the scope we’re looking
                // for, because otherwise some/most of the blocks in the chain might become
                // incorrect (i.e. they still are pointing at old cached_block).
                scope.cached_block = None;
                if scope.extent == extent {
                    scope.drops.push((kind, span, lvalue.clone()));
                    return;
                }
            }
            self.hir.span_bug(span,
                              &format!("extent {:?} not in scope to drop {:?}", extent, lvalue));
        }
    }

    pub fn extent_of_innermost_scope(&self) -> CodeExtent {
        self.scopes.last().map(|scope| scope.extent).unwrap()
    }

    pub fn extent_of_outermost_scope(&self) -> CodeExtent {
        self.scopes.first().map(|scope| scope.extent).unwrap()
    }
}
