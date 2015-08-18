// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code related to match expresions. These are sufficiently complex
//! to warrant their own module and submodules. :) This main module
//! includes the high-level algorithm, the submodules contain the
//! details.

use build::{BlockAnd, Builder};
use repr::*;
use hair::*;

// helper functions, broken out by category:
mod simplify;
mod test;
mod util;

impl<H:Hair> Builder<H> {
    pub fn match_expr(&mut self,
                      destination: &Lvalue<H>,
                      span: H::Span,
                      mut block: BasicBlock,
                      discriminant: ExprRef<H>,
                      arms: Vec<Arm<H>>)
                      -> BlockAnd<()>
    {
        let discriminant_lvalue =
            unpack!(block = self.as_lvalue(block, discriminant));

        let arm_blocks: Vec<BasicBlock> =
            arms.iter()
                .map(|_| self.cfg.start_new_block())
                .collect();

        let arm_bodies: Vec<ExprRef<H>> =
            arms.iter()
                .map(|arm| arm.body.clone())
                .collect();

        // assemble a list of candidates: there is one candidate per
        // pattern, which means there may be more than one candidate
        // *per arm*. These candidates are kept sorted such that the
        // highest priority candidate comes last in the list. This the
        // reverse of the order in which candidates are written in the
        // source.
        let candidates: Vec<Candidate<H>> =
            arms.into_iter()
                .zip(arm_blocks.iter())
                .rev() // highest priority comes last
                .flat_map(|(arm, &arm_block)| {
                    let guard = arm.guard;
                    arm.patterns.into_iter()
                                .rev()
                                .map(move |pat| (arm_block, pat, guard.clone()))
                })
                .map(|(arm_block, pattern, guard)| {
                    Candidate {
                        match_pairs: vec![self.match_pair(discriminant_lvalue.clone(), pattern)],
                        bindings: vec![],
                        guard: guard,
                        arm_block: arm_block,
                    }
                })
                .collect();

        // this will generate code to test discriminant_lvalue and
        // branch to the appropriate arm block
        let var_extent = self.extent_of_innermost_scope().unwrap();
        self.match_candidates(span, var_extent, candidates, block);

        // all the arm blocks will rejoin here
        let end_block = self.cfg.start_new_block();

        for (arm_body, &arm_block) in arm_bodies.into_iter().zip(arm_blocks.iter()) {
            let mut arm_block = arm_block;
            unpack!(arm_block = self.into(destination, arm_block, arm_body));
            self.cfg.terminate(arm_block, Terminator::Goto { target: end_block });
        }

        end_block.unit()
    }

    pub fn expr_into_pattern(&mut self,
                             mut block: BasicBlock,
                             var_extent: H::CodeExtent,          // lifetime of vars
                             irrefutable_pat: PatternRef<H>,
                             initializer: ExprRef<H>)
                             -> BlockAnd<()>
    {
        // optimize the case of `let x = ...`
        let irrefutable_pat = self.hir.mirror(irrefutable_pat);
        match irrefutable_pat.kind {
            PatternKind::Binding { mutability,
                                   name,
                                   mode: BindingMode::ByValue,
                                   var,
                                   ty,
                                   subpattern: None } => {
                let index = self.declare_binding(var_extent, mutability, name,
                                                 var, ty, irrefutable_pat.span);
                let lvalue = Lvalue::Var(index);
                return self.into(&lvalue, block, initializer);
            }
            _ => { }
        }
        let lvalue = unpack!(block = self.as_lvalue(block, initializer));
        self.lvalue_into_pattern(block, var_extent,
                                 PatternRef::Mirror(Box::new(irrefutable_pat)), &lvalue)
    }

    pub fn lvalue_into_pattern(&mut self,
                               block: BasicBlock,
                               var_extent: H::CodeExtent,
                               irrefutable_pat: PatternRef<H>,
                               initializer: &Lvalue<H>)
                               -> BlockAnd<()>
    {
        // create a dummy candidate
        let mut candidate = Candidate::<H> {
            match_pairs: vec![self.match_pair(initializer.clone(), irrefutable_pat)],
            bindings: vec![],
            guard: None,
            arm_block: block
        };

        // Simplify the candidate. Since the pattern is irrefutable, this should
        // always convert all match-pairs into bindings.
        self.simplify_candidate(&mut candidate);

        if !candidate.match_pairs.is_empty() {
            self.hir.span_bug(
                candidate.match_pairs[0].pattern.span,
                &format!("match pairs {:?} remaining after simplifying irrefutable pattern",
                         candidate.match_pairs));
        }

        // now apply the bindings, which will also declare the variables
        self.bind_matched_candidate(block, var_extent, candidate.bindings);

        block.unit()
    }

    pub fn declare_uninitialized_variables(&mut self,
                                           var_extent: H::CodeExtent,
                                           pattern: PatternRef<H>)
    {
        let pattern = self.hir.mirror(pattern);
        match pattern.kind {
            PatternKind::Binding { mutability, name, mode: _, var, ty, subpattern } => {
                self.declare_binding(var_extent, mutability, name, var, ty, pattern.span);
                if let Some(subpattern) = subpattern {
                    self.declare_uninitialized_variables(var_extent, subpattern);
                }
            }
            PatternKind::Array { prefix, slice, suffix } |
            PatternKind::Slice { prefix, slice, suffix } => {
                for subpattern in prefix.into_iter().chain(slice).chain(suffix) {
                    self.declare_uninitialized_variables(var_extent, subpattern);
                }
            }
            PatternKind::Constant { .. } | PatternKind::Range { .. } | PatternKind::Wild => {
            }
            PatternKind::Deref { subpattern } => {
                self.declare_uninitialized_variables(var_extent, subpattern);
            }
            PatternKind::Leaf { subpatterns } |
            PatternKind::Variant { subpatterns, .. } => {
                for subpattern in subpatterns {
                    self.declare_uninitialized_variables(var_extent, subpattern.pattern);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Candidate<H:Hair> {
    // all of these must be satisfied...
    match_pairs: Vec<MatchPair<H>>,

    // ...these bindings established...
    bindings: Vec<Binding<H>>,

    // ...and the guard must be evaluated...
    guard: Option<ExprRef<H>>,

    // ...and then we branch here.
    arm_block: BasicBlock,
}

#[derive(Clone, Debug)]
struct Binding<H:Hair> {
    span: H::Span,
    source: Lvalue<H>,
    name: H::Ident,
    var_id: H::VarId,
    var_ty: H::Ty,
    mutability: Mutability,
    binding_mode: BindingMode<H>,
}

#[derive(Clone, Debug)]
struct MatchPair<H:Hair> {
    // this lvalue...
    lvalue: Lvalue<H>,

    // ... must match this pattern.
    pattern: Pattern<H>,
}

#[derive(Clone, Debug, PartialEq)]
enum TestKind<H:Hair> {
    // test the branches of enum
    Switch { adt_def: H::AdtDef },

    // test for equality
    Eq { value: Constant<H>, ty: H::Ty },

    // test whether the value falls within an inclusive range
    Range { lo: Constant<H>, hi: Constant<H>, ty: H::Ty },

    // test length of the slice is equal to len
    Len { len: usize, op: BinOp },
}

#[derive(Debug)]
struct Test<H:Hair> {
    span: H::Span,

    // the kind of test to be performed,
    kind: TestKind<H>,

    // the outcome we expect,
    outcome: usize,

    // and the match pairs that will result
    match_pairs: Vec<MatchPair<H>>
}

///////////////////////////////////////////////////////////////////////////
// Main matching algorithm

impl<H:Hair> Builder<H> {
    fn match_candidates(&mut self,
                        span: H::Span,
                        var_extent: H::CodeExtent,
                        mut candidates: Vec<Candidate<H>>,
                        mut block: BasicBlock)
    {
        debug!("matched_candidate(span={:?}, var_extent={:?}, block={:?}, candidates={:?})",
               span, var_extent, block, candidates);

        // Start by simplifying candidates. Once this process is
        // complete, all the match pairs which remain require some
        // form of test, whether it be a switch or pattern comparison.
        for candidate in &mut candidates {
            self.simplify_candidate(candidate);
        }

        // The candidates are inversely sorted by priority. Check to
        // see whether the candidates in the front of the queue (and
        // hence back of the vec) have satisfied all their match
        // pairs.
        let fully_matched =
            candidates.iter().rev().take_while(|c| c.match_pairs.is_empty()).count();
        debug!("match_candidates: {:?} candidates fully matched", fully_matched);
        for _ in 0..fully_matched {
            // If so, apply any bindings, test the guard (if any), and
            // branch to the arm.
            let candidate = candidates.pop().unwrap();
            match self.bind_and_guard_matched_candidate(block, var_extent, candidate) {
                None => { return; }
                Some(b) => { block = b; }
            }
        }

        // If there are no candidates that still need testing, we're done.
        // Since all matches are exhaustive, execution should never reach this point.
        if candidates.is_empty() {
            return self.panic(block);
        }

        // otherwise, extract the next match pair and construct tests
        let match_pair = &candidates.last().unwrap().match_pairs[0];
        let test = self.test(match_pair);
        debug!("match_candidates: test={:?} match_pair={:?}", test, match_pair);
        let target_blocks = self.perform_test(block, &match_pair.lvalue, &test);

        for (outcome, target_block) in target_blocks.into_iter().enumerate() {
            let applicable_candidates: Vec<Candidate<H>> =
                candidates.iter()
                          .filter_map(|candidate| {
                              self.candidate_under_assumption(&match_pair.lvalue,
                                                              &test.kind,
                                                              outcome,
                                                              candidate)
                          })
                          .collect();
            self.match_candidates(span, var_extent, applicable_candidates, target_block);
        }
    }

    /// Initializes each of the bindings from the candidate by
    /// moving/copying/ref'ing the source as appropriate. Tests the
    /// guard, if any, and then branches to the arm. Returns the block
    /// for the case where the guard fails.
    ///
    /// Note: we check earlier that if there is a guard, there cannot
    /// be move bindings.  This isn't really important for the
    /// self-consistency of this fn, but the reason for it should be
    /// clear: after we've done the assignments, if there were move
    /// bindings, further tests would be a use-after-move (which would
    /// in turn be detected by the borrowck code that runs on the
    /// MIR).
    fn bind_and_guard_matched_candidate(&mut self,
                                        mut block: BasicBlock,
                                        var_extent: H::CodeExtent,
                                        candidate: Candidate<H>)
                                        -> Option<BasicBlock> {
        debug!("bind_and_guard_matched_candidate(block={:?}, var_extent={:?}, candidate={:?})",
               block, var_extent, candidate);

        debug_assert!(candidate.match_pairs.is_empty());

        self.bind_matched_candidate(block, var_extent, candidate.bindings);

        if let Some(guard) = candidate.guard {
            // the block to branch to if the guard fails; if there is no
            // guard, this block is simply unreachable
            let cond = unpack!(block = self.as_operand(block, guard));
            let otherwise = self.cfg.start_new_block();
            self.cfg.terminate(block, Terminator::If { cond: cond,
                                                       targets: [candidate.arm_block, otherwise]});
            Some(otherwise)
        } else {
            self.cfg.terminate(block, Terminator::Goto { target: candidate.arm_block });
            None
        }
    }

    fn bind_matched_candidate(&mut self,
                              block: BasicBlock,
                              var_extent: H::CodeExtent,
                              bindings: Vec<Binding<H>>) {
        debug!("bind_matched_candidate(block={:?}, var_extent={:?}, bindings={:?})",
               block, var_extent, bindings);

        // Assign each of the bindings. This may trigger moves out of the candidate.
        for binding in bindings {
            // Create a variable for the `var_id` being bound. In the
            // case where there are multiple patterns for a single
            // arm, it may already exist.
            let var_index = if !self.var_indices.contains_key(&binding.var_id) {
                self.declare_binding(var_extent,
                                     binding.mutability,
                                     binding.name,
                                     binding.var_id,
                                     binding.var_ty,
                                     binding.span)
            } else {
                self.var_indices[&binding.var_id]
            };

            let rvalue = match binding.binding_mode {
                BindingMode::ByValue =>
                    Rvalue::Use(Operand::Consume(binding.source)),
                BindingMode::ByRef(region, borrow_kind) =>
                    Rvalue::Ref(region, borrow_kind, binding.source),
            };

            self.cfg.push_assign(block, binding.span, &Lvalue::Var(var_index), rvalue);
        }
    }

    fn declare_binding(&mut self,
                       var_extent: H::CodeExtent,
                       mutability: Mutability,
                       name: H::Ident,
                       var_id: H::VarId,
                       var_ty: H::Ty,
                       span: H::Span)
                       -> u32
    {
        debug!("declare_binding(var_id={:?}, name={:?}, var_ty={:?}, var_extent={:?}, span={:?})",
               var_id, name, var_ty, var_extent, span);

        let index = self.var_decls.len();
        self.var_decls.push(VarDecl::<H> {
            mutability: mutability,
            name: name,
            ty: var_ty.clone(),
        });
        let index = index as u32;
        self.schedule_drop(span, var_extent, DropKind::Deep, &Lvalue::Var(index), var_ty);
        self.var_indices.insert(var_id, index);

        debug!("declare_binding: index={:?}", index);

        index
    }
}

