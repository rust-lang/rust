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

use build::{BlockAnd, BlockAndExtension, Builder};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::bitvec::BitVector;
use rustc::middle::const_val::ConstVal;
use rustc::ty::{AdtDef, Ty};
use rustc::mir::*;
use hair::*;
use syntax::ast::{Name, NodeId};
use syntax_pos::Span;

// helper functions, broken out by category:
mod simplify;
mod test;
mod util;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    pub fn match_expr(&mut self,
                      destination: &Lvalue<'tcx>,
                      span: Span,
                      mut block: BasicBlock,
                      discriminant: ExprRef<'tcx>,
                      arms: Vec<Arm<'tcx>>)
                      -> BlockAnd<()> {
        let discriminant_lvalue = unpack!(block = self.as_lvalue(block, discriminant));

        let mut arm_blocks = ArmBlocks {
            blocks: arms.iter()
                        .map(|_| self.cfg.start_new_block())
                        .collect(),
        };

        // Get the arm bodies and their scopes, while declaring bindings.
        let arm_bodies: Vec<_> = arms.iter().map(|arm| {
            let body = self.hir.mirror(arm.body.clone());
            let scope = self.declare_bindings(None, body.span, &arm.patterns[0]);
            (body, scope.unwrap_or(self.visibility_scope))
        }).collect();

        // assemble a list of candidates: there is one candidate per
        // pattern, which means there may be more than one candidate
        // *per arm*. These candidates are kept sorted such that the
        // highest priority candidate comes first in the list.
        // (i.e. same order as in source)
        let candidates: Vec<_> =
            arms.iter()
                .enumerate()
                .flat_map(|(arm_index, arm)| {
                    arm.patterns.iter()
                                .map(move |pat| (arm_index, pat, arm.guard.clone()))
                })
                .map(|(arm_index, pattern, guard)| {
                    Candidate {
                        span: pattern.span,
                        match_pairs: vec![MatchPair::new(discriminant_lvalue.clone(), pattern)],
                        bindings: vec![],
                        guard: guard,
                        arm_index: arm_index,
                    }
                })
                .collect();

        // this will generate code to test discriminant_lvalue and
        // branch to the appropriate arm block
        let otherwise = self.match_candidates(span, &mut arm_blocks, candidates, block);

        if !otherwise.is_empty() {
            // All matches are exhaustive. However, because some matches
            // only have exponentially-large exhaustive decision trees, we
            // sometimes generate an inexhaustive decision tree.
            //
            // In that case, the inexhaustive tips of the decision tree
            // can't be reached - terminate them with an `unreachable`.
            let source_info = self.source_info(span);

            let mut otherwise = otherwise;
            otherwise.sort();
            otherwise.dedup(); // variant switches can introduce duplicate target blocks
            for block in otherwise {
                self.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
            }
        }

        // all the arm blocks will rejoin here
        let end_block = self.cfg.start_new_block();

        let outer_source_info = self.source_info(span);
        for (arm_index, (body, visibility_scope)) in arm_bodies.into_iter().enumerate() {
            let mut arm_block = arm_blocks.blocks[arm_index];
            // Re-enter the visibility scope we created the bindings in.
            self.visibility_scope = visibility_scope;
            unpack!(arm_block = self.into(destination, arm_block, body));
            self.cfg.terminate(arm_block, outer_source_info,
                               TerminatorKind::Goto { target: end_block });
        }
        self.visibility_scope = outer_source_info.scope;

        end_block.unit()
    }

    pub fn expr_into_pattern(&mut self,
                             mut block: BasicBlock,
                             irrefutable_pat: Pattern<'tcx>,
                             initializer: ExprRef<'tcx>)
                             -> BlockAnd<()> {
        // optimize the case of `let x = ...`
        match *irrefutable_pat.kind {
            PatternKind::Binding { mode: BindingMode::ByValue,
                                   var,
                                   subpattern: None, .. } => {
                self.storage_live_for_bindings(block, &irrefutable_pat);
                let lvalue = Lvalue::Local(self.var_indices[&var]);
                return self.into(&lvalue, block, initializer);
            }
            _ => {}
        }
        let lvalue = unpack!(block = self.as_lvalue(block, initializer));
        self.lvalue_into_pattern(block,
                                 irrefutable_pat,
                                 &lvalue)
    }

    pub fn lvalue_into_pattern(&mut self,
                               mut block: BasicBlock,
                               irrefutable_pat: Pattern<'tcx>,
                               initializer: &Lvalue<'tcx>)
                               -> BlockAnd<()> {
        // create a dummy candidate
        let mut candidate = Candidate {
            span: irrefutable_pat.span,
            match_pairs: vec![MatchPair::new(initializer.clone(), &irrefutable_pat)],
            bindings: vec![],
            guard: None,
            arm_index: 0, // since we don't call `match_candidates`, this field is unused
        };

        // Simplify the candidate. Since the pattern is irrefutable, this should
        // always convert all match-pairs into bindings.
        unpack!(block = self.simplify_candidate(block, &mut candidate));

        if !candidate.match_pairs.is_empty() {
            span_bug!(candidate.match_pairs[0].pattern.span,
                      "match pairs {:?} remaining after simplifying \
                       irrefutable pattern",
                      candidate.match_pairs);
        }

        // now apply the bindings, which will also declare the variables
        self.bind_matched_candidate(block, candidate.bindings);

        block.unit()
    }

    /// Declares the bindings of the given pattern and returns the visibility scope
    /// for the bindings in this patterns, if such a scope had to be created.
    /// NOTE: Declaring the bindings should always be done in their drop scope.
    pub fn declare_bindings(&mut self,
                            mut var_scope: Option<VisibilityScope>,
                            scope_span: Span,
                            pattern: &Pattern<'tcx>)
                            -> Option<VisibilityScope> {
        match *pattern.kind {
            PatternKind::Binding { mutability, name, mode: _, var, ty, ref subpattern } => {
                if var_scope.is_none() {
                    var_scope = Some(self.new_visibility_scope(scope_span));
                }
                let source_info = SourceInfo {
                    span: pattern.span,
                    scope: var_scope.unwrap()
                };
                self.declare_binding(source_info, mutability, name, var, ty);
                if let Some(subpattern) = subpattern.as_ref() {
                    var_scope = self.declare_bindings(var_scope, scope_span, subpattern);
                }
            }
            PatternKind::Array { ref prefix, ref slice, ref suffix } |
            PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
                for subpattern in prefix.iter().chain(slice).chain(suffix) {
                    var_scope = self.declare_bindings(var_scope, scope_span, subpattern);
                }
            }
            PatternKind::Constant { .. } | PatternKind::Range { .. } | PatternKind::Wild => {
            }
            PatternKind::Deref { ref subpattern } => {
                var_scope = self.declare_bindings(var_scope, scope_span, subpattern);
            }
            PatternKind::Leaf { ref subpatterns } |
            PatternKind::Variant { ref subpatterns, .. } => {
                for subpattern in subpatterns {
                    var_scope = self.declare_bindings(var_scope, scope_span, &subpattern.pattern);
                }
            }
        }
        var_scope
    }

    /// Emit `StorageLive` for every binding in the pattern.
    pub fn storage_live_for_bindings(&mut self,
                                     block: BasicBlock,
                                     pattern: &Pattern<'tcx>) {
        match *pattern.kind {
            PatternKind::Binding { var, ref subpattern, .. } => {
                let lvalue = Lvalue::Local(self.var_indices[&var]);
                let source_info = self.source_info(pattern.span);
                self.cfg.push(block, Statement {
                    source_info: source_info,
                    kind: StatementKind::StorageLive(lvalue)
                });

                if let Some(subpattern) = subpattern.as_ref() {
                    self.storage_live_for_bindings(block, subpattern);
                }
            }
            PatternKind::Array { ref prefix, ref slice, ref suffix } |
            PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
                for subpattern in prefix.iter().chain(slice).chain(suffix) {
                    self.storage_live_for_bindings(block, subpattern);
                }
            }
            PatternKind::Constant { .. } | PatternKind::Range { .. } | PatternKind::Wild => {
            }
            PatternKind::Deref { ref subpattern } => {
                self.storage_live_for_bindings(block, subpattern);
            }
            PatternKind::Leaf { ref subpatterns } |
            PatternKind::Variant { ref subpatterns, .. } => {
                for subpattern in subpatterns {
                    self.storage_live_for_bindings(block, &subpattern.pattern);
                }
            }
        }
    }
}

/// List of blocks for each arm (and potentially other metadata in the
/// future).
struct ArmBlocks {
    blocks: Vec<BasicBlock>,
}

#[derive(Clone, Debug)]
pub struct Candidate<'pat, 'tcx:'pat> {
    // span of the original pattern that gave rise to this candidate
    span: Span,

    // all of these must be satisfied...
    match_pairs: Vec<MatchPair<'pat, 'tcx>>,

    // ...these bindings established...
    bindings: Vec<Binding<'tcx>>,

    // ...and the guard must be evaluated...
    guard: Option<ExprRef<'tcx>>,

    // ...and then we branch to arm with this index.
    arm_index: usize,
}

#[derive(Clone, Debug)]
struct Binding<'tcx> {
    span: Span,
    source: Lvalue<'tcx>,
    name: Name,
    var_id: NodeId,
    var_ty: Ty<'tcx>,
    mutability: Mutability,
    binding_mode: BindingMode<'tcx>,
}

#[derive(Clone, Debug)]
pub struct MatchPair<'pat, 'tcx:'pat> {
    // this lvalue...
    lvalue: Lvalue<'tcx>,

    // ... must match this pattern.
    pattern: &'pat Pattern<'tcx>,

    // HACK(eddyb) This is used to toggle whether a Slice pattern
    // has had its length checked. This is only necessary because
    // the "rest" part of the pattern right now has type &[T] and
    // as such, it requires an Rvalue::Slice to be generated.
    // See RFC 495 / issue #23121 for the eventual (proper) solution.
    slice_len_checked: bool
}

#[derive(Clone, Debug, PartialEq)]
enum TestKind<'tcx> {
    // test the branches of enum
    Switch {
        adt_def: &'tcx AdtDef,
        variants: BitVector,
    },

    // test the branches of enum
    SwitchInt {
        switch_ty: Ty<'tcx>,
        options: Vec<ConstVal>,
        indices: FxHashMap<ConstVal, usize>,
    },

    // test for equality
    Eq {
        value: ConstVal,
        ty: Ty<'tcx>,
    },

    // test whether the value falls within an inclusive range
    Range {
        lo: Literal<'tcx>,
        hi: Literal<'tcx>,
        ty: Ty<'tcx>,
    },

    // test length of the slice is equal to len
    Len {
        len: u64,
        op: BinOp,
    },
}

#[derive(Debug)]
pub struct Test<'tcx> {
    span: Span,
    kind: TestKind<'tcx>,
}

///////////////////////////////////////////////////////////////////////////
// Main matching algorithm

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// The main match algorithm. It begins with a set of candidates
    /// `candidates` and has the job of generating code to determine
    /// which of these candidates, if any, is the correct one. The
    /// candidates are sorted such that the first item in the list
    /// has the highest priority. When a candidate is found to match
    /// the value, we will generate a branch to the appropriate
    /// block found in `arm_blocks`.
    ///
    /// The return value is a list of "otherwise" blocks. These are
    /// points in execution where we found that *NONE* of the
    /// candidates apply.  In principle, this means that the input
    /// list was not exhaustive, though at present we sometimes are
    /// not smart enough to recognize all exhaustive inputs.
    ///
    /// It might be surprising that the input can be inexhaustive.
    /// Indeed, initially, it is not, because all matches are
    /// exhaustive in Rust. But during processing we sometimes divide
    /// up the list of candidates and recurse with a non-exhaustive
    /// list. This is important to keep the size of the generated code
    /// under control. See `test_candidates` for more details.
    fn match_candidates<'pat>(&mut self,
                              span: Span,
                              arm_blocks: &mut ArmBlocks,
                              mut candidates: Vec<Candidate<'pat, 'tcx>>,
                              mut block: BasicBlock)
                              -> Vec<BasicBlock>
    {
        debug!("matched_candidate(span={:?}, block={:?}, candidates={:?})",
               span, block, candidates);

        // Start by simplifying candidates. Once this process is
        // complete, all the match pairs which remain require some
        // form of test, whether it be a switch or pattern comparison.
        for candidate in &mut candidates {
            unpack!(block = self.simplify_candidate(block, candidate));
        }

        // The candidates are sorted by priority. Check to see
        // whether the higher priority candidates (and hence at
        // the front of the vec) have satisfied all their match
        // pairs.
        let fully_matched =
            candidates.iter().take_while(|c| c.match_pairs.is_empty()).count();
        debug!("match_candidates: {:?} candidates fully matched", fully_matched);
        let mut unmatched_candidates = candidates.split_off(fully_matched);
        for candidate in candidates {
            // If so, apply any bindings, test the guard (if any), and
            // branch to the arm.
            if let Some(b) = self.bind_and_guard_matched_candidate(block, arm_blocks, candidate) {
                block = b;
            } else {
                // if None is returned, then any remaining candidates
                // are unreachable (at least not through this path).
                return vec![];
            }
        }

        // If there are no candidates that still need testing, we're done.
        // Since all matches are exhaustive, execution should never reach this point.
        if unmatched_candidates.is_empty() {
            return vec![block];
        }

        // Test candidates where possible.
        let (otherwise, tested_candidates) =
            self.test_candidates(span, arm_blocks, &unmatched_candidates, block);

        // If the target candidates were exhaustive, then we are done.
        if otherwise.is_empty() {
            return vec![];
        }

        // If all candidates were sorted into `target_candidates` somewhere, then
        // the initial set was inexhaustive.
        let untested_candidates = unmatched_candidates.split_off(tested_candidates);
        if untested_candidates.len() == 0 {
            return otherwise;
        }

        // Otherwise, let's process those remaining candidates.
        let join_block = self.join_otherwise_blocks(span, otherwise);
        self.match_candidates(span, arm_blocks, untested_candidates, join_block)
    }

    fn join_otherwise_blocks(&mut self,
                             span: Span,
                             mut otherwise: Vec<BasicBlock>)
                             -> BasicBlock
    {
        let source_info = self.source_info(span);
        otherwise.sort();
        otherwise.dedup(); // variant switches can introduce duplicate target blocks
        if otherwise.len() == 1 {
            otherwise[0]
        } else {
            let join_block = self.cfg.start_new_block();
            for block in otherwise {
                self.cfg.terminate(block, source_info,
                                   TerminatorKind::Goto { target: join_block });
            }
            join_block
        }
    }

    /// This is the most subtle part of the matching algorithm.  At
    /// this point, the input candidates have been fully simplified,
    /// and so we know that all remaining match-pairs require some
    /// sort of test. To decide what test to do, we take the highest
    /// priority candidate (last one in the list) and extract the
    /// first match-pair from the list. From this we decide what kind
    /// of test is needed using `test`, defined in the `test` module.
    ///
    /// *Note:* taking the first match pair is somewhat arbitrary, and
    /// we might do better here by choosing more carefully what to
    /// test.
    ///
    /// For example, consider the following possible match-pairs:
    ///
    /// 1. `x @ Some(P)` -- we will do a `Switch` to decide what variant `x` has
    /// 2. `x @ 22` -- we will do a `SwitchInt`
    /// 3. `x @ 3..5` -- we will do a range test
    /// 4. etc.
    ///
    /// Once we know what sort of test we are going to perform, this
    /// test may also help us with other candidates. So we walk over
    /// the candidates (from high to low priority) and check. This
    /// gives us, for each outcome of the test, a transformed list of
    /// candidates.  For example, if we are testing the current
    /// variant of `x.0`, and we have a candidate `{x.0 @ Some(v), x.1
    /// @ 22}`, then we would have a resulting candidate of `{(x.0 as
    /// Some).0 @ v, x.1 @ 22}`. Note that the first match-pair is now
    /// simpler (and, in fact, irrefutable).
    ///
    /// But there may also be candidates that the test just doesn't
    /// apply to. The classical example involves wildcards:
    ///
    /// ```rust,ignore
    /// match (x, y, z) {
    ///     (true, _, true) => true,    // (0)
    ///     (_, true, _) => true,       // (1)
    ///     (false, false, _) => false, // (2)
    ///     (true, _, false) => false,  // (3)
    /// }
    /// ```
    ///
    /// In that case, after we test on `x`, there are 2 overlapping candidate
    /// sets:
    ///
    /// - If the outcome is that `x` is true, candidates 0, 1, and 3
    /// - If the outcome is that `x` is false, candidates 1 and 2
    ///
    /// Here, the traditional "decision tree" method would generate 2
    /// separate code-paths for the 2 separate cases.
    ///
    /// In some cases, this duplication can create an exponential amount of
    /// code. This is most easily seen by noticing that this method terminates
    /// with precisely the reachable arms being reachable - but that problem
    /// is trivially NP-complete:
    ///
    /// ```rust
    ///     match (var0, var1, var2, var3, ..) {
    ///         (true, _, _, false, true, ...) => false,
    ///         (_, true, true, false, _, ...) => false,
    ///         (false, _, false, false, _, ...) => false,
    ///         ...
    ///         _ => true
    ///     }
    /// ```
    ///
    /// Here the last arm is reachable only if there is an assignment to
    /// the variables that does not match any of the literals. Therefore,
    /// compilation would take an exponential amount of time in some cases.
    ///
    /// That kind of exponential worst-case might not occur in practice, but
    /// our simplistic treatment of constants and guards would make it occur
    /// in very common situations - for example #29740:
    ///
    /// ```rust
    /// match x {
    ///     "foo" if foo_guard => ...,
    ///     "bar" if bar_guard => ...,
    ///     "baz" if baz_guard => ...,
    ///     ...
    /// }
    /// ```
    ///
    /// Here we first test the match-pair `x @ "foo"`, which is an `Eq` test.
    ///
    /// It might seem that we would end up with 2 disjoint candidate
    /// sets, consisting of the first candidate or the other 3, but our
    /// algorithm doesn't reason about "foo" being distinct from the other
    /// constants; it considers the latter arms to potentially match after
    /// both outcomes, which obviously leads to an exponential amount
    /// of tests.
    ///
    /// To avoid these kinds of problems, our algorithm tries to ensure
    /// the amount of generated tests is linear. When we do a k-way test,
    /// we return an additional "unmatched" set alongside the obvious `k`
    /// sets. When we encounter a candidate that would be present in more
    /// than one of the sets, we put it and all candidates below it into the
    /// "unmatched" set. This ensures these `k+1` sets are disjoint.
    ///
    /// After we perform our test, we branch into the appropriate candidate
    /// set and recurse with `match_candidates`. These sub-matches are
    /// obviously inexhaustive - as we discarded our otherwise set - so
    /// we set their continuation to do `match_candidates` on the
    /// "unmatched" set (which is again inexhaustive).
    ///
    /// If you apply this to the above test, you basically wind up
    /// with an if-else-if chain, testing each candidate in turn,
    /// which is precisely what we want.
    ///
    /// In addition to avoiding exponential-time blowups, this algorithm
    /// also has nice property that each guard and arm is only generated
    /// once.
    fn test_candidates<'pat>(&mut self,
                             span: Span,
                             arm_blocks: &mut ArmBlocks,
                             candidates: &[Candidate<'pat, 'tcx>],
                             block: BasicBlock)
                             -> (Vec<BasicBlock>, usize)
    {
        // extract the match-pair from the highest priority candidate
        let match_pair = &candidates.first().unwrap().match_pairs[0];
        let mut test = self.test(match_pair);

        // most of the time, the test to perform is simply a function
        // of the main candidate; but for a test like SwitchInt, we
        // may want to add cases based on the candidates that are
        // available
        match test.kind {
            TestKind::SwitchInt { switch_ty, ref mut options, ref mut indices } => {
                for candidate in candidates.iter() {
                    if !self.add_cases_to_switch(&match_pair.lvalue,
                                                 candidate,
                                                 switch_ty,
                                                 options,
                                                 indices) {
                        break;
                    }
                }
            }
            TestKind::Switch { adt_def: _, ref mut variants} => {
                for candidate in candidates.iter() {
                    if !self.add_variants_to_switch(&match_pair.lvalue,
                                                    candidate,
                                                    variants) {
                        break;
                    }
                }
            }
            _ => { }
        }

        // perform the test, branching to one of N blocks. For each of
        // those N possible outcomes, create a (initially empty)
        // vector of candidates. Those are the candidates that still
        // apply if the test has that particular outcome.
        debug!("match_candidates: test={:?} match_pair={:?}", test, match_pair);
        let target_blocks = self.perform_test(block, &match_pair.lvalue, &test);
        let mut target_candidates: Vec<_> = (0..target_blocks.len()).map(|_| vec![]).collect();

        // Sort the candidates into the appropriate vector in
        // `target_candidates`. Note that at some point we may
        // encounter a candidate where the test is not relevant; at
        // that point, we stop sorting.
        let tested_candidates =
            candidates.iter()
                      .take_while(|c| self.sort_candidate(&match_pair.lvalue,
                                                          &test,
                                                          c,
                                                          &mut target_candidates))
                      .count();
        assert!(tested_candidates > 0); // at least the last candidate ought to be tested
        debug!("tested_candidates: {}", tested_candidates);
        debug!("untested_candidates: {}", candidates.len() - tested_candidates);

        // For each outcome of test, process the candidates that still
        // apply. Collect a list of blocks where control flow will
        // branch if one of the `target_candidate` sets is not
        // exhaustive.
        let otherwise: Vec<_> =
            target_blocks.into_iter()
                         .zip(target_candidates)
                         .flat_map(|(target_block, target_candidates)| {
                             self.match_candidates(span,
                                                   arm_blocks,
                                                   target_candidates,
                                                   target_block)
                         })
                         .collect();

        (otherwise, tested_candidates)
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
    fn bind_and_guard_matched_candidate<'pat>(&mut self,
                                              mut block: BasicBlock,
                                              arm_blocks: &mut ArmBlocks,
                                              candidate: Candidate<'pat, 'tcx>)
                                              -> Option<BasicBlock> {
        debug!("bind_and_guard_matched_candidate(block={:?}, candidate={:?})",
               block, candidate);

        debug_assert!(candidate.match_pairs.is_empty());

        self.bind_matched_candidate(block, candidate.bindings);

        let arm_block = arm_blocks.blocks[candidate.arm_index];

        if let Some(guard) = candidate.guard {
            // the block to branch to if the guard fails; if there is no
            // guard, this block is simply unreachable
            let guard = self.hir.mirror(guard);
            let source_info = self.source_info(guard.span);
            let cond = unpack!(block = self.as_operand(block, guard));
            let otherwise = self.cfg.start_new_block();
            self.cfg.terminate(block, source_info,
                               TerminatorKind::If { cond: cond,
                                                    targets: (arm_block, otherwise)});
            Some(otherwise)
        } else {
            let source_info = self.source_info(candidate.span);
            self.cfg.terminate(block, source_info,
                               TerminatorKind::Goto { target: arm_block });
            None
        }
    }

    fn bind_matched_candidate(&mut self,
                              block: BasicBlock,
                              bindings: Vec<Binding<'tcx>>) {
        debug!("bind_matched_candidate(block={:?}, bindings={:?})",
               block, bindings);

        // Assign each of the bindings. This may trigger moves out of the candidate.
        for binding in bindings {
            // Find the variable for the `var_id` being bound. It
            // should have been created by a previous call to
            // `declare_bindings`.
            let var_index = self.var_indices[&binding.var_id];

            let rvalue = match binding.binding_mode {
                BindingMode::ByValue =>
                    Rvalue::Use(Operand::Consume(binding.source)),
                BindingMode::ByRef(region, borrow_kind) =>
                    Rvalue::Ref(region, borrow_kind, binding.source),
            };

            let source_info = self.source_info(binding.span);
            self.cfg.push(block, Statement {
                source_info: source_info,
                kind: StatementKind::StorageLive(Lvalue::Local(var_index))
            });
            self.cfg.push_assign(block, source_info,
                                 &Lvalue::Local(var_index), rvalue);
        }
    }

    fn declare_binding(&mut self,
                       source_info: SourceInfo,
                       mutability: Mutability,
                       name: Name,
                       var_id: NodeId,
                       var_ty: Ty<'tcx>)
                       -> Local
    {
        debug!("declare_binding(var_id={:?}, name={:?}, var_ty={:?}, source_info={:?})",
               var_id, name, var_ty, source_info);

        let var = self.local_decls.push(LocalDecl::<'tcx> {
            mutability: mutability,
            ty: var_ty.clone(),
            name: Some(name),
            source_info: Some(source_info),
        });
        let extent = self.hir.tcx().region_maps.var_scope(var_id);
        self.schedule_drop(source_info.span, extent, &Lvalue::Local(var), var_ty);
        self.var_indices.insert(var_id, var);

        debug!("declare_binding: var={:?}", var);

        var
    }
}
