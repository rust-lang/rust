//! Code related to match expressions. These are sufficiently complex to
//! warrant their own module and submodules. :) This main module includes the
//! high-level algorithm, the submodules contain the details.
//!
//! This also includes code for pattern bindings in `let` statements and
//! function parameters.

use crate::build::scope::DropKind;
use crate::build::ForGuard::{self, OutsideGuard, RefWithinGuard};
use crate::build::{BlockAnd, BlockAndExtension, Builder};
use crate::build::{GuardFrame, GuardFrameLocal, LocalsForNode};
use crate::hair::{self, *};
use rustc::hir::HirId;
use rustc::mir::*;
use rustc::middle::region;
use rustc::ty::{self, CanonicalUserTypeAnnotation, Ty};
use rustc::ty::layout::VariantIdx;
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use syntax::ast::Name;
use syntax_pos::Span;

// helper functions, broken out by category:
mod simplify;
mod test;
mod util;

use std::convert::TryFrom;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Generates MIR for a `match` expression.
    ///
    /// The MIR that we generate for a match looks like this.
    ///
    /// ```text
    /// [ 0. Pre-match ]
    ///        |
    /// [ 1. Evaluate Scrutinee (expression being matched on) ]
    /// [ (fake read of scrutinee) ]
    ///        |
    /// [ 2. Decision tree -- check discriminants ] <--------+
    ///        |                                             |
    ///        | (once a specific arm is chosen)             |
    ///        |                                             |
    /// [pre_binding_block]                           [otherwise_block]
    ///        |                                             |
    /// [ 3. Create "guard bindings" for arm ]               |
    /// [ (create fake borrows) ]                            |
    ///        |                                             |
    /// [ 4. Execute guard code ]                            |
    /// [ (read fake borrows) ] --(guard is false)-----------+
    ///        |
    ///        | (guard results in true)
    ///        |
    /// [ 5. Create real bindings and execute arm ]
    ///        |
    /// [ Exit match ]
    /// ```
    ///
    /// All of the different arms have been stacked on top of each other to
    /// simplify the diagram. For an arm with no guard the blocks marked 3 and
    /// 4 and the fake borrows are omitted.
    ///
    /// We generate MIR in the following steps:
    ///
    /// 1. Evaluate the scrutinee and add the fake read of it.
    /// 2. Create the prebinding and otherwise blocks.
    /// 3. Create the decision tree and record the places that we bind or test.
    /// 4. Determine the fake borrows that are needed from the above places.
    ///    Create the required temporaries for them.
    /// 5. Create everything else: Create everything else: the guards and the
    ///    arms.
    ///
    /// ## Fake Reads and borrows
    ///
    /// Match exhaustiveness checking is not able to handle the case where the
    /// place being matched on is mutated in the guards. There is an AST check
    /// that tries to stop this but it is buggy and overly restrictive. Instead
    /// we add "fake borrows" to the guards that prevent any mutation of the
    /// place being matched. There are a some subtleties:
    ///
    /// 1. Borrowing `*x` doesn't prevent assigning to `x`. If `x` is a shared
    ///    refence, the borrow isn't even tracked. As such we have to add fake
    ///    borrows of any prefixes of a place
    /// 2. We don't want `match x { _ => (), }` to conflict with mutable
    ///    borrows of `x`, so we only add fake borrows for places which are
    ///    bound or tested by the match.
    /// 3. We don't want the fake borrows to conflict with `ref mut` bindings,
    ///    so we use a special BorrowKind for them.
    /// 4. The fake borrows may be of places in inactive variants, so it would
    ///    be UB to generate code for them. They therefore have to be removed
    ///    by a MIR pass run after borrow checking.
    ///
    /// ## False edges
    ///
    /// We don't want to have the exact structure of the decision tree be
    /// visible through borrow checking. False edges ensure that the CFG as
    /// seen by borrow checking doesn't encode this. False edges are added:
    ///
    /// * From each prebinding block to the next prebinding block.
    /// * From each otherwise block to the next prebinding block.
    pub fn match_expr(
        &mut self,
        destination: &Place<'tcx>,
        span: Span,
        mut block: BasicBlock,
        scrutinee: ExprRef<'tcx>,
        arms: Vec<Arm<'tcx>>,
    ) -> BlockAnd<()> {
        let tcx = self.hir.tcx();

        // Step 1. Evaluate the scrutinee and add the fake read of it.

        let scrutinee_span = scrutinee.span();
        let scrutinee_place = unpack!(block = self.as_place(block, scrutinee));

        // Matching on a `scrutinee_place` with an uninhabited type doesn't
        // generate any memory reads by itself, and so if the place "expression"
        // contains unsafe operations like raw pointer dereferences or union
        // field projections, we wouldn't know to require an `unsafe` block
        // around a `match` equivalent to `std::intrinsics::unreachable()`.
        // See issue #47412 for this hole being discovered in the wild.
        //
        // HACK(eddyb) Work around the above issue by adding a dummy inspection
        // of `scrutinee_place`, specifically by applying `ReadForMatch`.
        //
        // NOTE: ReadForMatch also checks that the scrutinee is initialized.
        // This is currently needed to not allow matching on an uninitialized,
        // uninhabited value. If we get never patterns, those will check that
        // the place is initialized, and so this read would only be used to
        // check safety.

        let source_info = self.source_info(scrutinee_span);
        self.cfg.push(block, Statement {
            source_info,
            kind: StatementKind::FakeRead(
                FakeReadCause::ForMatchedPlace,
                scrutinee_place.clone(),
            ),
        });

        // Step 2. Create the otherwise and prebinding blocks.

        // create binding start block for link them by false edges
        let candidate_count = arms.iter().map(|c| c.patterns.len()).sum::<usize>();
        let pre_binding_blocks: Vec<_> = (0..candidate_count)
            .map(|_| self.cfg.start_new_block())
            .collect();

        let mut match_has_guard = false;

        let mut candidate_pre_binding_blocks = pre_binding_blocks.iter();
        let mut next_candidate_pre_binding_blocks = pre_binding_blocks.iter().skip(1);

        // Assemble a list of candidates: there is one candidate per pattern,
        // which means there may be more than one candidate *per arm*.
        let mut arm_candidates: Vec<_> = arms
            .iter()
            .map(|arm| {
                let arm_has_guard = arm.guard.is_some();
                match_has_guard |= arm_has_guard;
                let arm_candidates: Vec<_> = arm.patterns
                    .iter()
                    .zip(candidate_pre_binding_blocks.by_ref())
                    .map(
                        |(pattern, pre_binding_block)| {
                            Candidate {
                                span: pattern.span,
                                match_pairs: vec![
                                    MatchPair::new(scrutinee_place.clone(), pattern),
                                ],
                                bindings: vec![],
                                ascriptions: vec![],
                                otherwise_block: if arm_has_guard {
                                    Some(self.cfg.start_new_block())
                                } else {
                                    None
                                },
                                pre_binding_block: *pre_binding_block,
                                next_candidate_pre_binding_block:
                                    next_candidate_pre_binding_blocks.next().copied(),
                            }
                        },
                    )
                    .collect();
                (arm, arm_candidates)
            })
            .collect();

        // Step 3. Create the decision tree and record the places that we bind or test.

        // The set of places that we are creating fake borrows of. If there are
        // no match guards then we don't need any fake borrows, so don't track
        // them.
        let mut fake_borrows = if match_has_guard && tcx.generate_borrow_of_any_match_input() {
            Some(FxHashSet::default())
        } else {
            None
        };

        // These candidates are kept sorted such that the highest priority
        // candidate comes first in the list. (i.e., same order as in source)
        // As we gnerate the decision tree,
        let candidates = &mut arm_candidates
            .iter_mut()
            .flat_map(|(_, candidates)| candidates)
            .collect::<Vec<_>>();

        let outer_source_info = self.source_info(span);

        // this will generate code to test scrutinee_place and
        // branch to the appropriate arm block
        self.match_candidates(
            scrutinee_span,
            &mut Some(block),
            None,
            candidates,
            &mut fake_borrows,
        );

        // Step 4. Determine the fake borrows that are needed from the above
        // places. Create the required temporaries for them.

        let fake_borrow_temps = if let Some(ref borrows) = fake_borrows {
            self.calculate_fake_borrows(borrows, scrutinee_span)
        } else {
            Vec::new()
        };

        // Step 5. Create everything else: the guards and the arms.
        let match_scope = self.scopes.topmost();

        let arm_end_blocks: Vec<_> = arm_candidates.into_iter().map(|(arm, mut candidates)| {
            let arm_source_info = self.source_info(arm.span);
            let arm_scope = (arm.scope, arm_source_info);
            self.in_scope(arm_scope, arm.lint_level, |this| {
                let body = this.hir.mirror(arm.body.clone());
                let scope = this.declare_bindings(
                    None,
                    arm.span,
                    &arm.patterns[0],
                    ArmHasGuard(arm.guard.is_some()),
                    Some((Some(&scrutinee_place), scrutinee_span)),
                );

                let arm_block;
                if candidates.len() == 1 {
                    arm_block = this.bind_and_guard_matched_candidate(
                        candidates.pop().unwrap(),
                        arm.guard.clone(),
                        &fake_borrow_temps,
                        scrutinee_span,
                        match_scope,
                    );
                } else {
                    arm_block = this.cfg.start_new_block();
                    for candidate in candidates {
                        this.clear_top_scope(arm.scope);
                        let binding_end = this.bind_and_guard_matched_candidate(
                            candidate,
                            arm.guard.clone(),
                            &fake_borrow_temps,
                            scrutinee_span,
                            match_scope,
                        );
                        this.cfg.terminate(
                            binding_end,
                            source_info,
                            TerminatorKind::Goto { target: arm_block },
                        );
                    }
                }

                if let Some(source_scope) = scope {
                    this.source_scope = source_scope;
                }

                this.into(destination, arm_block, body)
            })
        }).collect();

        // all the arm blocks will rejoin here
        let end_block = self.cfg.start_new_block();

        for arm_block in arm_end_blocks {
            self.cfg.terminate(
                unpack!(arm_block),
                outer_source_info,
                TerminatorKind::Goto { target: end_block },
            );
        }

        self.source_scope = outer_source_info.scope;

        end_block.unit()
    }

    pub(super) fn expr_into_pattern(
        &mut self,
        mut block: BasicBlock,
        irrefutable_pat: Pattern<'tcx>,
        initializer: ExprRef<'tcx>,
    ) -> BlockAnd<()> {
        match *irrefutable_pat.kind {
            // Optimize the case of `let x = ...` to write directly into `x`
            PatternKind::Binding {
                mode: BindingMode::ByValue,
                var,
                subpattern: None,
                ..
            } => {
                let place =
                    self.storage_live_binding(block, var, irrefutable_pat.span, OutsideGuard);
                unpack!(block = self.into(&place, block, initializer));


                // Inject a fake read, see comments on `FakeReadCause::ForLet`.
                let source_info = self.source_info(irrefutable_pat.span);
                self.cfg.push(
                    block,
                    Statement {
                        source_info,
                        kind: StatementKind::FakeRead(FakeReadCause::ForLet, place),
                    },
                );

                self.schedule_drop_for_binding(var, irrefutable_pat.span, OutsideGuard);
                block.unit()
            }

            // Optimize the case of `let x: T = ...` to write directly
            // into `x` and then require that `T == typeof(x)`.
            //
            // Weirdly, this is needed to prevent the
            // `intrinsic-move-val.rs` test case from crashing. That
            // test works with uninitialized values in a rather
            // dubious way, so it may be that the test is kind of
            // broken.
            PatternKind::AscribeUserType {
                subpattern: Pattern {
                    kind: box PatternKind::Binding {
                        mode: BindingMode::ByValue,
                        var,
                        subpattern: None,
                        ..
                    },
                    ..
                },
                ascription: hair::pattern::Ascription {
                    user_ty: pat_ascription_ty,
                    variance: _,
                    user_ty_span,
                },
            } => {
                let place =
                    self.storage_live_binding(block, var, irrefutable_pat.span, OutsideGuard);
                unpack!(block = self.into(&place, block, initializer));

                // Inject a fake read, see comments on `FakeReadCause::ForLet`.
                let pattern_source_info = self.source_info(irrefutable_pat.span);
                self.cfg.push(
                    block,
                    Statement {
                        source_info: pattern_source_info,
                        kind: StatementKind::FakeRead(FakeReadCause::ForLet, place.clone()),
                    },
                );

                let ty_source_info = self.source_info(user_ty_span);
                let user_ty = box pat_ascription_ty.user_ty(
                    &mut self.canonical_user_type_annotations,
                    place.ty(&self.local_decls, self.hir.tcx()).ty,
                    ty_source_info.span,
                );
                self.cfg.push(
                    block,
                    Statement {
                        source_info: ty_source_info,
                        kind: StatementKind::AscribeUserType(
                            place,
                            // We always use invariant as the variance here. This is because the
                            // variance field from the ascription refers to the variance to use
                            // when applying the type to the value being matched, but this
                            // ascription applies rather to the type of the binding. e.g., in this
                            // example:
                            //
                            // ```
                            // let x: T = <expr>
                            // ```
                            //
                            // We are creating an ascription that defines the type of `x` to be
                            // exactly `T` (i.e., with invariance). The variance field, in
                            // contrast, is intended to be used to relate `T` to the type of
                            // `<expr>`.
                            ty::Variance::Invariant,
                            user_ty,
                        ),
                    },
                );

                self.schedule_drop_for_binding(var, irrefutable_pat.span, OutsideGuard);
                block.unit()
            }

            _ => {
                let place = unpack!(block = self.as_place(block, initializer));
                self.place_into_pattern(block, irrefutable_pat, &place, true)
            }
        }
    }

    pub fn place_into_pattern(
        &mut self,
        block: BasicBlock,
        irrefutable_pat: Pattern<'tcx>,
        initializer: &Place<'tcx>,
        set_match_place: bool,
    ) -> BlockAnd<()> {
        // create a dummy candidate
        let mut candidate = Candidate {
            span: irrefutable_pat.span,
            match_pairs: vec![MatchPair::new(initializer.clone(), &irrefutable_pat)],
            bindings: vec![],
            ascriptions: vec![],

            // since we don't call `match_candidates`, next fields are unused
            otherwise_block: None,
            pre_binding_block: block,
            next_candidate_pre_binding_block: None,
        };

        // Simplify the candidate. Since the pattern is irrefutable, this should
        // always convert all match-pairs into bindings.
        self.simplify_candidate(&mut candidate);

        if !candidate.match_pairs.is_empty() {
            // ICE if no other errors have been emitted. This used to be a hard error that wouldn't
            // be reached because `hair::pattern::check_match::check_match` wouldn't have let the
            // compiler continue. In our tests this is only ever hit by
            // `ui/consts/const-match-check.rs` with `--cfg eval1`, and that file already generates
            // a different error before hand.
            self.hir.tcx().sess.delay_span_bug(
                candidate.match_pairs[0].pattern.span,
                &format!(
                    "match pairs {:?} remaining after simplifying irrefutable pattern",
                    candidate.match_pairs,
                ),
            );
        }

        // for matches and function arguments, the place that is being matched
        // can be set when creating the variables. But the place for
        // let PATTERN = ... might not even exist until we do the assignment.
        // so we set it here instead
        if set_match_place {
            for binding in &candidate.bindings {
                let local = self.var_local_id(binding.var_id, OutsideGuard);

                if let Some(ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                    opt_match_place: Some((ref mut match_place, _)),
                    ..
                }))) = self.local_decls[local].is_user_variable
                {
                    *match_place = Some(initializer.clone());
                } else {
                    bug!("Let binding to non-user variable.")
                }
            }
        }

        self.ascribe_types(block, &candidate.ascriptions);

        // now apply the bindings, which will also declare the variables
        self.bind_matched_candidate_for_arm_body(block, &candidate.bindings);

        block.unit()
    }

    /// Declares the bindings of the given patterns and returns the visibility
    /// scope for the bindings in these patterns, if such a scope had to be
    /// created. NOTE: Declaring the bindings should always be done in their
    /// drop scope.
    pub fn declare_bindings(
        &mut self,
        mut visibility_scope: Option<SourceScope>,
        scope_span: Span,
        pattern: &Pattern<'tcx>,
        has_guard: ArmHasGuard,
        opt_match_place: Option<(Option<&Place<'tcx>>, Span)>,
    ) -> Option<SourceScope> {
        debug!("declare_bindings: pattern={:?}", pattern);
        self.visit_bindings(
            &pattern,
            UserTypeProjections::none(),
            &mut |this, mutability, name, mode, var, span, ty, user_ty| {
                if visibility_scope.is_none() {
                    visibility_scope =
                        Some(this.new_source_scope(scope_span, LintLevel::Inherited, None));
                }
                let source_info = SourceInfo { span, scope: this.source_scope };
                let visibility_scope = visibility_scope.unwrap();
                this.declare_binding(
                    source_info,
                    visibility_scope,
                    mutability,
                    name,
                    mode,
                    var,
                    ty,
                    user_ty,
                    has_guard,
                    opt_match_place.map(|(x, y)| (x.cloned(), y)),
                    pattern.span,
                );
            },
        );
        visibility_scope
    }

    pub fn storage_live_binding(
        &mut self,
        block: BasicBlock,
        var: HirId,
        span: Span,
        for_guard: ForGuard,
    ) -> Place<'tcx> {
        let local_id = self.var_local_id(var, for_guard);
        let source_info = self.source_info(span);
        self.cfg.push(
            block,
            Statement {
                source_info,
                kind: StatementKind::StorageLive(local_id),
            },
        );
        let var_ty = self.local_decls[local_id].ty;
        let region_scope = self.hir.region_scope_tree.var_scope(var.local_id);
        self.schedule_drop(span, region_scope, local_id, var_ty, DropKind::Storage);
        Place::Base(PlaceBase::Local(local_id))
    }

    pub fn schedule_drop_for_binding(&mut self, var: HirId, span: Span, for_guard: ForGuard) {
        let local_id = self.var_local_id(var, for_guard);
        let var_ty = self.local_decls[local_id].ty;
        let region_scope = self.hir.region_scope_tree.var_scope(var.local_id);
        self.schedule_drop(
            span,
            region_scope,
            local_id,
            var_ty,
            DropKind::Value,
        );
    }

    pub(super) fn visit_bindings(
        &mut self,
        pattern: &Pattern<'tcx>,
        pattern_user_ty: UserTypeProjections,
        f: &mut impl FnMut(
            &mut Self,
            Mutability,
            Name,
            BindingMode,
            HirId,
            Span,
            Ty<'tcx>,
            UserTypeProjections,
        ),
    ) {
        debug!("visit_bindings: pattern={:?} pattern_user_ty={:?}", pattern, pattern_user_ty);
        match *pattern.kind {
            PatternKind::Binding {
                mutability,
                name,
                mode,
                var,
                ty,
                ref subpattern,
                ..
            } => {
                f(self, mutability, name, mode, var, pattern.span, ty, pattern_user_ty.clone());
                if let Some(subpattern) = subpattern.as_ref() {
                    self.visit_bindings(subpattern, pattern_user_ty, f);
                }
            }

            PatternKind::Array {
                ref prefix,
                ref slice,
                ref suffix,
            }
            | PatternKind::Slice {
                ref prefix,
                ref slice,
                ref suffix,
            } => {
                let from = u32::try_from(prefix.len()).unwrap();
                let to = u32::try_from(suffix.len()).unwrap();
                for subpattern in prefix {
                    self.visit_bindings(subpattern, pattern_user_ty.clone().index(), f);
                }
                for subpattern in slice {
                    self.visit_bindings(subpattern, pattern_user_ty.clone().subslice(from, to), f);
                }
                for subpattern in suffix {
                    self.visit_bindings(subpattern, pattern_user_ty.clone().index(), f);
                }
            }

            PatternKind::Constant { .. } | PatternKind::Range { .. } | PatternKind::Wild => {}

            PatternKind::Deref { ref subpattern } => {
                self.visit_bindings(subpattern, pattern_user_ty.deref(), f);
            }

            PatternKind::AscribeUserType {
                ref subpattern,
                ascription: hair::pattern::Ascription {
                    ref user_ty,
                    user_ty_span,
                    variance: _,
                },
            } => {
                // This corresponds to something like
                //
                // ```
                // let A::<'a>(_): A<'static> = ...;
                // ```
                //
                // Note that the variance doesn't apply here, as we are tracking the effect
                // of `user_ty` on any bindings contained with subpattern.
                let annotation = CanonicalUserTypeAnnotation {
                    span: user_ty_span,
                    user_ty: user_ty.user_ty,
                    inferred_ty: subpattern.ty,
                };
                let projection = UserTypeProjection {
                    base: self.canonical_user_type_annotations.push(annotation),
                    projs: Vec::new(),
                };
                let subpattern_user_ty = pattern_user_ty.push_projection(&projection, user_ty_span);
                self.visit_bindings(subpattern, subpattern_user_ty, f)
            }

            PatternKind::Leaf { ref subpatterns } => {
                for subpattern in subpatterns {
                    let subpattern_user_ty = pattern_user_ty.clone().leaf(subpattern.field);
                    debug!("visit_bindings: subpattern_user_ty={:?}", subpattern_user_ty);
                    self.visit_bindings(&subpattern.pattern, subpattern_user_ty, f);
                }
            }

            PatternKind::Variant { adt_def, substs: _, variant_index, ref subpatterns } => {
                for subpattern in subpatterns {
                    let subpattern_user_ty = pattern_user_ty.clone().variant(
                        adt_def, variant_index, subpattern.field);
                    self.visit_bindings(&subpattern.pattern, subpattern_user_ty, f);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Candidate<'pat, 'tcx> {
    // span of the original pattern that gave rise to this candidate
    span: Span,

    // all of these must be satisfied...
    match_pairs: Vec<MatchPair<'pat, 'tcx>>,

    // ...these bindings established...
    bindings: Vec<Binding<'tcx>>,

    // ...and these types asserted...
    ascriptions: Vec<Ascription<'tcx>>,

    // ...and the guard must be evaluated, if false branch to Block...
    otherwise_block: Option<BasicBlock>,

    // ...and the blocks for add false edges between candidates
    pre_binding_block: BasicBlock,
    next_candidate_pre_binding_block: Option<BasicBlock>,
}

#[derive(Clone, Debug)]
struct Binding<'tcx> {
    span: Span,
    source: Place<'tcx>,
    name: Name,
    var_id: HirId,
    var_ty: Ty<'tcx>,
    mutability: Mutability,
    binding_mode: BindingMode,
}

/// Indicates that the type of `source` must be a subtype of the
/// user-given type `user_ty`; this is basically a no-op but can
/// influence region inference.
#[derive(Clone, Debug)]
struct Ascription<'tcx> {
    span: Span,
    source: Place<'tcx>,
    user_ty: PatternTypeProjection<'tcx>,
    variance: ty::Variance,
}

#[derive(Clone, Debug)]
pub struct MatchPair<'pat, 'tcx> {
    // this place...
    place: Place<'tcx>,

    // ... must match this pattern.
    pattern: &'pat Pattern<'tcx>,
}

#[derive(Clone, Debug, PartialEq)]
enum TestKind<'tcx> {
    /// Test the branches of enum.
    Switch {
        /// The enum being tested
        adt_def: &'tcx ty::AdtDef,
        /// The set of variants that we should create a branch for. We also
        /// create an additional "otherwise" case.
        variants: BitSet<VariantIdx>,
    },

    /// Test what value an `integer`, `bool` or `char` has.
    SwitchInt {
        /// The type of the value that we're testing.
        switch_ty: Ty<'tcx>,
        /// The (ordered) set of values that we test for.
        ///
        /// For integers and `char`s we create a branch to each of the values in
        /// `options`, as well as an "otherwise" branch for all other values, even
        /// in the (rare) case that options is exhaustive.
        ///
        /// For `bool` we always generate two edges, one for `true` and one for
        /// `false`.
        options: Vec<u128>,
        /// Reverse map used to ensure that the values in `options` are unique.
        indices: FxHashMap<&'tcx ty::Const<'tcx>, usize>,
    },

    /// Test for equality with value, possibly after an unsizing coercion to
    /// `ty`,
    Eq {
        value: &'tcx ty::Const<'tcx>,
        // Integer types are handled by `SwitchInt`, and constants with ADT
        // types are converted back into patterns, so this can only be `&str`,
        // `&[T]`, `f32` or `f64`.
        ty: Ty<'tcx>,
    },

    /// Test whether the value falls within an inclusive or exclusive range
    Range(PatternRange<'tcx>),

    /// Test length of the slice is equal to len
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

/// ArmHasGuard is isomorphic to a boolean flag. It indicates whether
/// a match arm has a guard expression attached to it.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ArmHasGuard(pub bool);

///////////////////////////////////////////////////////////////////////////
// Main matching algorithm

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// The main match algorithm. It begins with a set of candidates
    /// `candidates` and has the job of generating code to determine
    /// which of these candidates, if any, is the correct one. The
    /// candidates are sorted such that the first item in the list
    /// has the highest priority. When a candidate is found to match
    /// the value, we will generate a branch to the appropriate
    /// prebinding block.
    ///
    /// If we find that *NONE* of the candidates apply, we branch to the
    /// `otherwise_block`. In principle, this means that the input list was not
    /// exhaustive, though at present we sometimes are not smart enough to
    /// recognize all exhaustive inputs.
    ///
    /// It might be surprising that the input can be inexhaustive.
    /// Indeed, initially, it is not, because all matches are
    /// exhaustive in Rust. But during processing we sometimes divide
    /// up the list of candidates and recurse with a non-exhaustive
    /// list. This is important to keep the size of the generated code
    /// under control. See `test_candidates` for more details.
    ///
    /// If `fake_borrows` is Some, then places which need fake borrows
    /// will be added to it.
    fn match_candidates<'pat>(
        &mut self,
        span: Span,
        start_block: &mut Option<BasicBlock>,
        otherwise_block: Option<BasicBlock>,
        candidates: &mut [&mut Candidate<'pat, 'tcx>],
        fake_borrows: &mut Option<FxHashSet<Place<'tcx>>>,
    ) {
        debug!(
            "matched_candidate(span={:?}, candidates={:?}, start_block={:?}, otherwise_block={:?})",
            span,
            candidates,
            start_block,
            otherwise_block,
        );

        // Start by simplifying candidates. Once this process is complete, all
        // the match pairs which remain require some form of test, whether it
        // be a switch or pattern comparison.
        for candidate in &mut *candidates {
            self.simplify_candidate(candidate);
        }

        // The candidates are sorted by priority. Check to see whether the
        // higher priority candidates (and hence at the front of the slice)
        // have satisfied all their match pairs.
        let fully_matched = candidates
            .iter()
            .take_while(|c| c.match_pairs.is_empty())
            .count();
        debug!(
            "match_candidates: {:?} candidates fully matched",
            fully_matched
        );
        let (matched_candidates, unmatched_candidates) = candidates.split_at_mut(fully_matched);

        let block: BasicBlock;

        if !matched_candidates.is_empty() {
            let otherwise_block = self.select_matched_candidates(
                matched_candidates,
                start_block,
                fake_borrows,
            );

            if let Some(last_otherwise_block) = otherwise_block {
                block = last_otherwise_block
            } else {
                // Any remaining candidates are unreachable.
                if unmatched_candidates.is_empty() {
                    return;
                }
                block = self.cfg.start_new_block();
            };
        } else {
            block = *start_block.get_or_insert_with(|| self.cfg.start_new_block());
        }

        // If there are no candidates that still need testing, we're
        // done. Since all matches are exhaustive, execution should
        // never reach this point.
        if unmatched_candidates.is_empty() {
            let source_info = self.source_info(span);
            if let Some(otherwise) = otherwise_block {
                self.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Goto { target: otherwise },
                );
            } else {
                self.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::Unreachable,
                )
            }
            return;
        }

        // Test for the remaining candidates.
        self.test_candidates(
            span,
            unmatched_candidates,
            block,
            otherwise_block,
            fake_borrows,
        );
    }

    /// Link up matched candidates. For example, if we have something like
    /// this:
    ///
    /// ...
    /// Some(x) if cond => ...
    /// Some(x) => ...
    /// Some(x) if cond => ...
    /// ...
    ///
    /// We generate real edges from:
    /// * `block` to the prebinding_block of the first pattern,
    /// * the otherwise block of the first pattern to the second pattern,
    /// * the otherwise block of the third pattern to the a block with an
    ///   Unreachable terminator.
    ///
    /// As well as that we add fake edges from the otherwise blocks to the
    /// prebinding block of the next candidate in the original set of
    /// candidates.
    fn select_matched_candidates(
        &mut self,
        matched_candidates: &mut [&mut Candidate<'_, 'tcx>],
        start_block: &mut Option<BasicBlock>,
        fake_borrows: &mut Option<FxHashSet<Place<'tcx>>>,
    ) -> Option<BasicBlock> {
        debug_assert!(
            !matched_candidates.is_empty(),
            "select_matched_candidates called with no candidates",
        );

        // Insert a borrows of prefixes of places that are bound and are
        // behind a dereference projection.
        //
        // These borrows are taken to avoid situations like the following:
        //
        // match x[10] {
        //     _ if { x = &[0]; false } => (),
        //     y => (), // Out of bounds array access!
        // }
        //
        // match *x {
        //     // y is bound by reference in the guard and then by copy in the
        //     // arm, so y is 2 in the arm!
        //     y if { y == 1 && (x = &2) == () } => y,
        //     _ => 3,
        // }
        if let Some(fake_borrows) = fake_borrows {
            for Binding { source, .. }
                in matched_candidates.iter().flat_map(|candidate| &candidate.bindings)
            {
                let mut cursor = source;
                while let Place::Projection(box Projection { base, elem }) = cursor {
                    cursor = base;
                    if let ProjectionElem::Deref = elem {
                        fake_borrows.insert(cursor.clone());
                        break;
                    }
                }
            }
        }

        let fully_matched_with_guard = matched_candidates
            .iter()
            .position(|c| c.otherwise_block.is_none())
            .unwrap_or(matched_candidates.len() - 1);

        let (reachable_candidates, unreachable_candidates)
            = matched_candidates.split_at_mut(fully_matched_with_guard + 1);

        let first_candidate = &reachable_candidates[0];
        let first_prebinding_block = first_candidate.pre_binding_block;

        if let Some(start_block) = *start_block {
            let source_info = self.source_info(first_candidate.span);
            self.cfg.terminate(
                start_block,
                source_info,
                TerminatorKind::Goto { target: first_prebinding_block },
            );
        } else {
            *start_block = Some(first_prebinding_block);
        }

        for window in reachable_candidates.windows(2) {
            if let [first_candidate, second_candidate] = window {
                let source_info = self.source_info(first_candidate.span);
                if let Some(otherwise_block) = first_candidate.otherwise_block {
                    self.false_edges(
                        otherwise_block,
                        second_candidate.pre_binding_block,
                        first_candidate.next_candidate_pre_binding_block,
                        source_info,
                    );
                } else {
                    bug!("candidate other than the last has no guard");
                }
            } else {
                bug!("<[_]>::windows returned incorrectly sized window");
            }
        }

        debug!("match_candidates: add false edges for unreachable {:?}", unreachable_candidates);
        for candidate in unreachable_candidates {
            if let Some(otherwise) = candidate.otherwise_block {
                let source_info = self.source_info(candidate.span);
                let unreachable = self.cfg.start_new_block();
                self.false_edges(
                    otherwise,
                    unreachable,
                    candidate.next_candidate_pre_binding_block,
                    source_info,
                );
                self.cfg.terminate(unreachable, source_info, TerminatorKind::Unreachable);
            }
        }

        let last_candidate = reachable_candidates.last().unwrap();

        if let Some(otherwise) = last_candidate.otherwise_block {
            let source_info = self.source_info(last_candidate.span);
            let block = self.cfg.start_new_block();
            self.false_edges(
                otherwise,
                block,
                last_candidate.next_candidate_pre_binding_block,
                source_info,
            );
            Some(block)
        } else {
            None
        }
    }

    /// This is the most subtle part of the matching algorithm. At
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
    /// Tests may also help us with other candidates. So we walk over
    /// the candidates (from high to low priority) and check. This
    /// gives us, for each outcome of the test, a transformed list of
    /// candidates. For example, if we are testing the current
    /// variant of `x.0`, and we have a candidate `{x.0 @ Some(v), x.1
    /// @ 22}`, then we would have a resulting candidate of `{(x.0 as
    /// Some).0 @ v, x.1 @ 22}`. Note that the first match-pair is now
    /// simpler (and, in fact, irrefutable).
    ///
    /// But there may also be candidates that the test just doesn't
    /// apply to. The classical example involves wildcards:
    ///
    /// ```
    /// # let (x, y, z) = (true, true, true);
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
    fn test_candidates<'pat, 'b, 'c>(
        &mut self,
        span: Span,
        mut candidates: &'b mut [&'c mut Candidate<'pat, 'tcx>],
        block: BasicBlock,
        mut otherwise_block: Option<BasicBlock>,
        fake_borrows: &mut Option<FxHashSet<Place<'tcx>>>,
    ) {
        // extract the match-pair from the highest priority candidate
        let match_pair = &candidates.first().unwrap().match_pairs[0];
        let mut test = self.test(match_pair);
        let match_place = match_pair.place.clone();

        // most of the time, the test to perform is simply a function
        // of the main candidate; but for a test like SwitchInt, we
        // may want to add cases based on the candidates that are
        // available
        match test.kind {
            TestKind::SwitchInt {
                switch_ty,
                ref mut options,
                ref mut indices,
            } => {
                for candidate in candidates.iter() {
                    if !self.add_cases_to_switch(
                        &match_place,
                        candidate,
                        switch_ty,
                        options,
                        indices,
                    ) {
                        break;
                    }
                }
            }
            TestKind::Switch {
                adt_def: _,
                ref mut variants,
            } => {
                for candidate in candidates.iter() {
                    if !self.add_variants_to_switch(&match_place, candidate, variants) {
                        break;
                    }
                }
            }
            _ => {}
        }

        // Insert a Shallow borrow of any places that is switched on.
        fake_borrows.as_mut().map(|fb| {
            fb.insert(match_place.clone())
        });

        // perform the test, branching to one of N blocks. For each of
        // those N possible outcomes, create a (initially empty)
        // vector of candidates. Those are the candidates that still
        // apply if the test has that particular outcome.
        debug!(
            "match_candidates: test={:?} match_pair={:?}",
            test, match_pair
        );
        let mut target_candidates: Vec<Vec<&mut Candidate<'pat, 'tcx>>> = vec![];
        target_candidates.resize_with(test.targets(), Default::default);

        let total_candidate_count = candidates.len();

        // Sort the candidates into the appropriate vector in
        // `target_candidates`. Note that at some point we may
        // encounter a candidate where the test is not relevant; at
        // that point, we stop sorting.
        while let Some(candidate) = candidates.first_mut() {
            if let Some(idx) = self.sort_candidate(&match_place, &test, candidate) {
                let (candidate, rest) = candidates.split_first_mut().unwrap();
                target_candidates[idx].push(candidate);
                candidates = rest;
            } else {
                break;
            }
        }
        // at least the first candidate ought to be tested
        assert!(total_candidate_count > candidates.len());
        debug!("tested_candidates: {}", total_candidate_count - candidates.len());
        debug!("untested_candidates: {}", candidates.len());

        // HACK(matthewjasper) This is a closure so that we can let the test
        // create its blocks before the rest of the match. This currently
        // improves the speed of llvm when optimizing long string literal
        // matches
        let make_target_blocks = move |this: &mut Self| -> Vec<BasicBlock> {
            // For each outcome of test, process the candidates that still
            // apply. Collect a list of blocks where control flow will
            // branch if one of the `target_candidate` sets is not
            // exhaustive.
            if !candidates.is_empty() {
                let remainder_start = &mut None;
                this.match_candidates(
                    span,
                    remainder_start,
                    otherwise_block,
                    candidates,
                    fake_borrows,
                );
                otherwise_block = Some(remainder_start.unwrap());
            };

            target_candidates.into_iter().map(|mut candidates| {
                if candidates.len() != 0 {
                    let candidate_start = &mut None;
                    this.match_candidates(
                        span,
                        candidate_start,
                        otherwise_block,
                        &mut *candidates,
                        fake_borrows,
                    );
                    candidate_start.unwrap()
                } else {
                    *otherwise_block.get_or_insert_with(|| {
                        let unreachable = this.cfg.start_new_block();
                        let source_info = this.source_info(span);
                        this.cfg.terminate(
                            unreachable,
                            source_info,
                            TerminatorKind::Unreachable,
                        );
                        unreachable
                    })
                }
            }).collect()
        };

        self.perform_test(
            block,
            &match_place,
            &test,
            make_target_blocks,
        );
    }

    // Determine the fake borrows that are needed to ensure that the place
    // will evaluate to the same thing until an arm has been chosen.
    fn calculate_fake_borrows<'b>(
        &mut self,
        fake_borrows: &'b FxHashSet<Place<'tcx>>,
        temp_span: Span,
    ) -> Vec<(&'b Place<'tcx>, Local)> {
        let tcx = self.hir.tcx();

        debug!("add_fake_borrows fake_borrows = {:?}", fake_borrows);

        let mut all_fake_borrows = Vec::with_capacity(fake_borrows.len());

        // Insert a Shallow borrow of the prefixes of any fake borrows.
        for place in fake_borrows
        {
            let mut prefix_cursor = place;
            while let Place::Projection(box Projection { base, elem }) = prefix_cursor {
                if let ProjectionElem::Deref = elem {
                    // Insert a shallow borrow after a deref. For other
                    // projections the borrow of prefix_cursor will
                    // conflict with any mutation of base.
                    all_fake_borrows.push(base);
                }
                prefix_cursor = base;
            }

            all_fake_borrows.push(place);
        }

        // Deduplicate and ensure a deterministic order.
        all_fake_borrows.sort();
        all_fake_borrows.dedup();

        debug!("add_fake_borrows all_fake_borrows = {:?}", all_fake_borrows);

        all_fake_borrows.into_iter().map(|matched_place| {
            let fake_borrow_deref_ty = matched_place.ty(&self.local_decls, tcx).ty;
            let fake_borrow_ty = tcx.mk_imm_ref(tcx.lifetimes.re_erased, fake_borrow_deref_ty);
            let fake_borrow_temp = self.local_decls.push(
                LocalDecl::new_temp(fake_borrow_ty, temp_span)
            );

            (matched_place, fake_borrow_temp)
        }).collect()
    }
}

///////////////////////////////////////////////////////////////////////////
// Pattern binding - used for `let` and function parameters as well.

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Initializes each of the bindings from the candidate by
    /// moving/copying/ref'ing the source as appropriate. Tests the guard, if
    /// any, and then branches to the arm. Returns the block for the case where
    /// the guard fails.
    ///
    /// Note: we check earlier that if there is a guard, there cannot be move
    /// bindings (unless feature(bind_by_move_pattern_guards) is used). This
    /// isn't really important for the self-consistency of this fn, but the
    /// reason for it should be clear: after we've done the assignments, if
    /// there were move bindings, further tests would be a use-after-move.
    /// bind_by_move_pattern_guards avoids this by only moving the binding once
    /// the guard has evaluated to true (see below).
    fn bind_and_guard_matched_candidate<'pat>(
        &mut self,
        candidate: Candidate<'pat, 'tcx>,
        guard: Option<Guard<'tcx>>,
        fake_borrows: &Vec<(&Place<'tcx>, Local)>,
        scrutinee_span: Span,
        region_scope: region::Scope,
    ) -> BasicBlock {
        debug!("bind_and_guard_matched_candidate(candidate={:?})", candidate);

        debug_assert!(candidate.match_pairs.is_empty());

        let candidate_source_info = self.source_info(candidate.span);

        let mut block = candidate.pre_binding_block;

        // If we are adding our own statements, then we need a fresh block.
        let create_fresh_block = candidate.next_candidate_pre_binding_block.is_some()
            || !candidate.bindings.is_empty()
            || !candidate.ascriptions.is_empty()
            || guard.is_some();

        if create_fresh_block {
            let fresh_block = self.cfg.start_new_block();
            self.false_edges(
                block,
                fresh_block,
                candidate.next_candidate_pre_binding_block,
                candidate_source_info,
            );
            block = fresh_block;
            self.ascribe_types(block, &candidate.ascriptions);
        } else {
            return block;
        }

        // rust-lang/rust#27282: The `autoref` business deserves some
        // explanation here.
        //
        // The intent of the `autoref` flag is that when it is true,
        // then any pattern bindings of type T will map to a `&T`
        // within the context of the guard expression, but will
        // continue to map to a `T` in the context of the arm body. To
        // avoid surfacing this distinction in the user source code
        // (which would be a severe change to the language and require
        // far more revision to the compiler), when `autoref` is true,
        // then any occurrence of the identifier in the guard
        // expression will automatically get a deref op applied to it.
        //
        // So an input like:
        //
        // ```
        // let place = Foo::new();
        // match place { foo if inspect(foo)
        //     => feed(foo), ...  }
        // ```
        //
        // will be treated as if it were really something like:
        //
        // ```
        // let place = Foo::new();
        // match place { Foo { .. } if { let tmp1 = &place; inspect(*tmp1) }
        //     => { let tmp2 = place; feed(tmp2) }, ... }
        //
        // And an input like:
        //
        // ```
        // let place = Foo::new();
        // match place { ref mut foo if inspect(foo)
        //     => feed(foo), ...  }
        // ```
        //
        // will be treated as if it were really something like:
        //
        // ```
        // let place = Foo::new();
        // match place { Foo { .. } if { let tmp1 = & &mut place; inspect(*tmp1) }
        //     => { let tmp2 = &mut place; feed(tmp2) }, ... }
        // ```
        //
        // In short, any pattern binding will always look like *some*
        // kind of `&T` within the guard at least in terms of how the
        // MIR-borrowck views it, and this will ensure that guard
        // expressions cannot mutate their the match inputs via such
        // bindings. (It also ensures that guard expressions can at
        // most *copy* values from such bindings; non-Copy things
        // cannot be moved via pattern bindings in guard expressions.)
        //
        // ----
        //
        // Implementation notes (under assumption `autoref` is true).
        //
        // To encode the distinction above, we must inject the
        // temporaries `tmp1` and `tmp2`.
        //
        // There are two cases of interest: binding by-value, and binding by-ref.
        //
        // 1. Binding by-value: Things are simple.
        //
        //    * Establishing `tmp1` creates a reference into the
        //      matched place. This code is emitted by
        //      bind_matched_candidate_for_guard.
        //
        //    * `tmp2` is only initialized "lazily", after we have
        //      checked the guard. Thus, the code that can trigger
        //      moves out of the candidate can only fire after the
        //      guard evaluated to true. This initialization code is
        //      emitted by bind_matched_candidate_for_arm.
        //
        // 2. Binding by-reference: Things are tricky.
        //
        //    * Here, the guard expression wants a `&&` or `&&mut`
        //      into the original input. This means we need to borrow
        //      the reference that we create for the arm.
        //    * So we eagerly create the reference for the arm and then take a
        //      reference to that.
        if let Some(guard) = guard {
            let tcx = self.hir.tcx();

            self.bind_matched_candidate_for_guard(
                block,
                &candidate.bindings,
            );
            let guard_frame = GuardFrame {
                locals: candidate
                    .bindings
                    .iter()
                    .map(|b| GuardFrameLocal::new(b.var_id, b.binding_mode))
                    .collect(),
            };
            debug!("Entering guard building context: {:?}", guard_frame);
            self.guard_context.push(guard_frame);

            let re_erased = tcx.lifetimes.re_erased;
            let scrutinee_source_info = self.source_info(scrutinee_span);
            for &(place, temp) in fake_borrows {
                let borrow = Rvalue::Ref(
                    re_erased,
                    BorrowKind::Shallow,
                    place.clone(),
                );
                self.cfg.push_assign(
                    block,
                    scrutinee_source_info,
                    &Place::from(temp),
                    borrow,
                );
            }

            // the block to branch to if the guard fails; if there is no
            // guard, this block is simply unreachable
            let guard = match guard {
                Guard::If(e) => self.hir.mirror(e),
            };
            let source_info = self.source_info(guard.span);
            let guard_end = self.source_info(tcx.sess.source_map().end_point(guard.span));
            let (post_guard_block, otherwise_post_guard_block)
                = self.test_bool(block, guard, source_info);
            let guard_frame = self.guard_context.pop().unwrap();
            debug!(
                "Exiting guard building context with locals: {:?}",
                guard_frame
            );

            for &(_, temp) in fake_borrows {
                self.cfg.push(post_guard_block, Statement {
                    source_info: guard_end,
                    kind: StatementKind::FakeRead(
                        FakeReadCause::ForMatchGuard,
                        Place::from(temp),
                    ),
                });
            }

            self.exit_scope(
                source_info.span,
                region_scope,
                otherwise_post_guard_block,
                candidate.otherwise_block.unwrap(),
            );

            // We want to ensure that the matched candidates are bound
            // after we have confirmed this candidate *and* any
            // associated guard; Binding them on `block` is too soon,
            // because that would be before we've checked the result
            // from the guard.
            //
            // But binding them on the arm is *too late*, because
            // then all of the candidates for a single arm would be
            // bound in the same place, that would cause a case like:
            //
            // ```rust
            // match (30, 2) {
            //     (mut x, 1) | (2, mut x) if { true } => { ... }
            //     ...                                 // ^^^^^^^ (this is `arm_block`)
            // }
            // ```
            //
            // would yield a `arm_block` something like:
            //
            // ```
            // StorageLive(_4);        // _4 is `x`
            // _4 = &mut (_1.0: i32);  // this is handling `(mut x, 1)` case
            // _4 = &mut (_1.1: i32);  // this is handling `(2, mut x)` case
            // ```
            //
            // and that is clearly not correct.
            let by_value_bindings = candidate.bindings.iter().filter(|binding| {
                if let BindingMode::ByValue = binding.binding_mode { true } else { false }
            });
            // Read all of the by reference bindings to ensure that the
            // place they refer to can't be modified by the guard.
            for binding in by_value_bindings.clone() {
                let local_id = self.var_local_id(binding.var_id, RefWithinGuard);
                    let place = Place::from(local_id);
                self.cfg.push(
                    post_guard_block,
                    Statement {
                        source_info: guard_end,
                        kind: StatementKind::FakeRead(FakeReadCause::ForGuardBinding, place),
                    },
                );
            }
            self.bind_matched_candidate_for_arm_body(
                post_guard_block,
                by_value_bindings,
            );

            post_guard_block
        } else {
            assert!(candidate.otherwise_block.is_none());
            // (Here, it is not too early to bind the matched
            // candidate on `block`, because there is no guard result
            // that we have to inspect before we bind them.)
            self.bind_matched_candidate_for_arm_body(block, &candidate.bindings);
            block
        }
    }

    /// Append `AscribeUserType` statements onto the end of `block`
    /// for each ascription
    fn ascribe_types(&mut self, block: BasicBlock, ascriptions: &[Ascription<'tcx>]) {
        for ascription in ascriptions {
            let source_info = self.source_info(ascription.span);

            debug!(
                "adding user ascription at span {:?} of place {:?} and {:?}",
                source_info.span,
                ascription.source,
                ascription.user_ty,
            );

            let user_ty = box ascription.user_ty.clone().user_ty(
                &mut self.canonical_user_type_annotations,
                ascription.source.ty(&self.local_decls, self.hir.tcx()).ty,
                source_info.span
            );
            self.cfg.push(
                block,
                Statement {
                    source_info,
                    kind: StatementKind::AscribeUserType(
                        ascription.source.clone(),
                        ascription.variance,
                        user_ty,
                    ),
                },
            );
        }
    }

    fn bind_matched_candidate_for_guard(
        &mut self,
        block: BasicBlock,
        bindings: &[Binding<'tcx>],
    ) {
        debug!("bind_matched_candidate_for_guard(block={:?}, bindings={:?})", block, bindings);

        // Assign each of the bindings. Since we are binding for a
        // guard expression, this will never trigger moves out of the
        // candidate.
        let re_erased = self.hir.tcx().lifetimes.re_erased;
        for binding in bindings {
            let source_info = self.source_info(binding.span);

            // For each pattern ident P of type T, `ref_for_guard` is
            // a reference R: &T pointing to the location matched by
            // the pattern, and every occurrence of P within a guard
            // denotes *R.
            let ref_for_guard =
                self.storage_live_binding(block, binding.var_id, binding.span, RefWithinGuard);
            match binding.binding_mode {
                BindingMode::ByValue => {
                    let rvalue = Rvalue::Ref(re_erased, BorrowKind::Shared, binding.source.clone());
                    self.cfg
                        .push_assign(block, source_info, &ref_for_guard, rvalue);
                }
                BindingMode::ByRef(borrow_kind) => {
                    let value_for_arm = self.storage_live_binding(
                        block,
                        binding.var_id,
                        binding.span,
                        OutsideGuard,
                    );

                    let rvalue = Rvalue::Ref(re_erased, borrow_kind, binding.source.clone());
                    self.cfg
                        .push_assign(block, source_info, &value_for_arm, rvalue);
                    let rvalue = Rvalue::Ref(re_erased, BorrowKind::Shared, value_for_arm);
                    self.cfg
                        .push_assign(block, source_info, &ref_for_guard, rvalue);
                }
            }
        }
    }

    fn bind_matched_candidate_for_arm_body<'b>(
        &mut self,
        block: BasicBlock,
        bindings: impl IntoIterator<Item = &'b Binding<'tcx>>,
    ) where 'tcx: 'b {
        debug!("bind_matched_candidate_for_arm_body(block={:?})", block);

        let re_erased = self.hir.tcx().lifetimes.re_erased;
        // Assign each of the bindings. This may trigger moves out of the candidate.
        for binding in bindings {
            let source_info = self.source_info(binding.span);
            let local =
                self.storage_live_binding(block, binding.var_id, binding.span, OutsideGuard);
            self.schedule_drop_for_binding(binding.var_id, binding.span, OutsideGuard);
            let rvalue = match binding.binding_mode {
                BindingMode::ByValue => {
                    Rvalue::Use(self.consume_by_copy_or_move(binding.source.clone()))
                }
                BindingMode::ByRef(borrow_kind) => {
                    Rvalue::Ref(re_erased, borrow_kind, binding.source.clone())
                }
            };
            self.cfg.push_assign(block, source_info, &local, rvalue);
        }
    }

    /// Each binding (`ref mut var`/`ref var`/`mut var`/`var`, where the bound
    /// `var` has type `T` in the arm body) in a pattern maps to 2 locals. The
    /// first local is a binding for occurrences of `var` in the guard, which
    /// will have type `&T`. The second local is a binding for occurrences of
    /// `var` in the arm body, which will have type `T`.
    fn declare_binding(
        &mut self,
        source_info: SourceInfo,
        visibility_scope: SourceScope,
        mutability: Mutability,
        name: Name,
        mode: BindingMode,
        var_id: HirId,
        var_ty: Ty<'tcx>,
        user_ty: UserTypeProjections,
        has_guard: ArmHasGuard,
        opt_match_place: Option<(Option<Place<'tcx>>, Span)>,
        pat_span: Span,
    ) {
        debug!(
            "declare_binding(var_id={:?}, name={:?}, mode={:?}, var_ty={:?}, \
             visibility_scope={:?}, source_info={:?})",
            var_id, name, mode, var_ty, visibility_scope, source_info
        );

        let tcx = self.hir.tcx();
        let binding_mode = match mode {
            BindingMode::ByValue => ty::BindingMode::BindByValue(mutability.into()),
            BindingMode::ByRef(_) => ty::BindingMode::BindByReference(mutability.into()),
        };
        debug!("declare_binding: user_ty={:?}", user_ty);
        let local = LocalDecl::<'tcx> {
            mutability,
            ty: var_ty,
            user_ty,
            name: Some(name),
            source_info,
            visibility_scope,
            internal: false,
            is_block_tail: None,
            is_user_variable: Some(ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                binding_mode,
                // hypothetically, `visit_bindings` could try to unzip
                // an outermost hir::Ty as we descend, matching up
                // idents in pat; but complex w/ unclear UI payoff.
                // Instead, just abandon providing diagnostic info.
                opt_ty_info: None,
                opt_match_place,
                pat_span,
            }))),
        };
        let for_arm_body = self.local_decls.push(local);
        let locals = if has_guard.0 {
            let ref_for_guard = self.local_decls.push(LocalDecl::<'tcx> {
                // This variable isn't mutated but has a name, so has to be
                // immutable to avoid the unused mut lint.
                mutability: Mutability::Not,
                ty: tcx.mk_imm_ref(tcx.lifetimes.re_erased, var_ty),
                user_ty: UserTypeProjections::none(),
                name: Some(name),
                source_info,
                visibility_scope,
                internal: false,
                is_block_tail: None,
                is_user_variable: Some(ClearCrossCrate::Set(BindingForm::RefForGuard)),
            });
            LocalsForNode::ForGuard {
                ref_for_guard,
                for_arm_body,
            }
        } else {
            LocalsForNode::One(for_arm_body)
        };
        debug!("declare_binding: vars={:?}", locals);
        self.var_indices.insert(var_id, locals);
    }
}
