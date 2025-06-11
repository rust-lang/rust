//! Code related to match expressions. These are sufficiently complex to
//! warrant their own module and submodules. :) This main module includes the
//! high-level algorithm, the submodules contain the details.
//!
//! This also includes code for pattern bindings in `let` statements and
//! function parameters.

use std::assert_matches::assert_matches;
use std::borrow::Borrow;
use std::mem;
use std::sync::Arc;

use rustc_abi::VariantIdx;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::{BindingMode, ByRef, LetStmt, LocalSource, Node};
use rustc_middle::bug;
use rustc_middle::middle::region;
use rustc_middle::mir::{self, *};
use rustc_middle::thir::{self, *};
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation, Ty, ValTree, ValTreeKind};
use rustc_pattern_analysis::constructor::RangeEnd;
use rustc_pattern_analysis::rustc::{DeconstructedPat, RustcPatCtxt};
use rustc_span::{BytePos, Pos, Span, Symbol, sym};
use tracing::{debug, instrument};

use crate::builder::ForGuard::{self, OutsideGuard, RefWithinGuard};
use crate::builder::expr::as_place::PlaceBuilder;
use crate::builder::matches::user_ty::ProjectedUserTypesNode;
use crate::builder::scope::DropKind;
use crate::builder::{
    BlockAnd, BlockAndExtension, Builder, GuardFrame, GuardFrameLocal, LocalsForNode,
};

// helper functions, broken out by category:
mod match_pair;
mod test;
mod user_ty;
mod util;

/// Arguments to [`Builder::then_else_break_inner`] that are usually forwarded
/// to recursive invocations.
#[derive(Clone, Copy)]
struct ThenElseArgs {
    /// Used as the temp scope for lowering `expr`. If absent (for match guards),
    /// `self.local_scope()` is used.
    temp_scope_override: Option<region::Scope>,
    variable_source_info: SourceInfo,
    /// Determines how bindings should be handled when lowering `let` expressions.
    ///
    /// Forwarded to [`Builder::lower_let_expr`] when lowering [`ExprKind::Let`].
    declare_let_bindings: DeclareLetBindings,
}

/// Should lowering a `let` expression also declare its bindings?
///
/// Used by [`Builder::lower_let_expr`] when lowering [`ExprKind::Let`].
#[derive(Clone, Copy)]
pub(crate) enum DeclareLetBindings {
    /// Yes, declare `let` bindings as normal for `if` conditions.
    Yes,
    /// No, don't declare `let` bindings, because the caller declares them
    /// separately due to special requirements.
    ///
    /// Used for match guards and let-else.
    No,
    /// Let expressions are not permitted in this context, so it is a bug to
    /// try to lower one (e.g inside lazy-boolean-or or boolean-not).
    LetNotPermitted,
}

/// Used by [`Builder::bind_matched_candidate_for_arm_body`] to determine
/// whether or not to call [`Builder::storage_live_binding`] to emit
/// [`StatementKind::StorageLive`].
#[derive(Clone, Copy)]
pub(crate) enum EmitStorageLive {
    /// Yes, emit `StorageLive` as normal.
    Yes,
    /// No, don't emit `StorageLive`. The caller has taken responsibility for
    /// emitting `StorageLive` as appropriate.
    No,
}

/// Used by [`Builder::storage_live_binding`] and [`Builder::bind_matched_candidate_for_arm_body`]
/// to decide whether to schedule drops.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ScheduleDrops {
    /// Yes, the relevant functions should also schedule drops as appropriate.
    Yes,
    /// No, don't schedule drops. The caller has taken responsibility for any
    /// appropriate drops.
    No,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Lowers a condition in a way that ensures that variables bound in any let
    /// expressions are definitely initialized in the if body.
    ///
    /// If `declare_let_bindings` is false then variables created in `let`
    /// expressions will not be declared. This is for if let guards on arms with
    /// an or pattern, where the guard is lowered multiple times.
    pub(crate) fn then_else_break(
        &mut self,
        block: BasicBlock,
        expr_id: ExprId,
        temp_scope_override: Option<region::Scope>,
        variable_source_info: SourceInfo,
        declare_let_bindings: DeclareLetBindings,
    ) -> BlockAnd<()> {
        self.then_else_break_inner(
            block,
            expr_id,
            ThenElseArgs { temp_scope_override, variable_source_info, declare_let_bindings },
        )
    }

    fn then_else_break_inner(
        &mut self,
        block: BasicBlock, // Block that the condition and branch will be lowered into
        expr_id: ExprId,   // Condition expression to lower
        args: ThenElseArgs,
    ) -> BlockAnd<()> {
        let this = self;
        let expr = &this.thir[expr_id];
        let expr_span = expr.span;

        match expr.kind {
            ExprKind::LogicalOp { op: op @ LogicalOp::And, lhs, rhs } => {
                this.visit_coverage_branch_operation(op, expr_span);
                let lhs_then_block = this.then_else_break_inner(block, lhs, args).into_block();
                let rhs_then_block =
                    this.then_else_break_inner(lhs_then_block, rhs, args).into_block();
                rhs_then_block.unit()
            }
            ExprKind::LogicalOp { op: op @ LogicalOp::Or, lhs, rhs } => {
                this.visit_coverage_branch_operation(op, expr_span);
                let local_scope = this.local_scope();
                let (lhs_success_block, failure_block) =
                    this.in_if_then_scope(local_scope, expr_span, |this| {
                        this.then_else_break_inner(
                            block,
                            lhs,
                            ThenElseArgs {
                                declare_let_bindings: DeclareLetBindings::LetNotPermitted,
                                ..args
                            },
                        )
                    });
                let rhs_success_block = this
                    .then_else_break_inner(
                        failure_block,
                        rhs,
                        ThenElseArgs {
                            declare_let_bindings: DeclareLetBindings::LetNotPermitted,
                            ..args
                        },
                    )
                    .into_block();

                // Make the LHS and RHS success arms converge to a common block.
                // (We can't just make LHS goto RHS, because `rhs_success_block`
                // might contain statements that we don't want on the LHS path.)
                let success_block = this.cfg.start_new_block();
                this.cfg.goto(lhs_success_block, args.variable_source_info, success_block);
                this.cfg.goto(rhs_success_block, args.variable_source_info, success_block);
                success_block.unit()
            }
            ExprKind::Unary { op: UnOp::Not, arg } => {
                // Improve branch coverage instrumentation by noting conditions
                // nested within one or more `!` expressions.
                // (Skipped if branch coverage is not enabled.)
                if let Some(coverage_info) = this.coverage_info.as_mut() {
                    coverage_info.visit_unary_not(this.thir, expr_id);
                }

                let local_scope = this.local_scope();
                let (success_block, failure_block) =
                    this.in_if_then_scope(local_scope, expr_span, |this| {
                        // Help out coverage instrumentation by injecting a dummy statement with
                        // the original condition's span (including `!`). This fixes #115468.
                        if this.tcx.sess.instrument_coverage() {
                            this.cfg.push_coverage_span_marker(block, this.source_info(expr_span));
                        }
                        this.then_else_break_inner(
                            block,
                            arg,
                            ThenElseArgs {
                                declare_let_bindings: DeclareLetBindings::LetNotPermitted,
                                ..args
                            },
                        )
                    });
                this.break_for_else(success_block, args.variable_source_info);
                failure_block.unit()
            }
            ExprKind::Scope { region_scope, lint_level, value } => {
                let region_scope = (region_scope, this.source_info(expr_span));
                this.in_scope(region_scope, lint_level, |this| {
                    this.then_else_break_inner(block, value, args)
                })
            }
            ExprKind::Use { source } => this.then_else_break_inner(block, source, args),
            ExprKind::Let { expr, ref pat } => this.lower_let_expr(
                block,
                expr,
                pat,
                Some(args.variable_source_info.scope),
                args.variable_source_info.span,
                args.declare_let_bindings,
                EmitStorageLive::Yes,
            ),
            _ => {
                let mut block = block;
                let temp_scope = args.temp_scope_override.unwrap_or_else(|| this.local_scope());
                let mutability = Mutability::Mut;

                // Increment the decision depth, in case we encounter boolean expressions
                // further down.
                this.mcdc_increment_depth_if_enabled();
                let place = unpack!(
                    block = this.as_temp(
                        block,
                        TempLifetime {
                            temp_lifetime: Some(temp_scope),
                            backwards_incompatible: None
                        },
                        expr_id,
                        mutability
                    )
                );
                this.mcdc_decrement_depth_if_enabled();

                let operand = Operand::Move(Place::from(place));

                let then_block = this.cfg.start_new_block();
                let else_block = this.cfg.start_new_block();
                let term = TerminatorKind::if_(operand, then_block, else_block);

                // Record branch coverage info for this condition.
                // (Does nothing if branch coverage is not enabled.)
                this.visit_coverage_branch_condition(expr_id, then_block, else_block);

                let source_info = this.source_info(expr_span);
                this.cfg.terminate(block, source_info, term);
                this.break_for_else(else_block, source_info);

                then_block.unit()
            }
        }
    }

    /// Generates MIR for a `match` expression.
    ///
    /// The MIR that we generate for a match looks like this.
    ///
    /// ```text
    /// [ 0. Pre-match ]
    ///        |
    /// [ 1. Evaluate Scrutinee (expression being matched on) ]
    /// [ (PlaceMention of scrutinee) ]
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
    /// 1. Evaluate the scrutinee and add the PlaceMention of it ([Builder::lower_scrutinee]).
    /// 2. Create the decision tree ([Builder::lower_match_tree]).
    /// 3. Determine the fake borrows that are needed from the places that were
    ///    matched against and create the required temporaries for them
    ///    ([util::collect_fake_borrows]).
    /// 4. Create everything else: the guards and the arms ([Builder::lower_match_arms]).
    ///
    /// ## False edges
    ///
    /// We don't want to have the exact structure of the decision tree be visible through borrow
    /// checking. Specifically we want borrowck to think that:
    /// - at any point, any or none of the patterns and guards seen so far may have been tested;
    /// - after the match, any of the patterns may have matched.
    ///
    /// For example, all of these would fail to error if borrowck could see the real CFG (examples
    /// taken from `tests/ui/nll/match-cfg-fake-edges.rs`):
    /// ```ignore (too many errors, this is already in the test suite)
    /// let x = String::new();
    /// let _ = match true {
    ///     _ => {},
    ///     _ => drop(x),
    /// };
    /// // Borrowck must not know the second arm is never run.
    /// drop(x); //~ ERROR use of moved value
    ///
    /// let x;
    /// # let y = true;
    /// match y {
    ///     _ if { x = 2; true } => {},
    ///     // Borrowck must not know the guard is always run.
    ///     _ => drop(x), //~ ERROR used binding `x` is possibly-uninitialized
    /// };
    ///
    /// let x = String::new();
    /// # let y = true;
    /// match y {
    ///     false if { drop(x); true } => {},
    ///     // Borrowck must not know the guard is not run in the `true` case.
    ///     true => drop(x), //~ ERROR use of moved value: `x`
    ///     false => {},
    /// };
    ///
    /// # let mut y = (true, true);
    /// let r = &mut y.1;
    /// match y {
    ///     //~^ ERROR cannot use `y.1` because it was mutably borrowed
    ///     (false, true) => {}
    ///     // Borrowck must not know we don't test `y.1` when `y.0` is `true`.
    ///     (true, _) => drop(r),
    ///     (false, _) => {}
    /// };
    /// ```
    ///
    /// We add false edges to act as if we were naively matching each arm in order. What we need is
    /// a (fake) path from each candidate to the next, specifically from candidate C's pre-binding
    /// block to next candidate D's pre-binding block. For maximum precision (needed for deref
    /// patterns), we choose the earliest node on D's success path that doesn't also lead to C (to
    /// avoid loops).
    ///
    /// This turns out to be easy to compute: that block is the `start_block` of the first call to
    /// `match_candidates` where D is the first candidate in the list.
    ///
    /// For example:
    /// ```rust
    /// # let (x, y) = (true, true);
    /// match (x, y) {
    ///   (true, true) => 1,
    ///   (false, true) => 2,
    ///   (true, false) => 3,
    ///   _ => 4,
    /// }
    /// # ;
    /// ```
    /// In this example, the pre-binding block of arm 1 has a false edge to the block for result
    /// `false` of the first test on `x`. The other arms have false edges to the pre-binding blocks
    /// of the next arm.
    ///
    /// On top of this, we also add a false edge from the otherwise_block of each guard to the
    /// aforementioned start block of the next candidate, to ensure borrock doesn't rely on which
    /// guards may have run.
    #[instrument(level = "debug", skip(self, arms))]
    pub(crate) fn match_expr(
        &mut self,
        destination: Place<'tcx>,
        mut block: BasicBlock,
        scrutinee_id: ExprId,
        arms: &[ArmId],
        span: Span,
        scrutinee_span: Span,
    ) -> BlockAnd<()> {
        let scrutinee_place =
            unpack!(block = self.lower_scrutinee(block, scrutinee_id, scrutinee_span));

        let match_start_span = span.shrink_to_lo().to(scrutinee_span);
        let patterns = arms
            .iter()
            .map(|&arm| {
                let arm = &self.thir[arm];
                let has_match_guard =
                    if arm.guard.is_some() { HasMatchGuard::Yes } else { HasMatchGuard::No };
                (&*arm.pattern, has_match_guard)
            })
            .collect();
        let built_tree = self.lower_match_tree(
            block,
            scrutinee_span,
            &scrutinee_place,
            match_start_span,
            patterns,
            false,
        );

        self.lower_match_arms(
            destination,
            scrutinee_place,
            scrutinee_span,
            arms,
            built_tree,
            self.source_info(span),
        )
    }

    /// Evaluate the scrutinee and add the PlaceMention for it.
    fn lower_scrutinee(
        &mut self,
        mut block: BasicBlock,
        scrutinee_id: ExprId,
        scrutinee_span: Span,
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
        let scrutinee_place_builder = unpack!(block = self.as_place_builder(block, scrutinee_id));
        if let Some(scrutinee_place) = scrutinee_place_builder.try_to_place(self) {
            let source_info = self.source_info(scrutinee_span);
            self.cfg.push_place_mention(block, source_info, scrutinee_place);
        }

        block.and(scrutinee_place_builder)
    }

    /// Lower the bindings, guards and arm bodies of a `match` expression.
    ///
    /// The decision tree should have already been created
    /// (by [Builder::lower_match_tree]).
    ///
    /// `outer_source_info` is the SourceInfo for the whole match.
    pub(crate) fn lower_match_arms(
        &mut self,
        destination: Place<'tcx>,
        scrutinee_place_builder: PlaceBuilder<'tcx>,
        scrutinee_span: Span,
        arms: &[ArmId],
        built_match_tree: BuiltMatchTree<'tcx>,
        outer_source_info: SourceInfo,
    ) -> BlockAnd<()> {
        let arm_end_blocks: Vec<BasicBlock> = arms
            .iter()
            .map(|&arm| &self.thir[arm])
            .zip(built_match_tree.branches)
            .map(|(arm, branch)| {
                debug!("lowering arm {:?}\ncorresponding branch = {:?}", arm, branch);

                let arm_source_info = self.source_info(arm.span);
                let arm_scope = (arm.scope, arm_source_info);
                let match_scope = self.local_scope();
                self.in_scope(arm_scope, arm.lint_level, |this| {
                    let old_dedup_scope =
                        mem::replace(&mut this.fixed_temps_scope, Some(arm.scope));

                    // `try_to_place` may fail if it is unable to resolve the given
                    // `PlaceBuilder` inside a closure. In this case, we don't want to include
                    // a scrutinee place. `scrutinee_place_builder` will fail to be resolved
                    // if the only match arm is a wildcard (`_`).
                    // Example:
                    // ```
                    // let foo = (0, 1);
                    // let c = || {
                    //    match foo { _ => () };
                    // };
                    // ```
                    let scrutinee_place = scrutinee_place_builder.try_to_place(this);
                    let opt_scrutinee_place =
                        scrutinee_place.as_ref().map(|place| (Some(place), scrutinee_span));
                    let scope = this.declare_bindings(
                        None,
                        arm.span,
                        &arm.pattern,
                        arm.guard,
                        opt_scrutinee_place,
                    );

                    let arm_block = this.bind_pattern(
                        outer_source_info,
                        branch,
                        &built_match_tree.fake_borrow_temps,
                        scrutinee_span,
                        Some((arm, match_scope)),
                        EmitStorageLive::Yes,
                    );

                    this.fixed_temps_scope = old_dedup_scope;

                    if let Some(source_scope) = scope {
                        this.source_scope = source_scope;
                    }

                    this.expr_into_dest(destination, arm_block, arm.body)
                })
                .into_block()
            })
            .collect();

        // all the arm blocks will rejoin here
        let end_block = self.cfg.start_new_block();

        let end_brace = self.source_info(
            outer_source_info.span.with_lo(outer_source_info.span.hi() - BytePos::from_usize(1)),
        );
        for arm_block in arm_end_blocks {
            let block = &self.cfg.basic_blocks[arm_block];
            let last_location = block.statements.last().map(|s| s.source_info);

            self.cfg.goto(arm_block, last_location.unwrap_or(end_brace), end_block);
        }

        self.source_scope = outer_source_info.scope;

        end_block.unit()
    }

    /// For a top-level `match` arm or a `let` binding, binds the variables and
    /// ascribes types, and also checks the match arm guard (if present).
    ///
    /// `arm_scope` should be `Some` if and only if this is called for a
    /// `match` arm.
    ///
    /// In the presence of or-patterns, a match arm might have multiple
    /// sub-branches representing different ways to match, with each sub-branch
    /// requiring its own bindings and its own copy of the guard. This method
    /// handles those sub-branches individually, and then has them jump together
    /// to a common block.
    ///
    /// Returns a single block that the match arm can be lowered into.
    /// (For `let` bindings, this is the code that can use the bindings.)
    fn bind_pattern(
        &mut self,
        outer_source_info: SourceInfo,
        branch: MatchTreeBranch<'tcx>,
        fake_borrow_temps: &[(Place<'tcx>, Local, FakeBorrowKind)],
        scrutinee_span: Span,
        arm_match_scope: Option<(&Arm<'tcx>, region::Scope)>,
        emit_storage_live: EmitStorageLive,
    ) -> BasicBlock {
        if branch.sub_branches.len() == 1 {
            let [sub_branch] = branch.sub_branches.try_into().unwrap();
            // Avoid generating another `BasicBlock` when we only have one sub branch.
            self.bind_and_guard_matched_candidate(
                sub_branch,
                fake_borrow_temps,
                scrutinee_span,
                arm_match_scope,
                ScheduleDrops::Yes,
                emit_storage_live,
            )
        } else {
            // It's helpful to avoid scheduling drops multiple times to save
            // drop elaboration from having to clean up the extra drops.
            //
            // If we are in a `let` then we only schedule drops for the first
            // candidate.
            //
            // If we're in a `match` arm then we could have a case like so:
            //
            // Ok(x) | Err(x) if return => { /* ... */ }
            //
            // In this case we don't want a drop of `x` scheduled when we
            // return: it isn't bound by move until right before enter the arm.
            // To handle this we instead unschedule it's drop after each time
            // we lower the guard.
            let target_block = self.cfg.start_new_block();
            let mut schedule_drops = ScheduleDrops::Yes;
            let arm = arm_match_scope.unzip().0;
            // We keep a stack of all of the bindings and type ascriptions
            // from the parent candidates that we visit, that also need to
            // be bound for each candidate.
            for sub_branch in branch.sub_branches {
                if let Some(arm) = arm {
                    self.clear_top_scope(arm.scope);
                }
                let binding_end = self.bind_and_guard_matched_candidate(
                    sub_branch,
                    fake_borrow_temps,
                    scrutinee_span,
                    arm_match_scope,
                    schedule_drops,
                    emit_storage_live,
                );
                if arm.is_none() {
                    schedule_drops = ScheduleDrops::No;
                }
                self.cfg.goto(binding_end, outer_source_info, target_block);
            }

            target_block
        }
    }

    pub(super) fn expr_into_pattern(
        &mut self,
        mut block: BasicBlock,
        irrefutable_pat: &Pat<'tcx>,
        initializer_id: ExprId,
    ) -> BlockAnd<()> {
        match irrefutable_pat.kind {
            // Optimize the case of `let x = ...` to write directly into `x`
            PatKind::Binding { mode: BindingMode(ByRef::No, _), var, subpattern: None, .. } => {
                let place = self.storage_live_binding(
                    block,
                    var,
                    irrefutable_pat.span,
                    OutsideGuard,
                    ScheduleDrops::Yes,
                );
                block = self.expr_into_dest(place, block, initializer_id).into_block();

                // Inject a fake read, see comments on `FakeReadCause::ForLet`.
                let source_info = self.source_info(irrefutable_pat.span);
                self.cfg.push_fake_read(block, source_info, FakeReadCause::ForLet(None), place);

                self.schedule_drop_for_binding(var, irrefutable_pat.span, OutsideGuard);
                block.unit()
            }

            // Optimize the case of `let x: T = ...` to write directly
            // into `x` and then require that `T == typeof(x)`.
            PatKind::AscribeUserType {
                ref subpattern,
                ascription: thir::Ascription { ref annotation, variance: _ },
            } if let PatKind::Binding {
                mode: BindingMode(ByRef::No, _),
                var,
                subpattern: None,
                ..
            } = subpattern.kind =>
            {
                let place = self.storage_live_binding(
                    block,
                    var,
                    irrefutable_pat.span,
                    OutsideGuard,
                    ScheduleDrops::Yes,
                );
                block = self.expr_into_dest(place, block, initializer_id).into_block();

                // Inject a fake read, see comments on `FakeReadCause::ForLet`.
                let pattern_source_info = self.source_info(irrefutable_pat.span);
                let cause_let = FakeReadCause::ForLet(None);
                self.cfg.push_fake_read(block, pattern_source_info, cause_let, place);

                let ty_source_info = self.source_info(annotation.span);

                let base = self.canonical_user_type_annotations.push(annotation.clone());
                self.cfg.push(
                    block,
                    Statement {
                        source_info: ty_source_info,
                        kind: StatementKind::AscribeUserType(
                            Box::new((place, UserTypeProjection { base, projs: Vec::new() })),
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
                            ty::Invariant,
                        ),
                    },
                );

                self.schedule_drop_for_binding(var, irrefutable_pat.span, OutsideGuard);
                block.unit()
            }

            _ => {
                let initializer = &self.thir[initializer_id];
                let place_builder =
                    unpack!(block = self.lower_scrutinee(block, initializer_id, initializer.span));
                self.place_into_pattern(block, irrefutable_pat, place_builder, true)
            }
        }
    }

    pub(crate) fn place_into_pattern(
        &mut self,
        block: BasicBlock,
        irrefutable_pat: &Pat<'tcx>,
        initializer: PlaceBuilder<'tcx>,
        set_match_place: bool,
    ) -> BlockAnd<()> {
        let built_tree = self.lower_match_tree(
            block,
            irrefutable_pat.span,
            &initializer,
            irrefutable_pat.span,
            vec![(irrefutable_pat, HasMatchGuard::No)],
            false,
        );
        let [branch] = built_tree.branches.try_into().unwrap();

        // For matches and function arguments, the place that is being matched
        // can be set when creating the variables. But the place for
        // let PATTERN = ... might not even exist until we do the assignment.
        // so we set it here instead.
        if set_match_place {
            // `try_to_place` may fail if it is unable to resolve the given `PlaceBuilder` inside a
            // closure. In this case, we don't want to include a scrutinee place.
            // `scrutinee_place_builder` will fail for destructured assignments. This is because a
            // closure only captures the precise places that it will read and as a result a closure
            // may not capture the entire tuple/struct and rather have individual places that will
            // be read in the final MIR.
            // Example:
            // ```
            // let foo = (0, 1);
            // let c = || {
            //    let (v1, v2) = foo;
            // };
            // ```
            if let Some(place) = initializer.try_to_place(self) {
                // Because or-alternatives bind the same variables, we only explore the first one.
                let first_sub_branch = branch.sub_branches.first().unwrap();
                for binding in &first_sub_branch.bindings {
                    let local = self.var_local_id(binding.var_id, OutsideGuard);
                    if let LocalInfo::User(BindingForm::Var(VarBindingForm {
                        opt_match_place: Some((ref mut match_place, _)),
                        ..
                    })) = **self.local_decls[local].local_info.as_mut().unwrap_crate_local()
                    {
                        *match_place = Some(place);
                    } else {
                        bug!("Let binding to non-user variable.")
                    };
                }
            }
        }

        self.bind_pattern(
            self.source_info(irrefutable_pat.span),
            branch,
            &[],
            irrefutable_pat.span,
            None,
            EmitStorageLive::Yes,
        )
        .unit()
    }

    /// Declares the bindings of the given patterns and returns the visibility
    /// scope for the bindings in these patterns, if such a scope had to be
    /// created. NOTE: Declaring the bindings should always be done in their
    /// drop scope.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn declare_bindings(
        &mut self,
        mut visibility_scope: Option<SourceScope>,
        scope_span: Span,
        pattern: &Pat<'tcx>,
        guard: Option<ExprId>,
        opt_match_place: Option<(Option<&Place<'tcx>>, Span)>,
    ) -> Option<SourceScope> {
        self.visit_primary_bindings_special(
            pattern,
            &ProjectedUserTypesNode::None,
            &mut |this, name, mode, var, span, ty, user_tys| {
                let vis_scope = *visibility_scope
                    .get_or_insert_with(|| this.new_source_scope(scope_span, LintLevel::Inherited));
                let source_info = SourceInfo { span, scope: this.source_scope };
                let user_tys = user_tys.build_user_type_projections();

                this.declare_binding(
                    source_info,
                    vis_scope,
                    name,
                    mode,
                    var,
                    ty,
                    user_tys,
                    ArmHasGuard(guard.is_some()),
                    opt_match_place.map(|(x, y)| (x.cloned(), y)),
                    pattern.span,
                );
            },
        );
        if let Some(guard_expr) = guard {
            self.declare_guard_bindings(guard_expr, scope_span, visibility_scope);
        }
        visibility_scope
    }

    /// Declare bindings in a guard. This has to be done when declaring bindings
    /// for an arm to ensure that or patterns only have one version of each
    /// variable.
    pub(crate) fn declare_guard_bindings(
        &mut self,
        guard_expr: ExprId,
        scope_span: Span,
        visibility_scope: Option<SourceScope>,
    ) {
        match self.thir.exprs[guard_expr].kind {
            ExprKind::Let { expr: _, pat: ref guard_pat } => {
                // FIXME: pass a proper `opt_match_place`
                self.declare_bindings(visibility_scope, scope_span, guard_pat, None, None);
            }
            ExprKind::Scope { value, .. } => {
                self.declare_guard_bindings(value, scope_span, visibility_scope);
            }
            ExprKind::Use { source } => {
                self.declare_guard_bindings(source, scope_span, visibility_scope);
            }
            ExprKind::LogicalOp { op: LogicalOp::And, lhs, rhs } => {
                self.declare_guard_bindings(lhs, scope_span, visibility_scope);
                self.declare_guard_bindings(rhs, scope_span, visibility_scope);
            }
            _ => {}
        }
    }

    /// Emits a [`StatementKind::StorageLive`] for the given var, and also
    /// schedules a drop if requested (and possible).
    pub(crate) fn storage_live_binding(
        &mut self,
        block: BasicBlock,
        var: LocalVarId,
        span: Span,
        for_guard: ForGuard,
        schedule_drop: ScheduleDrops,
    ) -> Place<'tcx> {
        let local_id = self.var_local_id(var, for_guard);
        let source_info = self.source_info(span);
        self.cfg.push(block, Statement { source_info, kind: StatementKind::StorageLive(local_id) });
        // Although there is almost always scope for given variable in corner cases
        // like #92893 we might get variable with no scope.
        if let Some(region_scope) = self.region_scope_tree.var_scope(var.0.local_id)
            && matches!(schedule_drop, ScheduleDrops::Yes)
        {
            self.schedule_drop(span, region_scope, local_id, DropKind::Storage);
        }
        Place::from(local_id)
    }

    pub(crate) fn schedule_drop_for_binding(
        &mut self,
        var: LocalVarId,
        span: Span,
        for_guard: ForGuard,
    ) {
        let local_id = self.var_local_id(var, for_guard);
        if let Some(region_scope) = self.region_scope_tree.var_scope(var.0.local_id) {
            self.schedule_drop(span, region_scope, local_id, DropKind::Value);
        }
    }

    /// Visits all of the "primary" bindings in a pattern, i.e. the leftmost
    /// occurrence of each variable bound by the pattern.
    /// See [`PatKind::Binding::is_primary`] for more context.
    ///
    /// This variant provides only the limited subset of binding data needed
    /// by its callers, and should be a "pure" visit without side-effects.
    pub(super) fn visit_primary_bindings(
        &mut self,
        pattern: &Pat<'tcx>,
        f: &mut impl FnMut(&mut Self, LocalVarId, Span),
    ) {
        pattern.walk_always(|pat| {
            if let PatKind::Binding { var, is_primary: true, .. } = pat.kind {
                f(self, var, pat.span);
            }
        })
    }

    /// Visits all of the "primary" bindings in a pattern, while preparing
    /// additional user-type-annotation data needed by `declare_bindings`.
    ///
    /// This also has the side-effect of pushing all user type annotations
    /// onto `canonical_user_type_annotations`, so that they end up in MIR
    /// even if they aren't associated with any bindings.
    #[instrument(level = "debug", skip(self, f))]
    fn visit_primary_bindings_special(
        &mut self,
        pattern: &Pat<'tcx>,
        user_tys: &ProjectedUserTypesNode<'_>,
        f: &mut impl FnMut(
            &mut Self,
            Symbol,
            BindingMode,
            LocalVarId,
            Span,
            Ty<'tcx>,
            &ProjectedUserTypesNode<'_>,
        ),
    ) {
        // Avoid having to write the full method name at each recursive call.
        let visit_subpat = |this: &mut Self, subpat, user_tys: &_, f: &mut _| {
            this.visit_primary_bindings_special(subpat, user_tys, f)
        };

        match pattern.kind {
            PatKind::Binding { name, mode, var, ty, ref subpattern, is_primary, .. } => {
                if is_primary {
                    f(self, name, mode, var, pattern.span, ty, user_tys);
                }
                if let Some(subpattern) = subpattern.as_ref() {
                    visit_subpat(self, subpattern, user_tys, f);
                }
            }

            PatKind::Array { ref prefix, ref slice, ref suffix }
            | PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                let from = u64::try_from(prefix.len()).unwrap();
                let to = u64::try_from(suffix.len()).unwrap();
                for subpattern in prefix.iter() {
                    visit_subpat(self, subpattern, &user_tys.index(), f);
                }
                if let Some(subpattern) = slice {
                    visit_subpat(self, subpattern, &user_tys.subslice(from, to), f);
                }
                for subpattern in suffix.iter() {
                    visit_subpat(self, subpattern, &user_tys.index(), f);
                }
            }

            PatKind::Constant { .. }
            | PatKind::Range { .. }
            | PatKind::Missing
            | PatKind::Wild
            | PatKind::Never
            | PatKind::Error(_) => {}

            PatKind::Deref { ref subpattern } => {
                visit_subpat(self, subpattern, &user_tys.deref(), f);
            }

            PatKind::DerefPattern { ref subpattern, .. } => {
                visit_subpat(self, subpattern, &ProjectedUserTypesNode::None, f);
            }

            PatKind::AscribeUserType {
                ref subpattern,
                ascription: thir::Ascription { ref annotation, variance: _ },
            } => {
                // This corresponds to something like
                //
                // ```
                // let A::<'a>(_): A<'static> = ...;
                // ```
                //
                // Note that the variance doesn't apply here, as we are tracking the effect
                // of `user_ty` on any bindings contained with subpattern.

                // Caution: Pushing this user type here is load-bearing even for
                // patterns containing no bindings, to ensure that the type ends
                // up represented in MIR _somewhere_.
                let base_user_ty = self.canonical_user_type_annotations.push(annotation.clone());
                let subpattern_user_tys = user_tys.push_user_type(base_user_ty);
                visit_subpat(self, subpattern, &subpattern_user_tys, f)
            }

            PatKind::ExpandedConstant { ref subpattern, .. } => {
                visit_subpat(self, subpattern, user_tys, f)
            }

            PatKind::Leaf { ref subpatterns } => {
                for subpattern in subpatterns {
                    let subpattern_user_tys = user_tys.leaf(subpattern.field);
                    debug!("visit_primary_bindings: subpattern_user_tys={subpattern_user_tys:?}");
                    visit_subpat(self, &subpattern.pattern, &subpattern_user_tys, f);
                }
            }

            PatKind::Variant { adt_def, args: _, variant_index, ref subpatterns } => {
                for subpattern in subpatterns {
                    let subpattern_user_tys =
                        user_tys.variant(adt_def, variant_index, subpattern.field);
                    visit_subpat(self, &subpattern.pattern, &subpattern_user_tys, f);
                }
            }
            PatKind::Or { ref pats } => {
                // In cases where we recover from errors the primary bindings
                // may not all be in the leftmost subpattern. For example in
                // `let (x | y) = ...`, the primary binding of `y` occurs in
                // the right subpattern
                for subpattern in pats.iter() {
                    visit_subpat(self, subpattern, user_tys, f);
                }
            }
        }
    }
}

/// Data extracted from a pattern that doesn't affect which branch is taken. Collected during
/// pattern simplification and not mutated later.
#[derive(Debug, Clone)]
struct PatternExtraData<'tcx> {
    /// [`Span`] of the original pattern.
    span: Span,

    /// Bindings that must be established.
    bindings: Vec<Binding<'tcx>>,

    /// Types that must be asserted.
    ascriptions: Vec<Ascription<'tcx>>,

    /// Whether this corresponds to a never pattern.
    is_never: bool,
}

impl<'tcx> PatternExtraData<'tcx> {
    fn is_empty(&self) -> bool {
        self.bindings.is_empty() && self.ascriptions.is_empty()
    }
}

/// A pattern in a form suitable for lowering the match tree, with all irrefutable
/// patterns simplified away.
///
/// Here, "flat" indicates that irrefutable nodes in the pattern tree have been
/// recursively replaced with their refutable subpatterns. They are not
/// necessarily flat in an absolute sense.
///
/// Will typically be incorporated into a [`Candidate`].
#[derive(Debug, Clone)]
struct FlatPat<'tcx> {
    /// To match the pattern, all of these must be satisfied...
    match_pairs: Vec<MatchPairTree<'tcx>>,

    extra_data: PatternExtraData<'tcx>,
}

impl<'tcx> FlatPat<'tcx> {
    /// Creates a `FlatPat` containing a simplified [`MatchPairTree`] list/forest
    /// for the given pattern.
    fn new(place: PlaceBuilder<'tcx>, pattern: &Pat<'tcx>, cx: &mut Builder<'_, 'tcx>) -> Self {
        // Recursively build a tree of match pairs for the given pattern.
        let mut match_pairs = vec![];
        let mut extra_data = PatternExtraData {
            span: pattern.span,
            bindings: Vec::new(),
            ascriptions: Vec::new(),
            is_never: pattern.is_never_pattern(),
        };
        MatchPairTree::for_pattern(place, pattern, cx, &mut match_pairs, &mut extra_data);

        Self { match_pairs, extra_data }
    }
}

/// Candidates are a generalization of (a) top-level match arms, and
/// (b) sub-branches of or-patterns, allowing the match-lowering process to handle
/// them both in a mostly-uniform way. For example, the list of candidates passed
/// to [`Builder::match_candidates`] will often contain a mixture of top-level
/// candidates and or-pattern subcandidates.
///
/// At the start of match lowering, there is one candidate for each match arm.
/// During match lowering, arms with or-patterns will be expanded into a tree
/// of candidates, where each "leaf" candidate represents one of the ways for
/// the arm pattern to successfully match.
#[derive(Debug)]
struct Candidate<'tcx> {
    /// For the candidate to match, all of these must be satisfied...
    ///
    /// ---
    /// Initially contains a list of match pairs created by [`FlatPat`], but is
    /// subsequently mutated (in a queue-like way) while lowering the match tree.
    /// When this list becomes empty, the candidate is fully matched and becomes
    /// a leaf (see [`Builder::select_matched_candidate`]).
    ///
    /// Key mutations include:
    ///
    /// - When a match pair is fully satisfied by a test, it is removed from the
    ///   list, and its subpairs are added instead (see [`Builder::sort_candidate`]).
    /// - During or-pattern expansion, any leading or-pattern is removed, and is
    ///   converted into subcandidates (see [`Builder::expand_and_match_or_candidates`]).
    /// - After a candidate's subcandidates have been lowered, a copy of any remaining
    ///   or-patterns is added to each leaf subcandidate
    ///   (see [`Builder::test_remaining_match_pairs_after_or`]).
    ///
    /// Invariants:
    /// - All or-patterns ([`TestCase::Or`]) have been sorted to the end.
    match_pairs: Vec<MatchPairTree<'tcx>>,

    /// ...and if this is non-empty, one of these subcandidates also has to match...
    ///
    /// ---
    /// Initially a candidate has no subcandidates; they are added (and then immediately
    /// lowered) during or-pattern expansion. Their main function is to serve as _output_
    /// of match tree lowering, allowing later steps to see the leaf candidates that
    /// represent a match of the entire match arm.
    ///
    /// A candidate no subcandidates is either incomplete (if it has match pairs left),
    /// or is a leaf in the match tree. A candidate with one or more subcandidates is
    /// an internal node in the match tree.
    ///
    /// Invariant: at the end of match tree lowering, this must not contain an
    /// `is_never` candidate, because that would break binding consistency.
    /// - See [`Builder::remove_never_subcandidates`].
    subcandidates: Vec<Candidate<'tcx>>,

    /// ...and if there is a guard it must be evaluated; if it's `false` then branch to `otherwise_block`.
    ///
    /// ---
    /// For subcandidates, this is copied from the parent candidate, so it indicates
    /// whether the enclosing match arm has a guard.
    has_guard: bool,

    /// Holds extra pattern data that was prepared by [`FlatPat`], including bindings and
    /// ascriptions that must be established if this candidate succeeds.
    extra_data: PatternExtraData<'tcx>,

    /// When setting `self.subcandidates`, we store here the span of the or-pattern they came from.
    ///
    /// ---
    /// Invariant: it is `None` iff `subcandidates.is_empty()`.
    /// - FIXME: We sometimes don't unset this when clearing `subcandidates`.
    or_span: Option<Span>,

    /// The block before the `bindings` have been established.
    ///
    /// After the match tree has been lowered, [`Builder::lower_match_arms`]
    /// will use this as the start point for lowering bindings and guards, and
    /// then jump to a shared block containing the arm body.
    pre_binding_block: Option<BasicBlock>,

    /// The block to branch to if the guard or a nested candidate fails to match.
    otherwise_block: Option<BasicBlock>,

    /// The earliest block that has only candidates >= this one as descendents. Used for false
    /// edges, see the doc for [`Builder::match_expr`].
    false_edge_start_block: Option<BasicBlock>,
}

impl<'tcx> Candidate<'tcx> {
    fn new(
        place: PlaceBuilder<'tcx>,
        pattern: &Pat<'tcx>,
        has_guard: HasMatchGuard,
        cx: &mut Builder<'_, 'tcx>,
    ) -> Self {
        // Use `FlatPat` to build simplified match pairs, then immediately
        // incorporate them into a new candidate.
        Self::from_flat_pat(
            FlatPat::new(place, pattern, cx),
            matches!(has_guard, HasMatchGuard::Yes),
        )
    }

    /// Incorporates an already-simplified [`FlatPat`] into a new candidate.
    fn from_flat_pat(flat_pat: FlatPat<'tcx>, has_guard: bool) -> Self {
        let mut this = Candidate {
            match_pairs: flat_pat.match_pairs,
            extra_data: flat_pat.extra_data,
            has_guard,
            subcandidates: Vec::new(),
            or_span: None,
            otherwise_block: None,
            pre_binding_block: None,
            false_edge_start_block: None,
        };
        this.sort_match_pairs();
        this
    }

    /// Restores the invariant that or-patterns must be sorted to the end.
    fn sort_match_pairs(&mut self) {
        self.match_pairs.sort_by_key(|pair| matches!(pair.test_case, TestCase::Or { .. }));
    }

    /// Returns whether the first match pair of this candidate is an or-pattern.
    fn starts_with_or_pattern(&self) -> bool {
        matches!(&*self.match_pairs, [MatchPairTree { test_case: TestCase::Or { .. }, .. }, ..])
    }

    /// Visit the leaf candidates (those with no subcandidates) contained in
    /// this candidate.
    fn visit_leaves<'a>(&'a mut self, mut visit_leaf: impl FnMut(&'a mut Self)) {
        traverse_candidate(
            self,
            &mut (),
            &mut move |c, _| visit_leaf(c),
            move |c, _| c.subcandidates.iter_mut(),
            |_| {},
        );
    }

    /// Visit the leaf candidates in reverse order.
    fn visit_leaves_rev<'a>(&'a mut self, mut visit_leaf: impl FnMut(&'a mut Self)) {
        traverse_candidate(
            self,
            &mut (),
            &mut move |c, _| visit_leaf(c),
            move |c, _| c.subcandidates.iter_mut().rev(),
            |_| {},
        );
    }
}

/// A depth-first traversal of the `Candidate` and all of its recursive
/// subcandidates.
///
/// This signature is very generic, to support traversing candidate trees by
/// reference or by value, and to allow a mutable "context" to be shared by the
/// traversal callbacks. Most traversals can use the simpler
/// [`Candidate::visit_leaves`] wrapper instead.
fn traverse_candidate<'tcx, C, T, I>(
    candidate: C,
    context: &mut T,
    // Called when visiting a "leaf" candidate (with no subcandidates).
    visit_leaf: &mut impl FnMut(C, &mut T),
    // Called when visiting a "node" candidate (with one or more subcandidates).
    // Returns an iterator over the candidate's children (by value or reference).
    // Can perform setup before visiting the node's children.
    get_children: impl Copy + Fn(C, &mut T) -> I,
    // Called after visiting a "node" candidate's children.
    complete_children: impl Copy + Fn(&mut T),
) where
    C: Borrow<Candidate<'tcx>>, // Typically `Candidate` or `&mut Candidate`
    I: Iterator<Item = C>,
{
    if candidate.borrow().subcandidates.is_empty() {
        visit_leaf(candidate, context)
    } else {
        for child in get_children(candidate, context) {
            traverse_candidate(child, context, visit_leaf, get_children, complete_children);
        }
        complete_children(context)
    }
}

#[derive(Clone, Debug)]
struct Binding<'tcx> {
    span: Span,
    source: Place<'tcx>,
    var_id: LocalVarId,
    binding_mode: BindingMode,
}

/// Indicates that the type of `source` must be a subtype of the
/// user-given type `user_ty`; this is basically a no-op but can
/// influence region inference.
#[derive(Clone, Debug)]
struct Ascription<'tcx> {
    source: Place<'tcx>,
    annotation: CanonicalUserTypeAnnotation<'tcx>,
    variance: ty::Variance,
}

/// Partial summary of a [`thir::Pat`], indicating what sort of test should be
/// performed to match/reject the pattern, and what the desired test outcome is.
/// This avoids having to perform a full match on [`thir::PatKind`] in some places,
/// and helps [`TestKind::Switch`] and [`TestKind::SwitchInt`] know what target
/// values to use.
///
/// Created by [`MatchPairTree::for_pattern`], and then inspected primarily by:
/// - [`Builder::pick_test_for_match_pair`] (to choose a test)
/// - [`Builder::sort_candidate`] (to see how the test interacts with a match pair)
///
/// Note that or-patterns are not tested directly like the other variants.
/// Instead they participate in or-pattern expansion, where they are transformed into
/// subcandidates. See [`Builder::expand_and_match_or_candidates`].
#[derive(Debug, Clone)]
enum TestCase<'tcx> {
    Variant { adt_def: ty::AdtDef<'tcx>, variant_index: VariantIdx },
    Constant { value: mir::Const<'tcx> },
    Range(Arc<PatRange<'tcx>>),
    Slice { len: usize, variable_length: bool },
    Deref { temp: Place<'tcx>, mutability: Mutability },
    Never,
    Or { pats: Box<[FlatPat<'tcx>]> },
}

impl<'tcx> TestCase<'tcx> {
    fn as_range(&self) -> Option<&PatRange<'tcx>> {
        if let Self::Range(v) = self { Some(v.as_ref()) } else { None }
    }
}

/// Node in a tree of "match pairs", where each pair consists of a place to be
/// tested, and a test to perform on that place.
///
/// Each node also has a list of subpairs (possibly empty) that must also match,
/// and a reference to the THIR pattern it represents.
#[derive(Debug, Clone)]
pub(crate) struct MatchPairTree<'tcx> {
    /// This place...
    ///
    /// ---
    /// This can be `None` if it referred to a non-captured place in a closure.
    ///
    /// Invariant: Can only be `None` when `test_case` is `Or`.
    /// Therefore this must be `Some(_)` after or-pattern expansion.
    place: Option<Place<'tcx>>,

    /// ... must pass this test...
    test_case: TestCase<'tcx>,

    /// ... and these subpairs must match.
    ///
    /// ---
    /// Subpairs typically represent tests that can only be performed after their
    /// parent has succeeded. For example, the pattern `Some(3)` might have an
    /// outer match pair that tests for the variant `Some`, and then a subpair
    /// that tests its field for the value `3`.
    subpairs: Vec<Self>,

    /// Type field of the pattern this node was created from.
    pattern_ty: Ty<'tcx>,
    /// Span field of the pattern this node was created from.
    pattern_span: Span,
}

/// See [`Test`] for more.
#[derive(Clone, Debug, PartialEq)]
enum TestKind<'tcx> {
    /// Test what enum variant a value is.
    ///
    /// The subset of expected variants is not stored here; instead they are
    /// extracted from the [`TestCase`]s of the candidates participating in the
    /// test.
    Switch {
        /// The enum type being tested.
        adt_def: ty::AdtDef<'tcx>,
    },

    /// Test what value an integer or `char` has.
    ///
    /// The test's target values are not stored here; instead they are extracted
    /// from the [`TestCase`]s of the candidates participating in the test.
    SwitchInt,

    /// Test whether a `bool` is `true` or `false`.
    If,

    /// Test for equality with value, possibly after an unsizing coercion to
    /// `ty`,
    Eq {
        value: Const<'tcx>,
        // Integer types are handled by `SwitchInt`, and constants with ADT
        // types and `&[T]` types are converted back into patterns, so this can
        // only be `&str`, `f32` or `f64`.
        ty: Ty<'tcx>,
    },

    /// Test whether the value falls within an inclusive or exclusive range.
    Range(Arc<PatRange<'tcx>>),

    /// Test that the length of the slice is `== len` or `>= len`.
    Len { len: u64, op: BinOp },

    /// Call `Deref::deref[_mut]` on the value.
    Deref {
        /// Temporary to store the result of `deref()`/`deref_mut()`.
        temp: Place<'tcx>,
        mutability: Mutability,
    },

    /// Assert unreachability of never patterns.
    Never,
}

/// A test to perform to determine which [`Candidate`] matches a value.
///
/// [`Test`] is just the test to perform; it does not include the value
/// to be tested.
#[derive(Debug)]
pub(crate) struct Test<'tcx> {
    span: Span,
    kind: TestKind<'tcx>,
}

/// The branch to be taken after a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TestBranch<'tcx> {
    /// Success branch, used for tests with two possible outcomes.
    Success,
    /// Branch corresponding to this constant.
    Constant(Const<'tcx>, u128),
    /// Branch corresponding to this variant.
    Variant(VariantIdx),
    /// Failure branch for tests with two possible outcomes, and "otherwise" branch for other tests.
    Failure,
}

impl<'tcx> TestBranch<'tcx> {
    fn as_constant(&self) -> Option<&Const<'tcx>> {
        if let Self::Constant(v, _) = self { Some(v) } else { None }
    }
}

/// `ArmHasGuard` is a wrapper around a boolean flag. It indicates whether
/// a match arm has a guard expression attached to it.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ArmHasGuard(pub(crate) bool);

///////////////////////////////////////////////////////////////////////////
// Main matching algorithm

/// A sub-branch in the output of match lowering. Match lowering has generated MIR code that will
/// branch to `success_block` when the matched value matches the corresponding pattern. If there is
/// a guard, its failure must continue to `otherwise_block`, which will resume testing patterns.
#[derive(Debug, Clone)]
struct MatchTreeSubBranch<'tcx> {
    span: Span,
    /// The block that is branched to if the corresponding subpattern matches.
    success_block: BasicBlock,
    /// The block to branch to if this arm had a guard and the guard fails.
    otherwise_block: BasicBlock,
    /// The bindings to set up in this sub-branch.
    bindings: Vec<Binding<'tcx>>,
    /// The ascriptions to set up in this sub-branch.
    ascriptions: Vec<Ascription<'tcx>>,
    /// Whether the sub-branch corresponds to a never pattern.
    is_never: bool,
}

/// A branch in the output of match lowering.
#[derive(Debug, Clone)]
struct MatchTreeBranch<'tcx> {
    sub_branches: Vec<MatchTreeSubBranch<'tcx>>,
}

/// The result of generating MIR for a pattern-matching expression. Each input branch/arm/pattern
/// gives rise to an output `MatchTreeBranch`. If one of the patterns matches, we branch to the
/// corresponding `success_block`. If none of the patterns matches, we branch to `otherwise_block`.
///
/// Each branch is made of one of more sub-branches, corresponding to or-patterns. E.g.
/// ```ignore(illustrative)
/// match foo {
///     (x, false) | (false, x) => {}
///     (true, true) => {}
/// }
/// ```
/// Here the first arm gives the first `MatchTreeBranch`, which has two sub-branches, one for each
/// alternative of the or-pattern. They are kept separate because each needs to bind `x` to a
/// different place.
#[derive(Debug, Clone)]
pub(crate) struct BuiltMatchTree<'tcx> {
    branches: Vec<MatchTreeBranch<'tcx>>,
    otherwise_block: BasicBlock,
    /// If any of the branches had a guard, we collect here the places and locals to fakely borrow
    /// to ensure match guards can't modify the values as we match them. For more details, see
    /// [`util::collect_fake_borrows`].
    fake_borrow_temps: Vec<(Place<'tcx>, Local, FakeBorrowKind)>,
}

impl<'tcx> MatchTreeSubBranch<'tcx> {
    fn from_sub_candidate(
        candidate: Candidate<'tcx>,
        parent_data: &Vec<PatternExtraData<'tcx>>,
    ) -> Self {
        debug_assert!(candidate.match_pairs.is_empty());
        MatchTreeSubBranch {
            span: candidate.extra_data.span,
            success_block: candidate.pre_binding_block.unwrap(),
            otherwise_block: candidate.otherwise_block.unwrap(),
            bindings: parent_data
                .iter()
                .flat_map(|d| &d.bindings)
                .chain(&candidate.extra_data.bindings)
                .cloned()
                .collect(),
            ascriptions: parent_data
                .iter()
                .flat_map(|d| &d.ascriptions)
                .cloned()
                .chain(candidate.extra_data.ascriptions)
                .collect(),
            is_never: candidate.extra_data.is_never,
        }
    }
}

impl<'tcx> MatchTreeBranch<'tcx> {
    fn from_candidate(candidate: Candidate<'tcx>) -> Self {
        let mut sub_branches = Vec::new();
        traverse_candidate(
            candidate,
            &mut Vec::new(),
            &mut |candidate: Candidate<'_>, parent_data: &mut Vec<PatternExtraData<'_>>| {
                sub_branches.push(MatchTreeSubBranch::from_sub_candidate(candidate, parent_data));
            },
            |inner_candidate, parent_data| {
                parent_data.push(inner_candidate.extra_data);
                inner_candidate.subcandidates.into_iter()
            },
            |parent_data| {
                parent_data.pop();
            },
        );
        MatchTreeBranch { sub_branches }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HasMatchGuard {
    Yes,
    No,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// The entrypoint of the matching algorithm. Create the decision tree for the match expression,
    /// starting from `block`.
    ///
    /// `patterns` is a list of patterns, one for each arm. The associated boolean indicates whether
    /// the arm has a guard.
    ///
    /// `refutable` indicates whether the candidate list is refutable (for `if let` and `let else`)
    /// or not (for `let` and `match`). In the refutable case we return the block to which we branch
    /// on failure.
    pub(crate) fn lower_match_tree(
        &mut self,
        block: BasicBlock,
        scrutinee_span: Span,
        scrutinee_place_builder: &PlaceBuilder<'tcx>,
        match_start_span: Span,
        patterns: Vec<(&Pat<'tcx>, HasMatchGuard)>,
        refutable: bool,
    ) -> BuiltMatchTree<'tcx> {
        // Assemble the initial list of candidates. These top-level candidates are 1:1 with the
        // input patterns, but other parts of match lowering also introduce subcandidates (for
        // sub-or-patterns). So inside the algorithm, the candidates list may not correspond to
        // match arms directly.
        let mut candidates: Vec<Candidate<'_>> = patterns
            .into_iter()
            .map(|(pat, has_guard)| {
                Candidate::new(scrutinee_place_builder.clone(), pat, has_guard, self)
            })
            .collect();

        let fake_borrow_temps = util::collect_fake_borrows(
            self,
            &candidates,
            scrutinee_span,
            scrutinee_place_builder.base(),
        );

        // This will generate code to test scrutinee_place and branch to the appropriate arm block.
        // If none of the arms match, we branch to `otherwise_block`. When lowering a `match`
        // expression, exhaustiveness checking ensures that this block is unreachable.
        let mut candidate_refs = candidates.iter_mut().collect::<Vec<_>>();
        let otherwise_block =
            self.match_candidates(match_start_span, scrutinee_span, block, &mut candidate_refs);

        // Set up false edges so that the borrow-checker cannot make use of the specific CFG we
        // generated. We falsely branch from each candidate to the one below it to make it as if we
        // were testing match branches one by one in order. In the refutable case we also want a
        // false edge to the final failure block.
        let mut next_candidate_start_block = if refutable { Some(otherwise_block) } else { None };
        for candidate in candidates.iter_mut().rev() {
            let has_guard = candidate.has_guard;
            candidate.visit_leaves_rev(|leaf_candidate| {
                if let Some(next_candidate_start_block) = next_candidate_start_block {
                    let source_info = self.source_info(leaf_candidate.extra_data.span);
                    // Falsely branch to `next_candidate_start_block` before reaching pre_binding.
                    let old_pre_binding = leaf_candidate.pre_binding_block.unwrap();
                    let new_pre_binding = self.cfg.start_new_block();
                    self.false_edges(
                        old_pre_binding,
                        new_pre_binding,
                        next_candidate_start_block,
                        source_info,
                    );
                    leaf_candidate.pre_binding_block = Some(new_pre_binding);
                    if has_guard {
                        // Falsely branch to `next_candidate_start_block` also if the guard fails.
                        let new_otherwise = self.cfg.start_new_block();
                        let old_otherwise = leaf_candidate.otherwise_block.unwrap();
                        self.false_edges(
                            new_otherwise,
                            old_otherwise,
                            next_candidate_start_block,
                            source_info,
                        );
                        leaf_candidate.otherwise_block = Some(new_otherwise);
                    }
                }
                assert!(leaf_candidate.false_edge_start_block.is_some());
                next_candidate_start_block = leaf_candidate.false_edge_start_block;
            });
        }

        if !refutable {
            // Match checking ensures `otherwise_block` is actually unreachable in irrefutable
            // cases.
            let source_info = self.source_info(scrutinee_span);

            // Matching on a scrutinee place of an uninhabited type doesn't generate any memory
            // reads by itself, and so if the place is uninitialized we wouldn't know. In order to
            // disallow the following:
            // ```rust
            // let x: !;
            // match x {}
            // ```
            // we add a dummy read on the place.
            //
            // NOTE: If we require never patterns for empty matches, those will check that the place
            // is initialized, and so this read would no longer be needed.
            let cause_matched_place = FakeReadCause::ForMatchedPlace(None);

            if let Some(scrutinee_place) = scrutinee_place_builder.try_to_place(self) {
                self.cfg.push_fake_read(
                    otherwise_block,
                    source_info,
                    cause_matched_place,
                    scrutinee_place,
                );
            }

            self.cfg.terminate(otherwise_block, source_info, TerminatorKind::Unreachable);
        }

        BuiltMatchTree {
            branches: candidates.into_iter().map(MatchTreeBranch::from_candidate).collect(),
            otherwise_block,
            fake_borrow_temps,
        }
    }

    /// The main match algorithm. It begins with a set of candidates `candidates` and has the job of
    /// generating code that branches to an appropriate block if the scrutinee matches one of these
    /// candidates. The
    /// candidates are ordered such that the first item in the list
    /// has the highest priority. When a candidate is found to match
    /// the value, we will set and generate a branch to the appropriate
    /// pre-binding block.
    ///
    /// If none of the candidates apply, we continue to the returned `otherwise_block`.
    ///
    /// Note that while `match` expressions in the Rust language are exhaustive,
    /// candidate lists passed to this method are often _non-exhaustive_.
    /// For example, the match lowering process will frequently divide up the
    /// list of candidates, and recursively call this method with a non-exhaustive
    /// subset of candidates.
    /// See [`Builder::test_candidates`] for more details on this
    /// "backtracking automata" approach.
    ///
    /// For an example of how we use `otherwise_block`, consider:
    /// ```
    /// # fn foo((x, y): (bool, bool)) -> u32 {
    /// match (x, y) {
    ///     (true, true) => 1,
    ///     (_, false) => 2,
    ///     (false, true) => 3,
    /// }
    /// # }
    /// ```
    /// For this match, we generate something like:
    /// ```
    /// # fn foo((x, y): (bool, bool)) -> u32 {
    /// if x {
    ///     if y {
    ///         return 1
    ///     } else {
    ///         // continue
    ///     }
    /// } else {
    ///     // continue
    /// }
    /// if y {
    ///     if x {
    ///         // This is actually unreachable because the `(true, true)` case was handled above,
    ///         // but we don't know that from within the lowering algorithm.
    ///         // continue
    ///     } else {
    ///         return 3
    ///     }
    /// } else {
    ///     return 2
    /// }
    /// // this is the final `otherwise_block`, which is unreachable because the match was exhaustive.
    /// unreachable!()
    /// # }
    /// ```
    ///
    /// Every `continue` is an instance of branching to some `otherwise_block` somewhere deep within
    /// the algorithm. For more details on why we lower like this, see [`Builder::test_candidates`].
    ///
    /// Note how we test `x` twice. This is the tradeoff of backtracking automata: we prefer smaller
    /// code size so we accept non-optimal code paths.
    #[instrument(skip(self), level = "debug")]
    fn match_candidates(
        &mut self,
        span: Span,
        scrutinee_span: Span,
        start_block: BasicBlock,
        candidates: &mut [&mut Candidate<'tcx>],
    ) -> BasicBlock {
        ensure_sufficient_stack(|| {
            self.match_candidates_inner(span, scrutinee_span, start_block, candidates)
        })
    }

    /// Construct the decision tree for `candidates`. Don't call this, call `match_candidates`
    /// instead to reserve sufficient stack space.
    fn match_candidates_inner(
        &mut self,
        span: Span,
        scrutinee_span: Span,
        mut start_block: BasicBlock,
        candidates: &mut [&mut Candidate<'tcx>],
    ) -> BasicBlock {
        if let [first, ..] = candidates {
            if first.false_edge_start_block.is_none() {
                first.false_edge_start_block = Some(start_block);
            }
        }

        // Process a prefix of the candidates.
        let rest = match candidates {
            [] => {
                // If there are no candidates that still need testing, we're done.
                return start_block;
            }
            [first, remaining @ ..] if first.match_pairs.is_empty() => {
                // The first candidate has satisfied all its match pairs.
                // We record the blocks that will be needed by match arm lowering,
                // and then continue with the remaining candidates.
                let remainder_start = self.select_matched_candidate(first, start_block);
                remainder_start.and(remaining)
            }
            candidates if candidates.iter().any(|candidate| candidate.starts_with_or_pattern()) => {
                // If any candidate starts with an or-pattern, we want to expand or-patterns
                // before we do any more tests.
                //
                // The only candidate we strictly _need_ to expand here is the first one.
                // But by expanding other candidates as early as possible, we unlock more
                // opportunities to include them in test outcomes, making the match tree
                // smaller and simpler.
                self.expand_and_match_or_candidates(span, scrutinee_span, start_block, candidates)
            }
            candidates => {
                // The first candidate has some unsatisfied match pairs; we proceed to do more tests.
                self.test_candidates(span, scrutinee_span, candidates, start_block)
            }
        };

        // Process any candidates that remain.
        let remaining_candidates = unpack!(start_block = rest);
        self.match_candidates(span, scrutinee_span, start_block, remaining_candidates)
    }

    /// Link up matched candidates.
    ///
    /// For example, if we have something like this:
    ///
    /// ```ignore (illustrative)
    /// ...
    /// Some(x) if cond1 => ...
    /// Some(x) => ...
    /// Some(x) if cond2 => ...
    /// ...
    /// ```
    ///
    /// We generate real edges from:
    ///
    /// * `start_block` to the [pre-binding block] of the first pattern,
    /// * the [otherwise block] of the first pattern to the second pattern,
    /// * the [otherwise block] of the third pattern to a block with an
    ///   [`Unreachable` terminator](TerminatorKind::Unreachable).
    ///
    /// In addition, we later add fake edges from the otherwise blocks to the
    /// pre-binding block of the next candidate in the original set of
    /// candidates.
    ///
    /// [pre-binding block]: Candidate::pre_binding_block
    /// [otherwise block]: Candidate::otherwise_block
    fn select_matched_candidate(
        &mut self,
        candidate: &mut Candidate<'tcx>,
        start_block: BasicBlock,
    ) -> BasicBlock {
        assert!(candidate.otherwise_block.is_none());
        assert!(candidate.pre_binding_block.is_none());
        assert!(candidate.subcandidates.is_empty());

        candidate.pre_binding_block = Some(start_block);
        let otherwise_block = self.cfg.start_new_block();
        // Create the otherwise block for this candidate, which is the
        // pre-binding block for the next candidate.
        candidate.otherwise_block = Some(otherwise_block);
        otherwise_block
    }

    /// Takes a list of candidates such that some of the candidates' first match pairs are
    /// or-patterns. This expands as many or-patterns as possible and processes the resulting
    /// candidates. Returns the unprocessed candidates if any.
    fn expand_and_match_or_candidates<'b, 'c>(
        &mut self,
        span: Span,
        scrutinee_span: Span,
        start_block: BasicBlock,
        candidates: &'b mut [&'c mut Candidate<'tcx>],
    ) -> BlockAnd<&'b mut [&'c mut Candidate<'tcx>]> {
        // We can't expand or-patterns freely. The rule is:
        // - If a candidate doesn't start with an or-pattern, we include it in
        //   the expansion list as-is (i.e. it "expands" to itself).
        // - If a candidate has an or-pattern as its only remaining match pair,
        //   we can expand it.
        // - If it starts with an or-pattern but also has other match pairs,
        //   we can expand it, but we can't process more candidates after it.
        //
        // If we didn't stop, the `otherwise` cases could get mixed up. E.g. in the
        // following, or-pattern simplification (in `merge_trivial_subcandidates`) makes it
        // so the `1` and `2` cases branch to a same block (which then tests `false`). If we
        // took `(2, _)` in the same set of candidates, when we reach the block that tests
        // `false` we don't know whether we came from `1` or `2`, hence we can't know where
        // to branch on failure.
        //
        // ```ignore(illustrative)
        // match (1, true) {
        //     (1 | 2, false) => {},
        //     (2, _) => {},
        //     _ => {}
        // }
        // ```
        //
        // We therefore split the `candidates` slice in two, expand or-patterns in the first part,
        // and process the rest separately.
        let expand_until = candidates
            .iter()
            .position(|candidate| {
                // If a candidate starts with an or-pattern and has more match pairs,
                // we can expand it, but we must stop expanding _after_ it.
                candidate.match_pairs.len() > 1 && candidate.starts_with_or_pattern()
            })
            .map(|pos| pos + 1) // Stop _after_ the found candidate
            .unwrap_or(candidates.len()); // Otherwise, include all candidates
        let (candidates_to_expand, remaining_candidates) = candidates.split_at_mut(expand_until);

        // Expand one level of or-patterns for each candidate in `candidates_to_expand`.
        // We take care to preserve the relative ordering of candidates, so that
        // or-patterns are expanded in their parent's relative position.
        let mut expanded_candidates = Vec::new();
        for candidate in candidates_to_expand.iter_mut() {
            if candidate.starts_with_or_pattern() {
                let or_match_pair = candidate.match_pairs.remove(0);
                // Expand the or-pattern into subcandidates.
                self.create_or_subcandidates(candidate, or_match_pair);
                // Collect the newly created subcandidates.
                for subcandidate in candidate.subcandidates.iter_mut() {
                    expanded_candidates.push(subcandidate);
                }
                // Note that the subcandidates have been added to `expanded_candidates`,
                // but `candidate` itself has not. If the last candidate has more match pairs,
                // they are handled separately by `test_remaining_match_pairs_after_or`.
            } else {
                // A candidate that doesn't start with an or-pattern has nothing to
                // expand, so it is included in the post-expansion list as-is.
                expanded_candidates.push(candidate);
            }
        }

        // Recursively lower the part of the match tree represented by the
        // expanded candidates. This is where subcandidates actually get lowered!
        let remainder_start = self.match_candidates(
            span,
            scrutinee_span,
            start_block,
            expanded_candidates.as_mut_slice(),
        );

        // Postprocess subcandidates, and process any leftover match pairs.
        // (Only the last candidate can possibly have more match pairs.)
        debug_assert!({
            let mut all_except_last = candidates_to_expand.iter().rev().skip(1);
            all_except_last.all(|candidate| candidate.match_pairs.is_empty())
        });
        for candidate in candidates_to_expand.iter_mut() {
            if !candidate.subcandidates.is_empty() {
                self.merge_trivial_subcandidates(candidate);
                self.remove_never_subcandidates(candidate);
            }
        }
        // It's important to perform the above simplifications _before_ dealing
        // with remaining match pairs, to avoid exponential blowup if possible
        // (for trivial or-patterns), and avoid useless work (for never patterns).
        if let Some(last_candidate) = candidates_to_expand.last_mut() {
            self.test_remaining_match_pairs_after_or(span, scrutinee_span, last_candidate);
        }

        remainder_start.and(remaining_candidates)
    }

    /// Given a match-pair that corresponds to an or-pattern, expand each subpattern into a new
    /// subcandidate. Any candidate that has been expanded this way should also be postprocessed
    /// at the end of [`Self::expand_and_match_or_candidates`].
    fn create_or_subcandidates(
        &mut self,
        candidate: &mut Candidate<'tcx>,
        match_pair: MatchPairTree<'tcx>,
    ) {
        let TestCase::Or { pats } = match_pair.test_case else { bug!() };
        debug!("expanding or-pattern: candidate={:#?}\npats={:#?}", candidate, pats);
        candidate.or_span = Some(match_pair.pattern_span);
        candidate.subcandidates = pats
            .into_iter()
            .map(|flat_pat| Candidate::from_flat_pat(flat_pat, candidate.has_guard))
            .collect();
        candidate.subcandidates[0].false_edge_start_block = candidate.false_edge_start_block;
    }

    /// Try to merge all of the subcandidates of the given candidate into one. This avoids
    /// exponentially large CFGs in cases like `(1 | 2, 3 | 4, ...)`. The candidate should have been
    /// expanded with `create_or_subcandidates`.
    ///
    /// Given a pattern `(P | Q, R | S)` we (in principle) generate a CFG like
    /// so:
    ///
    /// ```text
    /// [ start ]
    ///      |
    /// [ match P, Q ]
    ///      |
    ///      +----------------------------------------+------------------------------------+
    ///      |                                        |                                    |
    ///      V                                        V                                    V
    /// [ P matches ]                           [ Q matches ]                        [ otherwise ]
    ///      |                                        |                                    |
    ///      V                                        V                                    |
    /// [ match R, S ]                          [ match R, S ]                             |
    ///      |                                        |                                    |
    ///      +--------------+------------+            +--------------+------------+        |
    ///      |              |            |            |              |            |        |
    ///      V              V            V            V              V            V        |
    /// [ R matches ] [ S matches ] [otherwise ] [ R matches ] [ S matches ] [otherwise ]  |
    ///      |              |            |            |              |            |        |
    ///      +--------------+------------|------------+--------------+            |        |
    ///      |                           |                                        |        |
    ///      |                           +----------------------------------------+--------+
    ///      |                           |
    ///      V                           V
    /// [ Success ]                 [ Failure ]
    /// ```
    ///
    /// In practice there are some complications:
    ///
    /// * If there's a guard, then the otherwise branch of the first match on
    ///   `R | S` goes to a test for whether `Q` matches, and the control flow
    ///   doesn't merge into a single success block until after the guard is
    ///   tested.
    /// * If neither `P` or `Q` has any bindings or type ascriptions and there
    ///   isn't a match guard, then we create a smaller CFG like:
    ///
    /// ```text
    ///     ...
    ///      +---------------+------------+
    ///      |               |            |
    /// [ P matches ] [ Q matches ] [ otherwise ]
    ///      |               |            |
    ///      +---------------+            |
    ///      |                           ...
    /// [ match R, S ]
    ///      |
    ///     ...
    /// ```
    ///
    /// Note that this takes place _after_ the subcandidates have participated
    /// in match tree lowering.
    fn merge_trivial_subcandidates(&mut self, candidate: &mut Candidate<'tcx>) {
        assert!(!candidate.subcandidates.is_empty());
        if candidate.has_guard {
            // FIXME(or_patterns; matthewjasper) Don't give up if we have a guard.
            return;
        }

        // FIXME(or_patterns; matthewjasper) Try to be more aggressive here.
        let can_merge = candidate.subcandidates.iter().all(|subcandidate| {
            subcandidate.subcandidates.is_empty() && subcandidate.extra_data.is_empty()
        });
        if !can_merge {
            return;
        }

        let mut last_otherwise = None;
        let shared_pre_binding_block = self.cfg.start_new_block();
        // This candidate is about to become a leaf, so unset `or_span`.
        let or_span = candidate.or_span.take().unwrap();
        let source_info = self.source_info(or_span);

        if candidate.false_edge_start_block.is_none() {
            candidate.false_edge_start_block = candidate.subcandidates[0].false_edge_start_block;
        }

        // Remove the (known-trivial) subcandidates from the candidate tree,
        // so that they aren't visible after match tree lowering, and wire them
        // all to join up at a single shared pre-binding block.
        // (Note that the subcandidates have already had their part of the match
        // tree lowered by this point, which is why we can add a goto to them.)
        for subcandidate in mem::take(&mut candidate.subcandidates) {
            let subcandidate_block = subcandidate.pre_binding_block.unwrap();
            self.cfg.goto(subcandidate_block, source_info, shared_pre_binding_block);
            last_otherwise = subcandidate.otherwise_block;
        }
        candidate.pre_binding_block = Some(shared_pre_binding_block);
        assert!(last_otherwise.is_some());
        candidate.otherwise_block = last_otherwise;
    }

    /// Never subcandidates may have a set of bindings inconsistent with their siblings,
    /// which would break later code. So we filter them out. Note that we can't filter out
    /// top-level candidates this way.
    fn remove_never_subcandidates(&mut self, candidate: &mut Candidate<'tcx>) {
        if candidate.subcandidates.is_empty() {
            return;
        }

        let false_edge_start_block = candidate.subcandidates[0].false_edge_start_block;
        candidate.subcandidates.retain_mut(|candidate| {
            if candidate.extra_data.is_never {
                candidate.visit_leaves(|subcandidate| {
                    let block = subcandidate.pre_binding_block.unwrap();
                    // That block is already unreachable but needs a terminator to make the MIR well-formed.
                    let source_info = self.source_info(subcandidate.extra_data.span);
                    self.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
                });
                false
            } else {
                true
            }
        });
        if candidate.subcandidates.is_empty() {
            // If `candidate` has become a leaf candidate, ensure it has a `pre_binding_block` and `otherwise_block`.
            let next_block = self.cfg.start_new_block();
            candidate.pre_binding_block = Some(next_block);
            candidate.otherwise_block = Some(next_block);
            // In addition, if `candidate` doesn't have `false_edge_start_block`, it should be assigned here.
            if candidate.false_edge_start_block.is_none() {
                candidate.false_edge_start_block = false_edge_start_block;
            }
        }
    }

    /// If more match pairs remain, test them after each subcandidate.
    /// We could have added them to the or-candidates during or-pattern expansion, but that
    /// would make it impossible to detect simplifiable or-patterns. That would guarantee
    /// exponentially large CFGs for cases like `(1 | 2, 3 | 4, ...)`.
    fn test_remaining_match_pairs_after_or(
        &mut self,
        span: Span,
        scrutinee_span: Span,
        candidate: &mut Candidate<'tcx>,
    ) {
        if candidate.match_pairs.is_empty() {
            return;
        }

        let or_span = candidate.or_span.unwrap_or(candidate.extra_data.span);
        let source_info = self.source_info(or_span);
        let mut last_otherwise = None;
        candidate.visit_leaves(|leaf_candidate| {
            last_otherwise = leaf_candidate.otherwise_block;
        });

        let remaining_match_pairs = mem::take(&mut candidate.match_pairs);
        // We're testing match pairs that remained after an `Or`, so the remaining
        // pairs should all be `Or` too, due to the sorting invariant.
        debug_assert!(
            remaining_match_pairs
                .iter()
                .all(|match_pair| matches!(match_pair.test_case, TestCase::Or { .. }))
        );

        // Visit each leaf candidate within this subtree, add a copy of the remaining
        // match pairs to it, and then recursively lower the rest of the match tree
        // from that point.
        candidate.visit_leaves(|leaf_candidate| {
            // At this point the leaf's own match pairs have all been lowered
            // and removed, so `extend` and assignment are equivalent,
            // but extending can also recycle any existing vector capacity.
            assert!(leaf_candidate.match_pairs.is_empty());
            leaf_candidate.match_pairs.extend(remaining_match_pairs.iter().cloned());

            let or_start = leaf_candidate.pre_binding_block.unwrap();
            let otherwise =
                self.match_candidates(span, scrutinee_span, or_start, &mut [leaf_candidate]);
            // In a case like `(P | Q, R | S)`, if `P` succeeds and `R | S` fails, we know `(Q,
            // R | S)` will fail too. If there is no guard, we skip testing of `Q` by branching
            // directly to `last_otherwise`. If there is a guard,
            // `leaf_candidate.otherwise_block` can be reached by guard failure as well, so we
            // can't skip `Q`.
            let or_otherwise = if leaf_candidate.has_guard {
                leaf_candidate.otherwise_block.unwrap()
            } else {
                last_otherwise.unwrap()
            };
            self.cfg.goto(otherwise, source_info, or_otherwise);
        });
    }

    /// Pick a test to run. Which test doesn't matter as long as it is guaranteed to fully match at
    /// least one match pair. We currently simply pick the test corresponding to the first match
    /// pair of the first candidate in the list.
    ///
    /// *Note:* taking the first match pair is somewhat arbitrary, and we might do better here by
    /// choosing more carefully what to test.
    ///
    /// For example, consider the following possible match-pairs:
    ///
    /// 1. `x @ Some(P)` -- we will do a [`Switch`] to decide what variant `x` has
    /// 2. `x @ 22` -- we will do a [`SwitchInt`] to decide what value `x` has
    /// 3. `x @ 3..5` -- we will do a [`Range`] test to decide what range `x` falls in
    /// 4. etc.
    ///
    /// [`Switch`]: TestKind::Switch
    /// [`SwitchInt`]: TestKind::SwitchInt
    /// [`Range`]: TestKind::Range
    fn pick_test(&mut self, candidates: &[&mut Candidate<'tcx>]) -> (Place<'tcx>, Test<'tcx>) {
        // Extract the match-pair from the highest priority candidate
        let match_pair = &candidates[0].match_pairs[0];
        let test = self.pick_test_for_match_pair(match_pair);
        // Unwrap is ok after simplification.
        let match_place = match_pair.place.unwrap();
        debug!(?test, ?match_pair);

        (match_place, test)
    }

    /// Given a test, we partition the input candidates into several buckets.
    /// If a candidate matches in exactly one of the branches of `test`
    /// (and no other branches), we put it into the corresponding bucket.
    /// If it could match in more than one of the branches of `test`, the test
    /// doesn't usefully apply to it, and we stop partitioning candidates.
    ///
    /// Importantly, we also **mutate** the branched candidates to remove match pairs
    /// that are entailed by the outcome of the test, and add any sub-pairs of the
    /// removed pairs.
    ///
    /// This returns a pair of
    /// - the candidates that weren't sorted;
    /// - for each possible outcome of the test, the candidates that match in that outcome.
    ///
    /// For example:
    /// ```
    /// # let (x, y, z) = (true, true, true);
    /// match (x, y, z) {
    ///     (true , _    , true ) => true,  // (0)
    ///     (false, false, _    ) => false, // (1)
    ///     (_    , true , _    ) => true,  // (2)
    ///     (true , _    , false) => false, // (3)
    /// }
    /// # ;
    /// ```
    ///
    /// Assume we are testing on `x`. Conceptually, there are 2 overlapping candidate sets:
    /// - If the outcome is that `x` is true, candidates {0, 2, 3} are possible
    /// - If the outcome is that `x` is false, candidates {1, 2} are possible
    ///
    /// Following our algorithm:
    /// - Candidate 0 is sorted into outcome `x == true`
    /// - Candidate 1 is sorted into outcome `x == false`
    /// - Candidate 2 remains unsorted, because testing `x` has no effect on it
    /// - Candidate 3 remains unsorted, because a previous candidate (2) was unsorted
    ///   - This helps preserve the illusion that candidates are tested "in order"
    ///
    /// The sorted candidates are mutated to remove entailed match pairs:
    /// - candidate 0 becomes `[z @ true]` since we know that `x` was `true`;
    /// - candidate 1 becomes `[y @ false]` since we know that `x` was `false`.
    fn sort_candidates<'b, 'c>(
        &mut self,
        match_place: Place<'tcx>,
        test: &Test<'tcx>,
        mut candidates: &'b mut [&'c mut Candidate<'tcx>],
    ) -> (
        &'b mut [&'c mut Candidate<'tcx>],
        FxIndexMap<TestBranch<'tcx>, Vec<&'b mut Candidate<'tcx>>>,
    ) {
        // For each of the possible outcomes, collect vector of candidates that apply if the test
        // has that particular outcome.
        let mut target_candidates: FxIndexMap<_, Vec<&mut Candidate<'_>>> = Default::default();

        let total_candidate_count = candidates.len();

        // Sort the candidates into the appropriate vector in `target_candidates`. Note that at some
        // point we may encounter a candidate where the test is not relevant; at that point, we stop
        // sorting.
        while let Some(candidate) = candidates.first_mut() {
            let Some(branch) =
                self.sort_candidate(match_place, test, candidate, &target_candidates)
            else {
                break;
            };
            let (candidate, rest) = candidates.split_first_mut().unwrap();
            target_candidates.entry(branch).or_insert_with(Vec::new).push(candidate);
            candidates = rest;
        }

        // At least the first candidate ought to be tested
        assert!(
            total_candidate_count > candidates.len(),
            "{total_candidate_count}, {candidates:#?}"
        );
        debug!("tested_candidates: {}", total_candidate_count - candidates.len());
        debug!("untested_candidates: {}", candidates.len());

        (candidates, target_candidates)
    }

    /// This is the most subtle part of the match lowering algorithm. At this point, there are
    /// no fully-satisfied candidates, and no or-patterns to expand, so we actually need to
    /// perform some sort of test to make progress.
    ///
    /// Once we pick what sort of test we are going to perform, this test will help us winnow down
    /// our candidates. So we walk over the candidates (from high to low priority) and check. We
    /// compute, for each outcome of the test, a list of (modified) candidates. If a candidate
    /// matches in exactly one branch of our test, we add it to the corresponding outcome. We also
    /// **mutate its list of match pairs** if appropriate, to reflect the fact that we know which
    /// outcome occurred.
    ///
    /// For example, if we are testing `x.0`'s variant, and we have a candidate `(x.0 @ Some(v), x.1
    /// @ 22)`, then we would have a resulting candidate of `((x.0 as Some).0 @ v, x.1 @ 22)` in the
    /// branch corresponding to `Some`. To ensure we make progress, we always pick a test that
    /// results in simplifying the first candidate.
    ///
    /// But there may also be candidates that the test doesn't
    /// apply to. The classical example is wildcards:
    ///
    /// ```
    /// # let (x, y, z) = (true, true, true);
    /// match (x, y, z) {
    ///     (true , _    , true ) => true,  // (0)
    ///     (false, false, _    ) => false, // (1)
    ///     (_    , true , _    ) => true,  // (2)
    ///     (true , _    , false) => false, // (3)
    /// }
    /// # ;
    /// ```
    ///
    /// Here, the traditional "decision tree" method would generate 2 separate code-paths for the 2
    /// possible values of `x`. This would however duplicate some candidates, which would need to be
    /// lowered several times.
    ///
    /// In some cases, this duplication can create an exponential amount of
    /// code. This is most easily seen by noticing that this method terminates
    /// with precisely the reachable arms being reachable - but that problem
    /// is trivially NP-complete:
    ///
    /// ```ignore (illustrative)
    /// match (var0, var1, var2, var3, ...) {
    ///     (true , _   , _    , false, true, ...) => false,
    ///     (_    , true, true , false, _   , ...) => false,
    ///     (false, _   , false, false, _   , ...) => false,
    ///     ...
    ///     _ => true
    /// }
    /// ```
    ///
    /// Here the last arm is reachable only if there is an assignment to
    /// the variables that does not match any of the literals. Therefore,
    /// compilation would take an exponential amount of time in some cases.
    ///
    /// In rustc, we opt instead for the "backtracking automaton" approach. This guarantees we never
    /// duplicate a candidate (except in the presence of or-patterns). In fact this guarantee is
    /// ensured by the fact that we carry around `&mut Candidate`s which can't be duplicated.
    ///
    /// To make this work, whenever we decide to perform a test, if we encounter a candidate that
    /// could match in more than one branch of the test, we stop. We generate code for the test and
    /// for the candidates in its branches; the remaining candidates will be tested if the
    /// candidates in the branches fail to match.
    ///
    /// For example, if we test on `x` in the following:
    /// ```
    /// # fn foo((x, y, z): (bool, bool, bool)) -> u32 {
    /// match (x, y, z) {
    ///     (true , _    , true ) => 0,
    ///     (false, false, _    ) => 1,
    ///     (_    , true , _    ) => 2,
    ///     (true , _    , false) => 3,
    /// }
    /// # }
    /// ```
    /// this function generates code that looks more of less like:
    /// ```
    /// # fn foo((x, y, z): (bool, bool, bool)) -> u32 {
    /// if x {
    ///     match (y, z) {
    ///         (_, true) => return 0,
    ///         _ => {} // continue matching
    ///     }
    /// } else {
    ///     match (y, z) {
    ///         (false, _) => return 1,
    ///         _ => {} // continue matching
    ///     }
    /// }
    /// // the block here is `remainder_start`
    /// match (x, y, z) {
    ///     (_    , true , _    ) => 2,
    ///     (true , _    , false) => 3,
    ///     _ => unreachable!(),
    /// }
    /// # }
    /// ```
    ///
    /// We return the unprocessed candidates.
    fn test_candidates<'b, 'c>(
        &mut self,
        span: Span,
        scrutinee_span: Span,
        candidates: &'b mut [&'c mut Candidate<'tcx>],
        start_block: BasicBlock,
    ) -> BlockAnd<&'b mut [&'c mut Candidate<'tcx>]> {
        // Choose a match pair from the first candidate, and use it to determine a
        // test to perform that will confirm or refute that match pair.
        let (match_place, test) = self.pick_test(candidates);

        // For each of the N possible test outcomes, build the vector of candidates that applies if
        // the test has that particular outcome. This also mutates the candidates to remove match
        // pairs that are fully satisfied by the relevant outcome.
        let (remaining_candidates, target_candidates) =
            self.sort_candidates(match_place, &test, candidates);

        // The block that we should branch to if none of the `target_candidates` match.
        let remainder_start = self.cfg.start_new_block();

        // For each outcome of the test, recursively lower the rest of the match tree
        // from that point. (Note that we haven't lowered the actual test yet!)
        let target_blocks: FxIndexMap<_, _> = target_candidates
            .into_iter()
            .map(|(branch, mut candidates)| {
                let branch_start = self.cfg.start_new_block();
                // Recursively lower the rest of the match tree after the relevant outcome.
                let branch_otherwise =
                    self.match_candidates(span, scrutinee_span, branch_start, &mut *candidates);

                // Link up the `otherwise` block of the subtree to `remainder_start`.
                let source_info = self.source_info(span);
                self.cfg.goto(branch_otherwise, source_info, remainder_start);
                (branch, branch_start)
            })
            .collect();

        // Perform the chosen test, branching to one of the N subtrees prepared above
        // (or to `remainder_start` if no outcome was satisfied).
        self.perform_test(
            span,
            scrutinee_span,
            start_block,
            remainder_start,
            match_place,
            &test,
            target_blocks,
        );

        remainder_start.and(remaining_candidates)
    }
}

///////////////////////////////////////////////////////////////////////////
// Pat binding - used for `let` and function parameters as well.

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Lowers a `let` expression that appears in a suitable context
    /// (e.g. an `if` condition or match guard).
    ///
    /// Also used for lowering let-else statements, since they have similar
    /// needs despite not actually using `let` expressions.
    ///
    /// Use [`DeclareLetBindings`] to control whether the `let` bindings are
    /// declared or not.
    pub(crate) fn lower_let_expr(
        &mut self,
        mut block: BasicBlock,
        expr_id: ExprId,
        pat: &Pat<'tcx>,
        source_scope: Option<SourceScope>,
        scope_span: Span,
        declare_let_bindings: DeclareLetBindings,
        emit_storage_live: EmitStorageLive,
    ) -> BlockAnd<()> {
        let expr_span = self.thir[expr_id].span;
        let scrutinee = unpack!(block = self.lower_scrutinee(block, expr_id, expr_span));
        let built_tree = self.lower_match_tree(
            block,
            expr_span,
            &scrutinee,
            pat.span,
            vec![(pat, HasMatchGuard::No)],
            true,
        );
        let [branch] = built_tree.branches.try_into().unwrap();

        self.break_for_else(built_tree.otherwise_block, self.source_info(expr_span));

        match declare_let_bindings {
            DeclareLetBindings::Yes => {
                let expr_place = scrutinee.try_to_place(self);
                let opt_expr_place = expr_place.as_ref().map(|place| (Some(place), expr_span));
                self.declare_bindings(
                    source_scope,
                    pat.span.to(scope_span),
                    pat,
                    None,
                    opt_expr_place,
                );
            }
            DeclareLetBindings::No => {} // Caller is responsible for bindings.
            DeclareLetBindings::LetNotPermitted => {
                self.tcx.dcx().span_bug(expr_span, "let expression not expected in this context")
            }
        }

        let success = self.bind_pattern(
            self.source_info(pat.span),
            branch,
            &[],
            expr_span,
            None,
            emit_storage_live,
        );

        // If branch coverage is enabled, record this branch.
        self.visit_coverage_conditional_let(pat, success, built_tree.otherwise_block);

        success.unit()
    }

    /// Initializes each of the bindings from the candidate by
    /// moving/copying/ref'ing the source as appropriate. Tests the guard, if
    /// any, and then branches to the arm. Returns the block for the case where
    /// the guard succeeds.
    ///
    /// Note: we do not check earlier that if there is a guard,
    /// there cannot be move bindings. We avoid a use-after-move by only
    /// moving the binding once the guard has evaluated to true (see below).
    fn bind_and_guard_matched_candidate(
        &mut self,
        sub_branch: MatchTreeSubBranch<'tcx>,
        fake_borrows: &[(Place<'tcx>, Local, FakeBorrowKind)],
        scrutinee_span: Span,
        arm_match_scope: Option<(&Arm<'tcx>, region::Scope)>,
        schedule_drops: ScheduleDrops,
        emit_storage_live: EmitStorageLive,
    ) -> BasicBlock {
        debug!("bind_and_guard_matched_candidate(subbranch={:?})", sub_branch);

        let block = sub_branch.success_block;

        if sub_branch.is_never {
            // This arm has a dummy body, we don't need to generate code for it. `block` is already
            // unreachable (except via false edge).
            let source_info = self.source_info(sub_branch.span);
            self.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
            return self.cfg.start_new_block();
        }

        self.ascribe_types(block, sub_branch.ascriptions);

        // Lower an instance of the arm guard (if present) for this candidate,
        // and then perform bindings for the arm body.
        if let Some((arm, match_scope)) = arm_match_scope
            && let Some(guard) = arm.guard
        {
            let tcx = self.tcx;

            // Bindings for guards require some extra handling to automatically
            // insert implicit references/dereferences.
            self.bind_matched_candidate_for_guard(
                block,
                schedule_drops,
                sub_branch.bindings.iter(),
            );
            let guard_frame = GuardFrame {
                locals: sub_branch
                    .bindings
                    .iter()
                    .map(|b| GuardFrameLocal::new(b.var_id))
                    .collect(),
            };
            debug!("entering guard building context: {:?}", guard_frame);
            self.guard_context.push(guard_frame);

            let re_erased = tcx.lifetimes.re_erased;
            let scrutinee_source_info = self.source_info(scrutinee_span);
            for &(place, temp, kind) in fake_borrows {
                let borrow = Rvalue::Ref(re_erased, BorrowKind::Fake(kind), place);
                self.cfg.push_assign(block, scrutinee_source_info, Place::from(temp), borrow);
            }

            let mut guard_span = rustc_span::DUMMY_SP;

            let (post_guard_block, otherwise_post_guard_block) =
                self.in_if_then_scope(match_scope, guard_span, |this| {
                    guard_span = this.thir[guard].span;
                    this.then_else_break(
                        block,
                        guard,
                        None, // Use `self.local_scope()` as the temp scope
                        this.source_info(arm.span),
                        DeclareLetBindings::No, // For guards, `let` bindings are declared separately
                    )
                });

            let source_info = self.source_info(guard_span);
            let guard_end = self.source_info(tcx.sess.source_map().end_point(guard_span));
            let guard_frame = self.guard_context.pop().unwrap();
            debug!("Exiting guard building context with locals: {:?}", guard_frame);

            for &(_, temp, _) in fake_borrows {
                let cause = FakeReadCause::ForMatchGuard;
                self.cfg.push_fake_read(post_guard_block, guard_end, cause, Place::from(temp));
            }

            self.cfg.goto(otherwise_post_guard_block, source_info, sub_branch.otherwise_block);

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
            // would yield an `arm_block` something like:
            //
            // ```
            // StorageLive(_4);        // _4 is `x`
            // _4 = &mut (_1.0: i32);  // this is handling `(mut x, 1)` case
            // _4 = &mut (_1.1: i32);  // this is handling `(2, mut x)` case
            // ```
            //
            // and that is clearly not correct.
            let by_value_bindings = sub_branch
                .bindings
                .iter()
                .filter(|binding| matches!(binding.binding_mode.0, ByRef::No));
            // Read all of the by reference bindings to ensure that the
            // place they refer to can't be modified by the guard.
            for binding in by_value_bindings.clone() {
                let local_id = self.var_local_id(binding.var_id, RefWithinGuard);
                let cause = FakeReadCause::ForGuardBinding;
                self.cfg.push_fake_read(post_guard_block, guard_end, cause, Place::from(local_id));
            }
            assert_matches!(
                schedule_drops,
                ScheduleDrops::Yes,
                "patterns with guards must schedule drops"
            );
            self.bind_matched_candidate_for_arm_body(
                post_guard_block,
                ScheduleDrops::Yes,
                by_value_bindings,
                emit_storage_live,
            );

            post_guard_block
        } else {
            // (Here, it is not too early to bind the matched
            // candidate on `block`, because there is no guard result
            // that we have to inspect before we bind them.)
            self.bind_matched_candidate_for_arm_body(
                block,
                schedule_drops,
                sub_branch.bindings.iter(),
                emit_storage_live,
            );
            block
        }
    }

    /// Append `AscribeUserType` statements onto the end of `block`
    /// for each ascription
    fn ascribe_types(
        &mut self,
        block: BasicBlock,
        ascriptions: impl IntoIterator<Item = Ascription<'tcx>>,
    ) {
        for ascription in ascriptions {
            let source_info = self.source_info(ascription.annotation.span);

            let base = self.canonical_user_type_annotations.push(ascription.annotation);
            self.cfg.push(
                block,
                Statement {
                    source_info,
                    kind: StatementKind::AscribeUserType(
                        Box::new((
                            ascription.source,
                            UserTypeProjection { base, projs: Vec::new() },
                        )),
                        ascription.variance,
                    ),
                },
            );
        }
    }

    /// Binding for guards is a bit different from binding for the arm body,
    /// because an extra layer of implicit reference/dereference is added.
    ///
    /// The idea is that any pattern bindings of type T will map to a `&T` within
    /// the context of the guard expression, but will continue to map to a `T`
    /// in the context of the arm body. To avoid surfacing this distinction in
    /// the user source code (which would be a severe change to the language and
    /// require far more revision to the compiler), any occurrence of the
    /// identifier in the guard expression will automatically get a deref op
    /// applied to it. (See the caller of [`Self::is_bound_var_in_guard`].)
    ///
    /// So an input like:
    ///
    /// ```ignore (illustrative)
    /// let place = Foo::new();
    /// match place { foo if inspect(foo)
    ///     => feed(foo), ... }
    /// ```
    ///
    /// will be treated as if it were really something like:
    ///
    /// ```ignore (illustrative)
    /// let place = Foo::new();
    /// match place { Foo { .. } if { let tmp1 = &place; inspect(*tmp1) }
    ///     => { let tmp2 = place; feed(tmp2) }, ... }
    /// ```
    ///
    /// And an input like:
    ///
    /// ```ignore (illustrative)
    /// let place = Foo::new();
    /// match place { ref mut foo if inspect(foo)
    ///     => feed(foo), ... }
    /// ```
    ///
    /// will be treated as if it were really something like:
    ///
    /// ```ignore (illustrative)
    /// let place = Foo::new();
    /// match place { Foo { .. } if { let tmp1 = & &mut place; inspect(*tmp1) }
    ///     => { let tmp2 = &mut place; feed(tmp2) }, ... }
    /// ```
    /// ---
    ///
    /// ## Implementation notes
    ///
    /// To encode the distinction above, we must inject the
    /// temporaries `tmp1` and `tmp2`.
    ///
    /// There are two cases of interest: binding by-value, and binding by-ref.
    ///
    /// 1. Binding by-value: Things are simple.
    ///
    ///    * Establishing `tmp1` creates a reference into the
    ///      matched place. This code is emitted by
    ///      [`Self::bind_matched_candidate_for_guard`].
    ///
    ///    * `tmp2` is only initialized "lazily", after we have
    ///      checked the guard. Thus, the code that can trigger
    ///      moves out of the candidate can only fire after the
    ///      guard evaluated to true. This initialization code is
    ///      emitted by [`Self::bind_matched_candidate_for_arm_body`].
    ///
    /// 2. Binding by-reference: Things are tricky.
    ///
    ///    * Here, the guard expression wants a `&&` or `&&mut`
    ///      into the original input. This means we need to borrow
    ///      the reference that we create for the arm.
    ///    * So we eagerly create the reference for the arm and then take a
    ///      reference to that.
    ///
    /// ---
    ///
    /// See these PRs for some historical context:
    /// - <https://github.com/rust-lang/rust/pull/49870> (introduction of autoref)
    /// - <https://github.com/rust-lang/rust/pull/59114> (always use autoref)
    fn bind_matched_candidate_for_guard<'b>(
        &mut self,
        block: BasicBlock,
        schedule_drops: ScheduleDrops,
        bindings: impl IntoIterator<Item = &'b Binding<'tcx>>,
    ) where
        'tcx: 'b,
    {
        debug!("bind_matched_candidate_for_guard(block={:?})", block);

        // Assign each of the bindings. Since we are binding for a
        // guard expression, this will never trigger moves out of the
        // candidate.
        let re_erased = self.tcx.lifetimes.re_erased;
        for binding in bindings {
            debug!("bind_matched_candidate_for_guard(binding={:?})", binding);
            let source_info = self.source_info(binding.span);

            // For each pattern ident P of type T, `ref_for_guard` is
            // a reference R: &T pointing to the location matched by
            // the pattern, and every occurrence of P within a guard
            // denotes *R.
            let ref_for_guard = self.storage_live_binding(
                block,
                binding.var_id,
                binding.span,
                RefWithinGuard,
                schedule_drops,
            );
            match binding.binding_mode.0 {
                ByRef::No => {
                    // The arm binding will be by value, so for the guard binding
                    // just take a shared reference to the matched place.
                    let rvalue = Rvalue::Ref(re_erased, BorrowKind::Shared, binding.source);
                    self.cfg.push_assign(block, source_info, ref_for_guard, rvalue);
                }
                ByRef::Yes(mutbl) => {
                    // The arm binding will be by reference, so eagerly create it now.
                    let value_for_arm = self.storage_live_binding(
                        block,
                        binding.var_id,
                        binding.span,
                        OutsideGuard,
                        schedule_drops,
                    );

                    let rvalue =
                        Rvalue::Ref(re_erased, util::ref_pat_borrow_kind(mutbl), binding.source);
                    self.cfg.push_assign(block, source_info, value_for_arm, rvalue);
                    // For the guard binding, take a shared reference to that reference.
                    let rvalue = Rvalue::Ref(re_erased, BorrowKind::Shared, value_for_arm);
                    self.cfg.push_assign(block, source_info, ref_for_guard, rvalue);
                }
            }
        }
    }

    fn bind_matched_candidate_for_arm_body<'b>(
        &mut self,
        block: BasicBlock,
        schedule_drops: ScheduleDrops,
        bindings: impl IntoIterator<Item = &'b Binding<'tcx>>,
        emit_storage_live: EmitStorageLive,
    ) where
        'tcx: 'b,
    {
        debug!("bind_matched_candidate_for_arm_body(block={:?})", block);

        let re_erased = self.tcx.lifetimes.re_erased;
        // Assign each of the bindings. This may trigger moves out of the candidate.
        for binding in bindings {
            let source_info = self.source_info(binding.span);
            let local = match emit_storage_live {
                // Here storages are already alive, probably because this is a binding
                // from let-else.
                // We just need to schedule drop for the value.
                EmitStorageLive::No => self.var_local_id(binding.var_id, OutsideGuard).into(),
                EmitStorageLive::Yes => self.storage_live_binding(
                    block,
                    binding.var_id,
                    binding.span,
                    OutsideGuard,
                    schedule_drops,
                ),
            };
            if matches!(schedule_drops, ScheduleDrops::Yes) {
                self.schedule_drop_for_binding(binding.var_id, binding.span, OutsideGuard);
            }
            let rvalue = match binding.binding_mode.0 {
                ByRef::No => Rvalue::Use(self.consume_by_copy_or_move(binding.source)),
                ByRef::Yes(mutbl) => {
                    Rvalue::Ref(re_erased, util::ref_pat_borrow_kind(mutbl), binding.source)
                }
            };
            self.cfg.push_assign(block, source_info, local, rvalue);
        }
    }

    /// Each binding (`ref mut var`/`ref var`/`mut var`/`var`, where the bound
    /// `var` has type `T` in the arm body) in a pattern maps to 2 locals. The
    /// first local is a binding for occurrences of `var` in the guard, which
    /// will have type `&T`. The second local is a binding for occurrences of
    /// `var` in the arm body, which will have type `T`.
    #[instrument(skip(self), level = "debug")]
    fn declare_binding(
        &mut self,
        source_info: SourceInfo,
        visibility_scope: SourceScope,
        name: Symbol,
        mode: BindingMode,
        var_id: LocalVarId,
        var_ty: Ty<'tcx>,
        user_ty: Option<Box<UserTypeProjections>>,
        has_guard: ArmHasGuard,
        opt_match_place: Option<(Option<Place<'tcx>>, Span)>,
        pat_span: Span,
    ) {
        let tcx = self.tcx;
        let debug_source_info = SourceInfo { span: source_info.span, scope: visibility_scope };
        let local = LocalDecl {
            mutability: mode.1,
            ty: var_ty,
            user_ty,
            source_info,
            local_info: ClearCrossCrate::Set(Box::new(LocalInfo::User(BindingForm::Var(
                VarBindingForm {
                    binding_mode: mode,
                    // hypothetically, `visit_primary_bindings` could try to unzip
                    // an outermost hir::Ty as we descend, matching up
                    // idents in pat; but complex w/ unclear UI payoff.
                    // Instead, just abandon providing diagnostic info.
                    opt_ty_info: None,
                    opt_match_place,
                    pat_span,
                },
            )))),
        };
        let for_arm_body = self.local_decls.push(local);
        if self.should_emit_debug_info_for_binding(name, var_id) {
            self.var_debug_info.push(VarDebugInfo {
                name,
                source_info: debug_source_info,
                value: VarDebugInfoContents::Place(for_arm_body.into()),
                composite: None,
                argument_index: None,
            });
        }
        let locals = if has_guard.0 {
            let ref_for_guard = self.local_decls.push(LocalDecl::<'tcx> {
                // This variable isn't mutated but has a name, so has to be
                // immutable to avoid the unused mut lint.
                mutability: Mutability::Not,
                ty: Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, var_ty),
                user_ty: None,
                source_info,
                local_info: ClearCrossCrate::Set(Box::new(LocalInfo::User(
                    BindingForm::RefForGuard,
                ))),
            });
            if self.should_emit_debug_info_for_binding(name, var_id) {
                self.var_debug_info.push(VarDebugInfo {
                    name,
                    source_info: debug_source_info,
                    value: VarDebugInfoContents::Place(ref_for_guard.into()),
                    composite: None,
                    argument_index: None,
                });
            }
            LocalsForNode::ForGuard { ref_for_guard, for_arm_body }
        } else {
            LocalsForNode::One(for_arm_body)
        };
        debug!(?locals);
        self.var_indices.insert(var_id, locals);
    }

    /// Some bindings are introduced when producing HIR from the AST and don't
    /// actually exist in the source. Skip producing debug info for those when
    /// we can recognize them.
    fn should_emit_debug_info_for_binding(&self, name: Symbol, var_id: LocalVarId) -> bool {
        // For now we only recognize the output of desugaring assigns.
        if name != sym::lhs {
            return true;
        }

        let tcx = self.tcx;
        for (_, node) in tcx.hir_parent_iter(var_id.0) {
            // FIXME(khuey) at what point is it safe to bail on the iterator?
            // Can we stop at the first non-Pat node?
            if matches!(node, Node::LetStmt(&LetStmt { source: LocalSource::AssignDesugar(_), .. }))
            {
                return false;
            }
        }

        true
    }

    /// Attempt to statically pick the `BasicBlock` that a value would resolve to at runtime.
    pub(crate) fn static_pattern_match(
        &self,
        cx: &RustcPatCtxt<'_, 'tcx>,
        valtree: ValTree<'tcx>,
        arms: &[ArmId],
        built_match_tree: &BuiltMatchTree<'tcx>,
    ) -> Option<BasicBlock> {
        let it = arms.iter().zip(built_match_tree.branches.iter());
        for (&arm_id, branch) in it {
            let pat = cx.lower_pat(&*self.thir.arms[arm_id].pattern);

            // Peel off or-patterns if they exist.
            if let rustc_pattern_analysis::rustc::Constructor::Or = pat.ctor() {
                for pat in pat.iter_fields() {
                    // For top-level or-patterns (the only ones we accept right now), when the
                    // bindings are the same (e.g. there are none), the sub_branch is stored just
                    // once.
                    let sub_branch = branch
                        .sub_branches
                        .get(pat.idx)
                        .or_else(|| branch.sub_branches.last())
                        .unwrap();

                    match self.static_pattern_match_inner(valtree, &pat.pat) {
                        true => return Some(sub_branch.success_block),
                        false => continue,
                    }
                }
            } else if self.static_pattern_match_inner(valtree, &pat) {
                return Some(branch.sub_branches[0].success_block);
            }
        }

        None
    }

    /// Helper for [`Self::static_pattern_match`], checking whether the value represented by the
    /// `ValTree` matches the given pattern. This function does not recurse, meaning that it does
    /// not handle or-patterns, or patterns for types with fields.
    fn static_pattern_match_inner(
        &self,
        valtree: ty::ValTree<'tcx>,
        pat: &DeconstructedPat<'_, 'tcx>,
    ) -> bool {
        use rustc_pattern_analysis::constructor::{IntRange, MaybeInfiniteInt};
        use rustc_pattern_analysis::rustc::Constructor;

        match pat.ctor() {
            Constructor::Variant(variant_index) => {
                let ValTreeKind::Branch(box [actual_variant_idx]) = *valtree else {
                    bug!("malformed valtree for an enum")
                };

                let ValTreeKind::Leaf(actual_variant_idx) = ***actual_variant_idx else {
                    bug!("malformed valtree for an enum")
                };

                *variant_index == VariantIdx::from_u32(actual_variant_idx.to_u32())
            }
            Constructor::IntRange(int_range) => {
                let size = pat.ty().primitive_size(self.tcx);
                let actual_int = valtree.unwrap_leaf().to_bits(size);
                let actual_int = if pat.ty().is_signed() {
                    MaybeInfiniteInt::new_finite_int(actual_int, size.bits())
                } else {
                    MaybeInfiniteInt::new_finite_uint(actual_int)
                };
                IntRange::from_singleton(actual_int).is_subrange(int_range)
            }
            Constructor::Bool(pattern_value) => match valtree.unwrap_leaf().try_to_bool() {
                Ok(actual_value) => *pattern_value == actual_value,
                Err(()) => bug!("bool value with invalid bits"),
            },
            Constructor::F16Range(l, h, end) => {
                let actual = valtree.unwrap_leaf().to_f16();
                match end {
                    RangeEnd::Included => (*l..=*h).contains(&actual),
                    RangeEnd::Excluded => (*l..*h).contains(&actual),
                }
            }
            Constructor::F32Range(l, h, end) => {
                let actual = valtree.unwrap_leaf().to_f32();
                match end {
                    RangeEnd::Included => (*l..=*h).contains(&actual),
                    RangeEnd::Excluded => (*l..*h).contains(&actual),
                }
            }
            Constructor::F64Range(l, h, end) => {
                let actual = valtree.unwrap_leaf().to_f64();
                match end {
                    RangeEnd::Included => (*l..=*h).contains(&actual),
                    RangeEnd::Excluded => (*l..*h).contains(&actual),
                }
            }
            Constructor::F128Range(l, h, end) => {
                let actual = valtree.unwrap_leaf().to_f128();
                match end {
                    RangeEnd::Included => (*l..=*h).contains(&actual),
                    RangeEnd::Excluded => (*l..*h).contains(&actual),
                }
            }
            Constructor::Wildcard => true,

            // These we may eventually support:
            Constructor::Struct
            | Constructor::Ref
            | Constructor::DerefPattern(_)
            | Constructor::Slice(_)
            | Constructor::UnionField
            | Constructor::Or
            | Constructor::Str(_) => bug!("unsupported pattern constructor {:?}", pat.ctor()),

            // These should never occur here:
            Constructor::Opaque(_)
            | Constructor::Never
            | Constructor::NonExhaustive
            | Constructor::Hidden
            | Constructor::Missing
            | Constructor::PrivateUninhabited => {
                bug!("unsupported pattern constructor {:?}", pat.ctor())
            }
        }
    }
}
