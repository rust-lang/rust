//! Various diagnostics for expressions that are collected together in one pass
//! through the body using inference results: mismatched arg counts, missing
//! fields, etc.

use std::fmt;

use base_db::Crate;
use chalk_solve::rust_ir::AdtKind;
use either::Either;
use hir_def::{
    AdtId, AssocItemId, DefWithBodyId, HasModule, ItemContainerId, Lookup,
    lang_item::LangItem,
    resolver::{HasResolver, ValueNs},
};
use intern::sym;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use rustc_pattern_analysis::constructor::Constructor;
use syntax::{
    AstNode,
    ast::{self, UnaryOp},
};
use tracing::debug;
use triomphe::Arc;
use typed_arena::Arena;

use crate::{
    Adjust, InferenceResult, Interner, TraitEnvironment, Ty, TyExt, TyKind,
    db::HirDatabase,
    diagnostics::match_check::{
        self,
        pat_analysis::{self, DeconstructedPat, MatchCheckCtx, WitnessPat},
    },
    display::{DisplayTarget, HirDisplay},
};

pub(crate) use hir_def::{
    LocalFieldId, VariantId,
    expr_store::Body,
    hir::{Expr, ExprId, MatchArm, Pat, PatId, Statement},
};

pub enum BodyValidationDiagnostic {
    RecordMissingFields {
        record: Either<ExprId, PatId>,
        variant: VariantId,
        missed_fields: Vec<LocalFieldId>,
    },
    ReplaceFilterMapNextWithFindMap {
        method_call_expr: ExprId,
    },
    MissingMatchArms {
        match_expr: ExprId,
        uncovered_patterns: String,
    },
    NonExhaustiveLet {
        pat: PatId,
        uncovered_patterns: String,
    },
    RemoveTrailingReturn {
        return_expr: ExprId,
    },
    RemoveUnnecessaryElse {
        if_expr: ExprId,
    },
}

impl BodyValidationDiagnostic {
    pub fn collect(
        db: &dyn HirDatabase,
        owner: DefWithBodyId,
        validate_lints: bool,
    ) -> Vec<BodyValidationDiagnostic> {
        let _p = tracing::info_span!("BodyValidationDiagnostic::collect").entered();
        let infer = db.infer(owner);
        let body = db.body(owner);
        let env = db.trait_environment_for_body(owner);
        let mut validator =
            ExprValidator { owner, body, infer, diagnostics: Vec::new(), validate_lints, env };
        validator.validate_body(db);
        validator.diagnostics
    }
}

struct ExprValidator {
    owner: DefWithBodyId,
    body: Arc<Body>,
    infer: Arc<InferenceResult>,
    env: Arc<TraitEnvironment>,
    diagnostics: Vec<BodyValidationDiagnostic>,
    validate_lints: bool,
}

impl ExprValidator {
    fn validate_body(&mut self, db: &dyn HirDatabase) {
        let mut filter_map_next_checker = None;
        // we'll pass &mut self while iterating over body.exprs, so they need to be disjoint
        let body = Arc::clone(&self.body);

        if matches!(self.owner, DefWithBodyId::FunctionId(_)) {
            self.check_for_trailing_return(body.body_expr, &body);
        }

        for (id, expr) in body.exprs.iter() {
            if let Some((variant, missed_fields, true)) =
                record_literal_missing_fields(db, &self.infer, id, expr)
            {
                self.diagnostics.push(BodyValidationDiagnostic::RecordMissingFields {
                    record: Either::Left(id),
                    variant,
                    missed_fields,
                });
            }

            match expr {
                Expr::Match { expr, arms } => {
                    self.validate_match(id, *expr, arms, db);
                }
                Expr::Call { .. } | Expr::MethodCall { .. } => {
                    self.validate_call(db, id, expr, &mut filter_map_next_checker);
                }
                Expr::Closure { body: body_expr, .. } => {
                    self.check_for_trailing_return(*body_expr, &body);
                }
                Expr::If { .. } => {
                    self.check_for_unnecessary_else(id, expr, db);
                }
                Expr::Block { .. } | Expr::Async { .. } | Expr::Unsafe { .. } => {
                    self.validate_block(db, expr);
                }
                _ => {}
            }
        }

        for (id, pat) in body.pats.iter() {
            if let Some((variant, missed_fields, true)) =
                record_pattern_missing_fields(db, &self.infer, id, pat)
            {
                self.diagnostics.push(BodyValidationDiagnostic::RecordMissingFields {
                    record: Either::Right(id),
                    variant,
                    missed_fields,
                });
            }
        }
    }

    fn validate_call(
        &mut self,
        db: &dyn HirDatabase,
        call_id: ExprId,
        expr: &Expr,
        filter_map_next_checker: &mut Option<FilterMapNextChecker>,
    ) {
        if !self.validate_lints {
            return;
        }
        // Check that the number of arguments matches the number of parameters.

        if self.infer.expr_type_mismatches().next().is_some() {
            // FIXME: Due to shortcomings in the current type system implementation, only emit
            // this diagnostic if there are no type mismatches in the containing function.
        } else if let Expr::MethodCall { receiver, .. } = expr {
            let (callee, _) = match self.infer.method_resolution(call_id) {
                Some(it) => it,
                None => return,
            };

            let checker = filter_map_next_checker
                .get_or_insert_with(|| FilterMapNextChecker::new(&self.owner.resolver(db), db));

            if checker.check(call_id, receiver, &callee).is_some() {
                self.diagnostics.push(BodyValidationDiagnostic::ReplaceFilterMapNextWithFindMap {
                    method_call_expr: call_id,
                });
            }

            let receiver_ty = self.infer[*receiver].clone();
            checker.prev_receiver_ty = Some(receiver_ty);
        }
    }

    fn validate_match(
        &mut self,
        match_expr: ExprId,
        scrutinee_expr: ExprId,
        arms: &[MatchArm],
        db: &dyn HirDatabase,
    ) {
        let scrut_ty = &self.infer[scrutinee_expr];
        if scrut_ty.contains_unknown() {
            return;
        }

        let cx = MatchCheckCtx::new(self.owner.module(db), self.owner, db, self.env.clone());

        let pattern_arena = Arena::new();
        let mut m_arms = Vec::with_capacity(arms.len());
        let mut has_lowering_errors = false;
        // Note: Skipping the entire diagnostic rather than just not including a faulty match arm is
        // preferred to avoid the chance of false positives.
        for arm in arms {
            let Some(pat_ty) = self.infer.type_of_pat.get(arm.pat) else {
                return;
            };
            if pat_ty.contains_unknown() {
                return;
            }

            // We only include patterns whose type matches the type
            // of the scrutinee expression. If we had an InvalidMatchArmPattern
            // diagnostic or similar we could raise that in an else
            // block here.
            //
            // When comparing the types, we also have to consider that rustc
            // will automatically de-reference the scrutinee expression type if
            // necessary.
            //
            // FIXME we should use the type checker for this.
            if (pat_ty == scrut_ty
                || scrut_ty
                    .as_reference()
                    .map(|(match_expr_ty, ..)| match_expr_ty == pat_ty)
                    .unwrap_or(false))
                && types_of_subpatterns_do_match(arm.pat, &self.body, &self.infer)
            {
                // If we had a NotUsefulMatchArm diagnostic, we could
                // check the usefulness of each pattern as we added it
                // to the matrix here.
                let pat = self.lower_pattern(&cx, arm.pat, db, &mut has_lowering_errors);
                let m_arm = pat_analysis::MatchArm {
                    pat: pattern_arena.alloc(pat),
                    has_guard: arm.guard.is_some(),
                    arm_data: (),
                };
                m_arms.push(m_arm);
                if !has_lowering_errors {
                    continue;
                }
            }
            // If the pattern type doesn't fit the match expression, we skip this diagnostic.
            cov_mark::hit!(validate_match_bailed_out);
            return;
        }

        let known_valid_scrutinee = Some(self.is_known_valid_scrutinee(scrutinee_expr, db));
        let report = match cx.compute_match_usefulness(
            m_arms.as_slice(),
            scrut_ty.clone(),
            known_valid_scrutinee,
        ) {
            Ok(report) => report,
            Err(()) => return,
        };

        // FIXME Report unreachable arms
        // https://github.com/rust-lang/rust/blob/f31622a50/compiler/rustc_mir_build/src/thir/pattern/check_match.rs#L200

        let witnesses = report.non_exhaustiveness_witnesses;
        if !witnesses.is_empty() {
            self.diagnostics.push(BodyValidationDiagnostic::MissingMatchArms {
                match_expr,
                uncovered_patterns: missing_match_arms(
                    &cx,
                    scrut_ty,
                    witnesses,
                    m_arms.is_empty(),
                    self.owner.krate(db),
                ),
            });
        }
    }

    // [rustc's `is_known_valid_scrutinee`](https://github.com/rust-lang/rust/blob/c9bd03cb724e13cca96ad320733046cbdb16fbbe/compiler/rustc_mir_build/src/thir/pattern/check_match.rs#L288)
    //
    // While the above function in rustc uses thir exprs, r-a doesn't have them.
    // So, the logic here is getting same result as "hir lowering + match with lowered thir"
    // with "hir only"
    fn is_known_valid_scrutinee(&self, scrutinee_expr: ExprId, db: &dyn HirDatabase) -> bool {
        if self
            .infer
            .expr_adjustments
            .get(&scrutinee_expr)
            .is_some_and(|adjusts| adjusts.iter().any(|a| matches!(a.kind, Adjust::Deref(..))))
        {
            return false;
        }

        match &self.body[scrutinee_expr] {
            Expr::UnaryOp { op: UnaryOp::Deref, .. } => false,
            Expr::Path(path) => {
                let value_or_partial = self.owner.resolver(db).resolve_path_in_value_ns_fully(
                    db,
                    path,
                    self.body.expr_path_hygiene(scrutinee_expr),
                );
                value_or_partial.is_none_or(|v| !matches!(v, ValueNs::StaticId(_)))
            }
            Expr::Field { expr, .. } => match self.infer.type_of_expr[*expr].kind(Interner) {
                TyKind::Adt(adt, ..)
                    if db.adt_datum(self.owner.krate(db), *adt).kind == AdtKind::Union =>
                {
                    false
                }
                _ => self.is_known_valid_scrutinee(*expr, db),
            },
            Expr::Index { base, .. } => self.is_known_valid_scrutinee(*base, db),
            Expr::Cast { expr, .. } => self.is_known_valid_scrutinee(*expr, db),
            Expr::Missing => false,
            _ => true,
        }
    }

    fn validate_block(&mut self, db: &dyn HirDatabase, expr: &Expr) {
        let (Expr::Block { statements, .. }
        | Expr::Async { statements, .. }
        | Expr::Unsafe { statements, .. }) = expr
        else {
            return;
        };
        let pattern_arena = Arena::new();
        let cx = MatchCheckCtx::new(self.owner.module(db), self.owner, db, self.env.clone());
        for stmt in &**statements {
            let &Statement::Let { pat, initializer, else_branch: None, .. } = stmt else {
                continue;
            };
            if self.infer.type_mismatch_for_pat(pat).is_some() {
                continue;
            }
            let Some(initializer) = initializer else { continue };
            let ty = &self.infer[initializer];
            if ty.contains_unknown() {
                continue;
            }

            let mut have_errors = false;
            let deconstructed_pat = self.lower_pattern(&cx, pat, db, &mut have_errors);

            // optimization, wildcard trivially hold
            if have_errors || matches!(deconstructed_pat.ctor(), Constructor::Wildcard) {
                continue;
            }

            let match_arm = rustc_pattern_analysis::MatchArm {
                pat: pattern_arena.alloc(deconstructed_pat),
                has_guard: false,
                arm_data: (),
            };
            let report = match cx.compute_match_usefulness(&[match_arm], ty.clone(), None) {
                Ok(v) => v,
                Err(e) => {
                    debug!(?e, "match usefulness error");
                    continue;
                }
            };
            let witnesses = report.non_exhaustiveness_witnesses;
            if !witnesses.is_empty() {
                self.diagnostics.push(BodyValidationDiagnostic::NonExhaustiveLet {
                    pat,
                    uncovered_patterns: missing_match_arms(
                        &cx,
                        ty,
                        witnesses,
                        false,
                        self.owner.krate(db),
                    ),
                });
            }
        }
    }

    fn lower_pattern<'p>(
        &self,
        cx: &MatchCheckCtx<'p>,
        pat: PatId,
        db: &dyn HirDatabase,
        have_errors: &mut bool,
    ) -> DeconstructedPat<'p> {
        let mut patcx = match_check::PatCtxt::new(db, &self.infer, &self.body);
        let pattern = patcx.lower_pattern(pat);
        let pattern = cx.lower_pat(&pattern);
        if !patcx.errors.is_empty() {
            *have_errors = true;
        }
        pattern
    }

    fn check_for_trailing_return(&mut self, body_expr: ExprId, body: &Body) {
        if !self.validate_lints {
            return;
        }
        match &body.exprs[body_expr] {
            Expr::Block { statements, tail, .. } => {
                let last_stmt = tail.or_else(|| match statements.last()? {
                    Statement::Expr { expr, .. } => Some(*expr),
                    _ => None,
                });
                if let Some(last_stmt) = last_stmt {
                    self.check_for_trailing_return(last_stmt, body);
                }
            }
            Expr::If { then_branch, else_branch, .. } => {
                self.check_for_trailing_return(*then_branch, body);
                if let Some(else_branch) = else_branch {
                    self.check_for_trailing_return(*else_branch, body);
                }
            }
            Expr::Match { arms, .. } => {
                for arm in arms.iter() {
                    let MatchArm { expr, .. } = arm;
                    self.check_for_trailing_return(*expr, body);
                }
            }
            Expr::Return { .. } => {
                self.diagnostics.push(BodyValidationDiagnostic::RemoveTrailingReturn {
                    return_expr: body_expr,
                });
            }
            _ => (),
        }
    }

    fn check_for_unnecessary_else(&mut self, id: ExprId, expr: &Expr, db: &dyn HirDatabase) {
        if !self.validate_lints {
            return;
        }
        if let Expr::If { condition: _, then_branch, else_branch } = expr {
            if else_branch.is_none() {
                return;
            }
            if let Expr::Block { statements, tail, .. } = &self.body.exprs[*then_branch] {
                let last_then_expr = tail.or_else(|| match statements.last()? {
                    Statement::Expr { expr, .. } => Some(*expr),
                    _ => None,
                });
                if let Some(last_then_expr) = last_then_expr {
                    let last_then_expr_ty = &self.infer[last_then_expr];
                    if last_then_expr_ty.is_never() {
                        // Only look at sources if the then branch diverges and we have an else branch.
                        let source_map = db.body_with_source_map(self.owner).1;
                        let Ok(source_ptr) = source_map.expr_syntax(id) else {
                            return;
                        };
                        let root = source_ptr.file_syntax(db);
                        let either::Left(ast::Expr::IfExpr(if_expr)) =
                            source_ptr.value.to_node(&root)
                        else {
                            return;
                        };
                        let mut top_if_expr = if_expr;
                        loop {
                            let parent = top_if_expr.syntax().parent();
                            let has_parent_expr_stmt_or_stmt_list =
                                parent.as_ref().is_some_and(|node| {
                                    ast::ExprStmt::can_cast(node.kind())
                                        | ast::StmtList::can_cast(node.kind())
                                });
                            if has_parent_expr_stmt_or_stmt_list {
                                // Only emit diagnostic if parent or direct ancestor is either
                                // an expr stmt or a stmt list.
                                break;
                            }
                            let Some(parent_if_expr) = parent.and_then(ast::IfExpr::cast) else {
                                // Bail if parent is neither an if expr, an expr stmt nor a stmt list.
                                return;
                            };
                            // Check parent if expr.
                            top_if_expr = parent_if_expr;
                        }

                        self.diagnostics
                            .push(BodyValidationDiagnostic::RemoveUnnecessaryElse { if_expr: id })
                    }
                }
            }
        }
    }
}

struct FilterMapNextChecker {
    filter_map_function_id: Option<hir_def::FunctionId>,
    next_function_id: Option<hir_def::FunctionId>,
    prev_filter_map_expr_id: Option<ExprId>,
    prev_receiver_ty: Option<chalk_ir::Ty<Interner>>,
}

impl FilterMapNextChecker {
    fn new(resolver: &hir_def::resolver::Resolver<'_>, db: &dyn HirDatabase) -> Self {
        // Find and store the FunctionIds for Iterator::filter_map and Iterator::next
        let (next_function_id, filter_map_function_id) = match LangItem::IteratorNext
            .resolve_function(db, resolver.krate())
        {
            Some(next_function_id) => (
                Some(next_function_id),
                match next_function_id.lookup(db).container {
                    ItemContainerId::TraitId(iterator_trait_id) => {
                        let iterator_trait_items = &db.trait_items(iterator_trait_id).items;
                        iterator_trait_items.iter().find_map(|(name, it)| match it {
                            &AssocItemId::FunctionId(id) if *name == sym::filter_map => Some(id),
                            _ => None,
                        })
                    }
                    _ => None,
                },
            ),
            None => (None, None),
        };
        Self {
            filter_map_function_id,
            next_function_id,
            prev_filter_map_expr_id: None,
            prev_receiver_ty: None,
        }
    }

    // check for instances of .filter_map(..).next()
    fn check(
        &mut self,
        current_expr_id: ExprId,
        receiver_expr_id: &ExprId,
        function_id: &hir_def::FunctionId,
    ) -> Option<()> {
        if *function_id == self.filter_map_function_id? {
            self.prev_filter_map_expr_id = Some(current_expr_id);
            return None;
        }

        if *function_id == self.next_function_id? {
            if let Some(prev_filter_map_expr_id) = self.prev_filter_map_expr_id {
                let is_dyn_trait = self
                    .prev_receiver_ty
                    .as_ref()
                    .is_some_and(|it| it.strip_references().dyn_trait().is_some());
                if *receiver_expr_id == prev_filter_map_expr_id && !is_dyn_trait {
                    return Some(());
                }
            }
        }

        self.prev_filter_map_expr_id = None;
        None
    }
}

pub fn record_literal_missing_fields(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    id: ExprId,
    expr: &Expr,
) -> Option<(VariantId, Vec<LocalFieldId>, /*exhaustive*/ bool)> {
    let (fields, exhaustive) = match expr {
        Expr::RecordLit { fields, spread, .. } => (fields, spread.is_none()),
        _ => return None,
    };

    let variant_def = infer.variant_resolution_for_expr(id)?;
    if let VariantId::UnionId(_) = variant_def {
        return None;
    }

    let variant_data = variant_def.variant_data(db);

    let specified_fields: FxHashSet<_> = fields.iter().map(|f| &f.name).collect();
    let missed_fields: Vec<LocalFieldId> = variant_data
        .fields()
        .iter()
        .filter_map(|(f, d)| if specified_fields.contains(&d.name) { None } else { Some(f) })
        .collect();
    if missed_fields.is_empty() {
        return None;
    }
    Some((variant_def, missed_fields, exhaustive))
}

pub fn record_pattern_missing_fields(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    id: PatId,
    pat: &Pat,
) -> Option<(VariantId, Vec<LocalFieldId>, /*exhaustive*/ bool)> {
    let (fields, exhaustive) = match pat {
        Pat::Record { path: _, args, ellipsis } => (args, !ellipsis),
        _ => return None,
    };

    let variant_def = infer.variant_resolution_for_pat(id)?;
    if let VariantId::UnionId(_) = variant_def {
        return None;
    }

    let variant_data = variant_def.variant_data(db);

    let specified_fields: FxHashSet<_> = fields.iter().map(|f| &f.name).collect();
    let missed_fields: Vec<LocalFieldId> = variant_data
        .fields()
        .iter()
        .filter_map(|(f, d)| if specified_fields.contains(&d.name) { None } else { Some(f) })
        .collect();
    if missed_fields.is_empty() {
        return None;
    }
    Some((variant_def, missed_fields, exhaustive))
}

fn types_of_subpatterns_do_match(pat: PatId, body: &Body, infer: &InferenceResult) -> bool {
    fn walk(pat: PatId, body: &Body, infer: &InferenceResult, has_type_mismatches: &mut bool) {
        match infer.type_mismatch_for_pat(pat) {
            Some(_) => *has_type_mismatches = true,
            None if *has_type_mismatches => (),
            None => {
                let pat = &body[pat];
                if let Pat::ConstBlock(expr) | Pat::Lit(expr) = *pat {
                    *has_type_mismatches |= infer.type_mismatch_for_expr(expr).is_some();
                    if *has_type_mismatches {
                        return;
                    }
                }
                pat.walk_child_pats(|subpat| walk(subpat, body, infer, has_type_mismatches))
            }
        }
    }

    let mut has_type_mismatches = false;
    walk(pat, body, infer, &mut has_type_mismatches);
    !has_type_mismatches
}

fn missing_match_arms<'p>(
    cx: &MatchCheckCtx<'p>,
    scrut_ty: &Ty,
    witnesses: Vec<WitnessPat<'p>>,
    arms_is_empty: bool,
    krate: Crate,
) -> String {
    struct DisplayWitness<'a, 'p>(&'a WitnessPat<'p>, &'a MatchCheckCtx<'p>, DisplayTarget);
    impl fmt::Display for DisplayWitness<'_, '_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let DisplayWitness(witness, cx, display_target) = *self;
            let pat = cx.hoist_witness_pat(witness);
            write!(f, "{}", pat.display(cx.db, display_target))
        }
    }

    let non_empty_enum = match scrut_ty.as_adt() {
        Some((AdtId::EnumId(e), _)) => !cx.db.enum_variants(e).variants.is_empty(),
        _ => false,
    };
    let display_target = DisplayTarget::from_crate(cx.db, krate);
    if arms_is_empty && !non_empty_enum {
        format!("type `{}` is non-empty", scrut_ty.display(cx.db, display_target))
    } else {
        let pat_display = |witness| DisplayWitness(witness, cx, display_target);
        const LIMIT: usize = 3;
        match &*witnesses {
            [witness] => format!("`{}` not covered", pat_display(witness)),
            [head @ .., tail] if head.len() < LIMIT => {
                let head = head.iter().map(pat_display);
                format!("`{}` and `{}` not covered", head.format("`, `"), pat_display(tail))
            }
            _ => {
                let (head, tail) = witnesses.split_at(LIMIT);
                let head = head.iter().map(pat_display);
                format!("`{}` and {} more not covered", head.format("`, `"), tail.len())
            }
        }
    }
}
