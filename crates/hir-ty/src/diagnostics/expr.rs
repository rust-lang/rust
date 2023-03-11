//! Various diagnostics for expressions that are collected together in one pass
//! through the body using inference results: mismatched arg counts, missing
//! fields, etc.

use std::fmt;
use std::sync::Arc;

use either::Either;
use hir_def::lang_item::LangItem;
use hir_def::{resolver::HasResolver, AdtId, AssocItemId, DefWithBodyId, HasModule};
use hir_def::{ItemContainerId, Lookup};
use hir_expand::name;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use typed_arena::Arena;

use crate::{
    db::HirDatabase,
    diagnostics::match_check::{
        self,
        deconstruct_pat::DeconstructedPat,
        usefulness::{compute_match_usefulness, MatchCheckCtx},
    },
    display::HirDisplay,
    InferenceResult, Ty, TyExt,
};

pub(crate) use hir_def::{
    body::Body,
    expr::{Expr, ExprId, MatchArm, Pat, PatId},
    LocalFieldId, VariantId,
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
}

impl BodyValidationDiagnostic {
    pub fn collect(db: &dyn HirDatabase, owner: DefWithBodyId) -> Vec<BodyValidationDiagnostic> {
        let _p = profile::span("BodyValidationDiagnostic::collect");
        let infer = db.infer(owner);
        let mut validator = ExprValidator::new(owner, infer);
        validator.validate_body(db);
        validator.diagnostics
    }
}

struct ExprValidator {
    owner: DefWithBodyId,
    infer: Arc<InferenceResult>,
    pub(super) diagnostics: Vec<BodyValidationDiagnostic>,
}

impl ExprValidator {
    fn new(owner: DefWithBodyId, infer: Arc<InferenceResult>) -> ExprValidator {
        ExprValidator { owner, infer, diagnostics: Vec::new() }
    }

    fn validate_body(&mut self, db: &dyn HirDatabase) {
        let body = db.body(self.owner);
        let mut filter_map_next_checker = None;

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
        // Check that the number of arguments matches the number of parameters.

        // FIXME: Due to shortcomings in the current type system implementation, only emit this
        // diagnostic if there are no type mismatches in the containing function.
        if self.infer.expr_type_mismatches().next().is_some() {
            return;
        }

        match expr {
            Expr::MethodCall { receiver, .. } => {
                let (callee, _) = match self.infer.method_resolution(call_id) {
                    Some(it) => it,
                    None => return,
                };

                if filter_map_next_checker
                    .get_or_insert_with(|| {
                        FilterMapNextChecker::new(&self.owner.resolver(db.upcast()), db)
                    })
                    .check(call_id, receiver, &callee)
                    .is_some()
                {
                    self.diagnostics.push(
                        BodyValidationDiagnostic::ReplaceFilterMapNextWithFindMap {
                            method_call_expr: call_id,
                        },
                    );
                }
            }
            _ => return,
        };
    }

    fn validate_match(
        &mut self,
        match_expr: ExprId,
        scrutinee_expr: ExprId,
        arms: &[MatchArm],
        db: &dyn HirDatabase,
    ) {
        let body = db.body(self.owner);

        let scrut_ty = &self.infer[scrutinee_expr];
        if scrut_ty.is_unknown() {
            return;
        }

        let pattern_arena = Arena::new();
        let cx = MatchCheckCtx::new(self.owner.module(db.upcast()), self.owner, db, &pattern_arena);

        let mut m_arms = Vec::with_capacity(arms.len());
        let mut has_lowering_errors = false;
        for arm in arms {
            if let Some(pat_ty) = self.infer.type_of_pat.get(arm.pat) {
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
                    && types_of_subpatterns_do_match(arm.pat, &body, &self.infer)
                {
                    // If we had a NotUsefulMatchArm diagnostic, we could
                    // check the usefulness of each pattern as we added it
                    // to the matrix here.
                    let m_arm = match_check::MatchArm {
                        pat: self.lower_pattern(&cx, arm.pat, db, &body, &mut has_lowering_errors),
                        has_guard: arm.guard.is_some(),
                    };
                    m_arms.push(m_arm);
                    if !has_lowering_errors {
                        continue;
                    }
                }
            }

            // If we can't resolve the type of a pattern, or the pattern type doesn't
            // fit the match expression, we skip this diagnostic. Skipping the entire
            // diagnostic rather than just not including this match arm is preferred
            // to avoid the chance of false positives.
            cov_mark::hit!(validate_match_bailed_out);
            return;
        }

        let report = compute_match_usefulness(&cx, &m_arms, scrut_ty);

        // FIXME Report unreacheble arms
        // https://github.com/rust-lang/rust/blob/f31622a50/compiler/rustc_mir_build/src/thir/pattern/check_match.rs#L200

        let witnesses = report.non_exhaustiveness_witnesses;
        if !witnesses.is_empty() {
            self.diagnostics.push(BodyValidationDiagnostic::MissingMatchArms {
                match_expr,
                uncovered_patterns: missing_match_arms(&cx, scrut_ty, witnesses, arms),
            });
        }
    }

    fn lower_pattern<'p>(
        &self,
        cx: &MatchCheckCtx<'_, 'p>,
        pat: PatId,
        db: &dyn HirDatabase,
        body: &Body,
        have_errors: &mut bool,
    ) -> &'p DeconstructedPat<'p> {
        let mut patcx = match_check::PatCtxt::new(db, &self.infer, body);
        let pattern = patcx.lower_pattern(pat);
        let pattern = cx.pattern_arena.alloc(DeconstructedPat::from_pat(cx, &pattern));
        if !patcx.errors.is_empty() {
            *have_errors = true;
        }
        pattern
    }
}

struct FilterMapNextChecker {
    filter_map_function_id: Option<hir_def::FunctionId>,
    next_function_id: Option<hir_def::FunctionId>,
    prev_filter_map_expr_id: Option<ExprId>,
}

impl FilterMapNextChecker {
    fn new(resolver: &hir_def::resolver::Resolver, db: &dyn HirDatabase) -> Self {
        // Find and store the FunctionIds for Iterator::filter_map and Iterator::next
        let (next_function_id, filter_map_function_id) = match db
            .lang_item(resolver.krate(), LangItem::IteratorNext)
            .and_then(|it| it.as_function())
        {
            Some(next_function_id) => (
                Some(next_function_id),
                match next_function_id.lookup(db.upcast()).container {
                    ItemContainerId::TraitId(iterator_trait_id) => {
                        let iterator_trait_items = &db.trait_data(iterator_trait_id).items;
                        iterator_trait_items.iter().find_map(|(name, it)| match it {
                            &AssocItemId::FunctionId(id) if *name == name![filter_map] => Some(id),
                            _ => None,
                        })
                    }
                    _ => None,
                },
            ),
            None => (None, None),
        };
        Self { filter_map_function_id, next_function_id, prev_filter_map_expr_id: None }
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
                if *receiver_expr_id == prev_filter_map_expr_id {
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
        Expr::RecordLit { fields, spread, ellipsis, is_assignee_expr, .. } => {
            let exhaustive = if *is_assignee_expr { !*ellipsis } else { spread.is_none() };
            (fields, exhaustive)
        }
        _ => return None,
    };

    let variant_def = infer.variant_resolution_for_expr(id)?;
    if let VariantId::UnionId(_) = variant_def {
        return None;
    }

    let variant_data = variant_def.variant_data(db.upcast());

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

    let variant_data = variant_def.variant_data(db.upcast());

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
            None => {
                body[pat].walk_child_pats(|subpat| walk(subpat, body, infer, has_type_mismatches))
            }
        }
    }

    let mut has_type_mismatches = false;
    walk(pat, body, infer, &mut has_type_mismatches);
    !has_type_mismatches
}

fn missing_match_arms<'p>(
    cx: &MatchCheckCtx<'_, 'p>,
    scrut_ty: &Ty,
    witnesses: Vec<DeconstructedPat<'p>>,
    arms: &[MatchArm],
) -> String {
    struct DisplayWitness<'a, 'p>(&'a DeconstructedPat<'p>, &'a MatchCheckCtx<'a, 'p>);
    impl fmt::Display for DisplayWitness<'_, '_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let DisplayWitness(witness, cx) = *self;
            let pat = witness.to_pat(cx);
            write!(f, "{}", pat.display(cx.db))
        }
    }

    let non_empty_enum = match scrut_ty.as_adt() {
        Some((AdtId::EnumId(e), _)) => !cx.db.enum_data(e).variants.is_empty(),
        _ => false,
    };
    if arms.is_empty() && !non_empty_enum {
        format!("type `{}` is non-empty", scrut_ty.display(cx.db))
    } else {
        let pat_display = |witness| DisplayWitness(witness, cx);
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
