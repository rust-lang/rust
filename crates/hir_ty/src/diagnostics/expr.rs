//! Various diagnostics for expressions that are collected together in one pass
//! through the body using inference results: mismatched arg counts, missing
//! fields, etc.

use std::sync::Arc;

use hir_def::{
    expr::Statement, path::path, resolver::HasResolver, type_ref::Mutability, AssocItemId,
    DefWithBodyId, HasModule,
};
use hir_expand::name;
use itertools::Either;
use rustc_hash::FxHashSet;
use typed_arena::Arena;

use crate::{
    db::HirDatabase,
    diagnostics::match_check::{
        self,
        deconstruct_pat::DeconstructedPat,
        usefulness::{compute_match_usefulness, MatchCheckCtx},
    },
    AdtId, InferenceResult, Interner, Ty, TyExt, TyKind,
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
    MismatchedArgCount {
        call_expr: ExprId,
        expected: usize,
        found: usize,
    },
    RemoveThisSemicolon {
        expr: ExprId,
    },
    MissingOkOrSomeInTailExpr {
        expr: ExprId,
        required: String,
    },
    MissingMatchArms {
        match_expr: ExprId,
    },
    AddReferenceHere {
        arg_expr: ExprId,
        mutability: Mutability,
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
                    self.validate_match(id, *expr, arms, db, self.infer.clone());
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
        let body_expr = &body[body.body_expr];
        if let Expr::Block { statements, tail, .. } = body_expr {
            if let Some(t) = tail {
                self.validate_results_in_tail_expr(body.body_expr, *t, db);
            } else if let Some(Statement::Expr { expr: id, .. }) = statements.last() {
                self.validate_missing_tail_expr(body.body_expr, *id);
            }
        }

        let infer = &self.infer;
        let diagnostics = &mut self.diagnostics;

        infer
            .expr_type_mismatches()
            .filter_map(|(expr, mismatch)| {
                let (expr_without_ref, mutability) =
                    check_missing_refs(infer, expr, &mismatch.expected)?;

                Some((expr_without_ref, mutability))
            })
            .for_each(|(arg_expr, mutability)| {
                diagnostics
                    .push(BodyValidationDiagnostic::AddReferenceHere { arg_expr, mutability });
            });
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

        let is_method_call = matches!(expr, Expr::MethodCall { .. });
        let (sig, mut arg_count) = match expr {
            Expr::Call { callee, args } => {
                let callee = &self.infer.type_of_expr[*callee];
                let sig = match callee.callable_sig(db) {
                    Some(sig) => sig,
                    None => return,
                };
                (sig, args.len())
            }
            Expr::MethodCall { receiver, args, .. } => {
                let (callee, subst) = match self.infer.method_resolution(call_id) {
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
                let receiver = &self.infer.type_of_expr[*receiver];
                if receiver.strip_references().is_unknown() {
                    // if the receiver is of unknown type, it's very likely we
                    // don't know enough to correctly resolve the method call.
                    // This is kind of a band-aid for #6975.
                    return;
                }

                let sig = db.callable_item_signature(callee.into()).substitute(Interner, &subst);

                (sig, args.len() + 1)
            }
            _ => return,
        };

        if sig.is_varargs {
            return;
        }

        if sig.legacy_const_generics_indices.is_empty() {
            let mut param_count = sig.params().len();

            if arg_count != param_count {
                if is_method_call {
                    param_count -= 1;
                    arg_count -= 1;
                }
                self.diagnostics.push(BodyValidationDiagnostic::MismatchedArgCount {
                    call_expr: call_id,
                    expected: param_count,
                    found: arg_count,
                });
            }
        } else {
            // With `#[rustc_legacy_const_generics]` there are basically two parameter counts that
            // are allowed.
            let count_non_legacy = sig.params().len();
            let count_legacy = sig.params().len() + sig.legacy_const_generics_indices.len();
            if arg_count != count_non_legacy && arg_count != count_legacy {
                self.diagnostics.push(BodyValidationDiagnostic::MismatchedArgCount {
                    call_expr: call_id,
                    // Since most users will use the legacy way to call them, report against that.
                    expected: count_legacy,
                    found: arg_count,
                });
            }
        }
    }

    fn validate_match(
        &mut self,
        id: ExprId,
        match_expr: ExprId,
        arms: &[MatchArm],
        db: &dyn HirDatabase,
        infer: Arc<InferenceResult>,
    ) {
        let body = db.body(self.owner);

        let match_expr_ty = &infer[match_expr];
        if match_expr_ty.is_unknown() {
            return;
        }

        let pattern_arena = Arena::new();
        let cx = MatchCheckCtx {
            module: self.owner.module(db.upcast()),
            body: self.owner,
            db,
            pattern_arena: &pattern_arena,
        };

        let mut m_arms = Vec::with_capacity(arms.len());
        let mut has_lowering_errors = false;
        for arm in arms {
            if let Some(pat_ty) = infer.type_of_pat.get(arm.pat) {
                // We only include patterns whose type matches the type
                // of the match expression. If we had an InvalidMatchArmPattern
                // diagnostic or similar we could raise that in an else
                // block here.
                //
                // When comparing the types, we also have to consider that rustc
                // will automatically de-reference the match expression type if
                // necessary.
                //
                // FIXME we should use the type checker for this.
                if (pat_ty == match_expr_ty
                    || match_expr_ty
                        .as_reference()
                        .map(|(match_expr_ty, ..)| match_expr_ty == pat_ty)
                        .unwrap_or(false))
                    && types_of_subpatterns_do_match(arm.pat, &body, &infer)
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

        let report = compute_match_usefulness(&cx, &m_arms, match_expr_ty);

        // FIXME Report unreacheble arms
        // https://github.com/rust-lang/rust/blob/f31622a50/compiler/rustc_mir_build/src/thir/pattern/check_match.rs#L200

        let witnesses = report.non_exhaustiveness_witnesses;
        // FIXME Report witnesses
        // eprintln!("compute_match_usefulness(..) -> {:?}", &witnesses);
        if !witnesses.is_empty() {
            self.diagnostics.push(BodyValidationDiagnostic::MissingMatchArms { match_expr: id });
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

    fn validate_results_in_tail_expr(&mut self, body_id: ExprId, id: ExprId, db: &dyn HirDatabase) {
        // the mismatch will be on the whole block currently
        let mismatch = match self.infer.type_mismatch_for_expr(body_id) {
            Some(m) => m,
            None => return,
        };

        let core_result_path = path![core::result::Result];
        let core_option_path = path![core::option::Option];

        let resolver = self.owner.resolver(db.upcast());
        let core_result_enum = match resolver.resolve_known_enum(db.upcast(), &core_result_path) {
            Some(it) => it,
            _ => return,
        };
        let core_option_enum = match resolver.resolve_known_enum(db.upcast(), &core_option_path) {
            Some(it) => it,
            _ => return,
        };

        let (params, required) = match mismatch.expected.kind(Interner) {
            TyKind::Adt(AdtId(hir_def::AdtId::EnumId(enum_id)), parameters)
                if *enum_id == core_result_enum =>
            {
                (parameters, "Ok".to_string())
            }
            TyKind::Adt(AdtId(hir_def::AdtId::EnumId(enum_id)), parameters)
                if *enum_id == core_option_enum =>
            {
                (parameters, "Some".to_string())
            }
            _ => return,
        };

        if params.len(Interner) > 0 && params.at(Interner, 0).ty(Interner) == Some(&mismatch.actual)
        {
            self.diagnostics
                .push(BodyValidationDiagnostic::MissingOkOrSomeInTailExpr { expr: id, required });
        }
    }

    fn validate_missing_tail_expr(&mut self, body_id: ExprId, possible_tail_id: ExprId) {
        let mismatch = match self.infer.type_mismatch_for_expr(body_id) {
            Some(m) => m,
            None => return,
        };

        let possible_tail_ty = match self.infer.type_of_expr.get(possible_tail_id) {
            Some(ty) => ty,
            None => return,
        };

        if !mismatch.actual.is_unit() || mismatch.expected != *possible_tail_ty {
            return;
        }

        self.diagnostics
            .push(BodyValidationDiagnostic::RemoveThisSemicolon { expr: possible_tail_id });
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
        let iterator_path = path![core::iter::Iterator];
        let mut filter_map_function_id = None;
        let mut next_function_id = None;

        if let Some(iterator_trait_id) = resolver.resolve_known_trait(db.upcast(), &iterator_path) {
            let iterator_trait_items = &db.trait_data(iterator_trait_id).items;
            for item in iterator_trait_items.iter() {
                if let (name, AssocItemId::FunctionId(id)) = item {
                    if *name == name![filter_map] {
                        filter_map_function_id = Some(*id);
                    }
                    if *name == name![next] {
                        next_function_id = Some(*id);
                    }
                }
                if filter_map_function_id.is_some() && next_function_id.is_some() {
                    break;
                }
            }
        }
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
        Expr::RecordLit { path: _, fields, spread } => (fields, spread.is_none()),
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

fn check_missing_refs(
    infer: &InferenceResult,
    arg: ExprId,
    param: &Ty,
) -> Option<(ExprId, Mutability)> {
    let arg_ty = infer.type_of_expr.get(arg)?;

    let reference_one = arg_ty.as_reference();
    let reference_two = param.as_reference();

    match (reference_one, reference_two) {
        (None, Some((referenced_ty, _, mutability))) if referenced_ty == arg_ty => {
            Some((arg, Mutability::from_mutable(matches!(mutability, chalk_ir::Mutability::Mut))))
        }
        (None, Some((referenced_ty, _, mutability))) => match referenced_ty.kind(Interner) {
            TyKind::Slice(subst) if matches!(arg_ty.kind(Interner), TyKind::Array(arr_subst, _) if arr_subst == subst) => {
                Some((
                    arg,
                    Mutability::from_mutable(matches!(mutability, chalk_ir::Mutability::Mut)),
                ))
            }
            _ => None,
        },
        _ => None,
    }
}
