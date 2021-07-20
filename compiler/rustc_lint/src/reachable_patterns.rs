#![allow(dead_code)]

use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast::Attribute;
// use rustc_errors::{Applicability};
use rustc_hir::{Arm, Expr, ExprKind, MatchSource, Pat, PatField, PatKind, QPath};
use rustc_middle::{lint::LintDiagnosticBuilder, ty};
use rustc_span::{sym, Span};

declare_lint! {
    /// The `reachable_patterns` lint detects when patterns of non_exhaustive
    /// structs or enums are missed.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// #[non_exhaustive]
    /// pub enum Bar {
    ///     A,
    ///     B,
    /// }
    ///
    /// match Bar::A {
    ///     Bar::A => {},
    ///     #[warn(reachable)]
    ///     _ => {},
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When a struct or enum is marked as `non_exhaustive` it may still be useful to warn
    /// when there are patterns that are not considered. This lint will catch these cases.
    REACHABLE_PATTERNS,
    Allow,
    "detect when patterns of types marked `non_exhaustive` are missed",
}

declare_lint_pass!(ReachablePattern => [REACHABLE_PATTERNS]);

impl<'tcx> LateLintPass<'tcx> for ReachablePattern {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Match(
            e @ Expr { kind: ExprKind::Path(..), .. },
            arms,
            MatchSource::Normal,
        ) = &expr.kind
        {
            if !arms.is_empty() {
                // TODO: can this ever not be there?
                if let Some(ty::Adt(def, _)) =
                    cx.maybe_typeck_results().map(|r| r.expr_ty(e).kind())
                {
                    if def.is_enum() && def.is_variant_list_non_exhaustive() {
                        let missing = get_missing_variants(cx, arms, e);
                        if !missing.is_empty() {
                            let last = arms.last().unwrap().span;
                            lint(cx, expr.span, last, missing)
                        }
                    }
                }
            }
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) {
        if let PatKind::Struct(_, fields @ [_, ..], true) = &pat.kind {
            if let Some(ty::Adt(def, _)) = cx.maybe_typeck_results().map(|r| r.pat_ty(pat).kind()) {
                if let Some(var) = def.variants.iter().next() {
                    if var.is_field_list_non_exhaustive() {
                        let missing = get_missing_fields(fields, &var.fields);
                        if !missing.is_empty() {
                            let last = fields.last().unwrap().span;
                            lint(cx, pat.span, last, missing)
                        }
                    }
                }
            }
        }
    }
}

fn lint(cx: &LateContext<'_>, all: Span, _last: Span, missing: Vec<String>) {
    cx.struct_span_lint(REACHABLE_PATTERNS, all, |lint: LintDiagnosticBuilder<'_>| {
        let mut l = lint.build("missing patterns of non_exhaustive type");
        l.help(&format!("add {} to match all reachable patterns", missing.join(",")));
        l.emit();
    });
}

fn get_missing_variants<'tcx>(
    cx: &LateContext<'tcx>,
    arms: &[Arm<'_>],
    e: &'tcx Expr<'_>,
) -> Vec<String> {
    let ty = cx.typeck_results().expr_ty(e);
    let mut missing_variants = vec![];
    if let ty::Adt(def, _) = ty.kind() {
        for variant in &def.variants {
            missing_variants.push(variant);
        }
    }
    let mut has_attr = false;
    for arm in arms {
        has_attr |= has_reachable_attr(cx, arm);

        if let PatKind::Path(ref path) = arm.pat.kind {
            if let QPath::Resolved(_, p) = path {
                missing_variants.retain(|e| e.ctor_def_id != Some(p.res.def_id()));
            }
        } else if let PatKind::TupleStruct(QPath::Resolved(_, p), ref patterns, ..) = arm.pat.kind {
            let is_pattern_exhaustive =
                |pat: &&Pat<'_>| matches!(pat.kind, PatKind::Wild | PatKind::Binding(.., None));
            if patterns.iter().all(is_pattern_exhaustive) {
                missing_variants.retain(|e| e.ctor_def_id != Some(p.res.def_id()));
            }
        }
    }

    if !has_attr {
        return vec![];
    }

    let missing_variants =
        missing_variants.iter().map(|v| format!("`{}`", cx.tcx.def_path_str(v.def_id))).collect();
    missing_variants
}

fn get_missing_fields(used: &[PatField<'_>], defined: &[ty::FieldDef]) -> Vec<String> {
    let mut missing = vec![];

    for def in defined {
        if !used.iter().any(|f| f.ident == def.ident) && def.vis.is_visible_locally() {
            missing.push(format!("`{}`", def.ident.name))
        }
    }

    missing
}

fn has_reachable_attr(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    cx.tcx.hir().attrs(arm.hir_id).iter().any(|attr: &Attribute| {
        (attr.has_name(sym::warn) || attr.has_name(sym::deny))
            && attr.meta().map_or(false, |a| {
                a.meta_item_list()
                    .and_then(|s| s.first().cloned())
                    .map_or(false, |a| a.has_name(sym::reachable))
            })
    })
}
