use crate::utils::{last_path_segment, span_lint_and_help};
use rustc_hir::{
    intravisit, Body, Expr, ExprKind, FieldPat, FnDecl, HirId, LocalSource, MatchSource, Mutability, Pat, PatKind,
    QPath, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{AdtDef, FieldDef, Ty, TyKind, VariantDef};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for patterns that aren't exact representations of the types
    /// they are applied to.
    ///
    /// **Why is this bad?** It isn't bad in general. But in some contexts it can be desirable
    /// because it increases ownership hints in the code, and will guard against some changes
    /// in ownership.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// // Bad
    /// let value = &Some(Box::new(23));
    /// match value {
    ///     Some(inner) => println!("{}", inner),
    ///     None => println!("none"),
    /// }
    ///
    /// // Good
    /// let value = &Some(Box::new(23));
    /// match *value {
    ///     Some(ref inner) => println!("{}", inner),
    ///     None => println!("none"),
    /// }
    /// ```
    pub PATTERN_TYPE_MISMATCH,
    restriction,
    "type of pattern does not match the expression type"
}

declare_lint_pass!(PatternTypeMismatch => [PATTERN_TYPE_MISMATCH]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PatternTypeMismatch {
    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Local(ref local) = stmt.kind {
            if let Some(init) = &local.init {
                if let Some(init_ty) = cx.tables.node_type_opt(init.hir_id) {
                    let pat = &local.pat;
                    if in_external_macro(cx.sess(), pat.span) {
                        return;
                    }
                    let deref_possible = match local.source {
                        LocalSource::Normal => DerefPossible::Possible,
                        _ => DerefPossible::Impossible,
                    };
                    apply_lint(cx, pat, init_ty, deref_possible);
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Match(ref expr, arms, source) = expr.kind {
            match source {
                MatchSource::Normal | MatchSource::IfLetDesugar { .. } | MatchSource::WhileLetDesugar => {
                    if let Some(expr_ty) = cx.tables.node_type_opt(expr.hir_id) {
                        'pattern_checks: for arm in arms {
                            let pat = &arm.pat;
                            if in_external_macro(cx.sess(), pat.span) {
                                continue 'pattern_checks;
                            }
                            if apply_lint(cx, pat, expr_ty, DerefPossible::Possible) {
                                break 'pattern_checks;
                            }
                        }
                    }
                },
                _ => (),
            }
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        hir_id: HirId,
    ) {
        if let Some(fn_sig) = cx.tables.liberated_fn_sigs().get(hir_id) {
            for (param, ty) in body.params.iter().zip(fn_sig.inputs().iter()) {
                apply_lint(cx, &param.pat, ty, DerefPossible::Impossible);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DerefPossible {
    Possible,
    Impossible,
}

fn apply_lint<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &Pat<'_>,
    expr_ty: Ty<'tcx>,
    deref_possible: DerefPossible,
) -> bool {
    let maybe_mismatch = find_first_mismatch(cx, pat, expr_ty, Level::Top);
    if let Some((span, mutability, level)) = maybe_mismatch {
        span_lint_and_help(
            cx,
            PATTERN_TYPE_MISMATCH,
            span,
            "type of pattern does not match the expression type",
            None,
            &format!(
                "{}explicitly match against a `{}` pattern and adjust the enclosed variable bindings",
                match (deref_possible, level) {
                    (DerefPossible::Possible, Level::Top) => "use `*` to dereference the match expression or ",
                    _ => "",
                },
                match mutability {
                    Mutability::Mut => "&mut _",
                    Mutability::Not => "&_",
                },
            ),
        );
        true
    } else {
        false
    }
}

#[derive(Debug, Copy, Clone)]
enum Level {
    Top,
    Lower,
}

#[allow(rustc::usage_of_ty_tykind)]
fn find_first_mismatch<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    pat: &Pat<'_>,
    ty: Ty<'tcx>,
    level: Level,
) -> Option<(Span, Mutability, Level)> {
    if let PatKind::Ref(ref sub_pat, _) = pat.kind {
        if let TyKind::Ref(_, sub_ty, _) = ty.kind {
            return find_first_mismatch(cx, sub_pat, sub_ty, Level::Lower);
        }
    }

    if let TyKind::Ref(_, _, mutability) = ty.kind {
        if is_non_ref_pattern(&pat.kind) {
            return Some((pat.span, mutability, level));
        }
    }

    if let PatKind::Struct(ref qpath, ref field_pats, _) = pat.kind {
        if let TyKind::Adt(ref adt_def, ref substs_ref) = ty.kind {
            if let Some(variant) = get_variant(adt_def, qpath) {
                let field_defs = &variant.fields;
                return find_first_mismatch_in_struct(cx, field_pats, field_defs, substs_ref);
            }
        }
    }

    if let PatKind::TupleStruct(ref qpath, ref pats, _) = pat.kind {
        if let TyKind::Adt(ref adt_def, ref substs_ref) = ty.kind {
            if let Some(variant) = get_variant(adt_def, qpath) {
                let field_defs = &variant.fields;
                let ty_iter = field_defs.iter().map(|field_def| field_def.ty(cx.tcx, substs_ref));
                return find_first_mismatch_in_tuple(cx, pats, ty_iter);
            }
        }
    }

    if let PatKind::Tuple(ref pats, _) = pat.kind {
        if let TyKind::Tuple(..) = ty.kind {
            return find_first_mismatch_in_tuple(cx, pats, ty.tuple_fields());
        }
    }

    if let PatKind::Or(sub_pats) = pat.kind {
        for pat in sub_pats {
            let maybe_mismatch = find_first_mismatch(cx, pat, ty, level);
            if let Some(mismatch) = maybe_mismatch {
                return Some(mismatch);
            }
        }
    }

    None
}

fn get_variant<'a>(adt_def: &'a AdtDef, qpath: &QPath<'_>) -> Option<&'a VariantDef> {
    if adt_def.is_struct() {
        if let Some(variant) = adt_def.variants.iter().next() {
            return Some(variant);
        }
    }

    if adt_def.is_enum() {
        let pat_ident = last_path_segment(qpath).ident;
        for variant in &adt_def.variants {
            if variant.ident == pat_ident {
                return Some(variant);
            }
        }
    }

    None
}

fn find_first_mismatch_in_tuple<'a, 'tcx, I>(
    cx: &LateContext<'a, 'tcx>,
    pats: &[&Pat<'_>],
    ty_iter_src: I,
) -> Option<(Span, Mutability, Level)>
where
    I: IntoIterator<Item = Ty<'tcx>>,
{
    let mut field_tys = ty_iter_src.into_iter();
    'fields: for pat in pats {
        let field_ty = if let Some(ty) = field_tys.next() {
            ty
        } else {
            break 'fields;
        };

        let maybe_mismatch = find_first_mismatch(cx, pat, field_ty, Level::Lower);
        if let Some(mismatch) = maybe_mismatch {
            return Some(mismatch);
        }
    }

    None
}

fn find_first_mismatch_in_struct<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    field_pats: &[FieldPat<'_>],
    field_defs: &[FieldDef],
    substs_ref: SubstsRef<'tcx>,
) -> Option<(Span, Mutability, Level)> {
    for field_pat in field_pats {
        'definitions: for field_def in field_defs {
            if field_pat.ident == field_def.ident {
                let field_ty = field_def.ty(cx.tcx, substs_ref);
                let pat = &field_pat.pat;
                let maybe_mismatch = find_first_mismatch(cx, pat, field_ty, Level::Lower);
                if let Some(mismatch) = maybe_mismatch {
                    return Some(mismatch);
                }
                break 'definitions;
            }
        }
    }

    None
}

fn is_non_ref_pattern(pat_kind: &PatKind<'_>) -> bool {
    match pat_kind {
        PatKind::Struct(..) | PatKind::Tuple(..) | PatKind::TupleStruct(..) | PatKind::Path(..) => true,
        PatKind::Or(sub_pats) => sub_pats.iter().any(|pat| is_non_ref_pattern(&pat.kind)),
        _ => false,
    }
}
