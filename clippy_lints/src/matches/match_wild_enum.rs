use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_refutable, peel_hir_pat_refs, recurse_or_patterns};
use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::{Arm, Expr, PatKind, PathSegment, QPath, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, VariantDef};
use rustc_span::sym;

use super::{MATCH_WILDCARD_FOR_SINGLE_VARIANTS, WILDCARD_ENUM_MATCH_ARM};

#[expect(clippy::too_many_lines)]
pub(crate) fn check(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>]) {
    let ty = cx.typeck_results().expr_ty(ex).peel_refs();
    let adt_def = match ty.kind() {
        ty::Adt(adt_def, _)
            if adt_def.is_enum()
                && !(is_type_diagnostic_item(cx, ty, sym::Option) || is_type_diagnostic_item(cx, ty, sym::Result)) =>
        {
            adt_def
        },
        _ => return,
    };

    // First pass - check for violation, but don't do much book-keeping because this is hopefully
    // the uncommon case, and the book-keeping is slightly expensive.
    let mut wildcard_span = None;
    let mut wildcard_ident = None;
    let mut has_non_wild = false;
    for arm in arms {
        match peel_hir_pat_refs(arm.pat).0.kind {
            PatKind::Wild if arm.guard.is_none() => wildcard_span = Some(arm.pat.span),
            PatKind::Binding(_, _, ident, None) => {
                wildcard_span = Some(arm.pat.span);
                wildcard_ident = Some(ident);
            },
            _ => has_non_wild = true,
        }
    }
    let wildcard_span = match wildcard_span {
        Some(x) if has_non_wild => x,
        _ => return,
    };

    // Accumulate the variants which should be put in place of the wildcard because they're not
    // already covered.
    let has_hidden_external = adt_def.variants().iter().any(|x| is_hidden_and_external(cx, x));
    let mut missing_variants: Vec<_> = adt_def
        .variants()
        .iter()
        .filter(|x| !is_hidden_and_external(cx, x))
        .collect();

    let mut path_prefix = CommonPrefixSearcher::None;
    for arm in arms {
        // Guards mean that this case probably isn't exhaustively covered. Technically
        // this is incorrect, as we should really check whether each variant is exhaustively
        // covered by the set of guards that cover it, but that's really hard to do.
        recurse_or_patterns(arm.pat, |pat| {
            let path = match &peel_hir_pat_refs(pat).0.kind {
                PatKind::Path(path) => {
                    let id = match cx.qpath_res(path, pat.hir_id) {
                        Res::Def(
                            DefKind::Const | DefKind::ConstParam | DefKind::AnonConst | DefKind::InlineConst,
                            _,
                        ) => return,
                        Res::Def(_, id) => id,
                        _ => return,
                    };
                    if arm.guard.is_none() {
                        missing_variants.retain(|e| e.ctor_def_id() != Some(id));
                    }
                    path
                },
                PatKind::TupleStruct(path, patterns, ..) => {
                    if let Some(id) = cx.qpath_res(path, pat.hir_id).opt_def_id() {
                        if arm.guard.is_none() && patterns.iter().all(|p| !is_refutable(cx, p)) {
                            missing_variants.retain(|e| e.ctor_def_id() != Some(id));
                        }
                    }
                    path
                },
                PatKind::Struct(path, patterns, ..) => {
                    if let Some(id) = cx.qpath_res(path, pat.hir_id).opt_def_id() {
                        if arm.guard.is_none() && patterns.iter().all(|p| !is_refutable(cx, p.pat)) {
                            missing_variants.retain(|e| e.def_id != id);
                        }
                    }
                    path
                },
                _ => return,
            };
            match path {
                QPath::Resolved(_, path) => path_prefix.with_path(path.segments),
                QPath::TypeRelative(
                    Ty {
                        kind: TyKind::Path(QPath::Resolved(_, path)),
                        ..
                    },
                    _,
                ) => path_prefix.with_prefix(path.segments),
                _ => (),
            }
        });
    }

    let format_suggestion = |variant: &VariantDef| {
        format!(
            "{}{}{}{}",
            if let Some(ident) = wildcard_ident {
                format!("{} @ ", ident.name)
            } else {
                String::new()
            },
            if let CommonPrefixSearcher::Path(path_prefix) = path_prefix {
                let mut s = String::new();
                for seg in path_prefix {
                    s.push_str(seg.ident.as_str());
                    s.push_str("::");
                }
                s
            } else {
                let mut s = cx.tcx.def_path_str(adt_def.did());
                s.push_str("::");
                s
            },
            variant.name,
            match variant.ctor_kind() {
                Some(CtorKind::Fn) if variant.fields.len() == 1 => "(_)",
                Some(CtorKind::Fn) => "(..)",
                Some(CtorKind::Const) => "",
                None => "{ .. }",
            }
        )
    };

    match missing_variants.as_slice() {
        [] => (),
        [x] if !adt_def.is_variant_list_non_exhaustive() && !has_hidden_external => span_lint_and_sugg(
            cx,
            MATCH_WILDCARD_FOR_SINGLE_VARIANTS,
            wildcard_span,
            "wildcard matches only a single variant and will also match any future added variants",
            "try this",
            format_suggestion(x),
            Applicability::MaybeIncorrect,
        ),
        variants => {
            let mut suggestions: Vec<_> = variants.iter().copied().map(format_suggestion).collect();
            let message = if adt_def.is_variant_list_non_exhaustive() || has_hidden_external {
                suggestions.push("_".into());
                "wildcard matches known variants and will also match future added variants"
            } else {
                "wildcard match will also match any future added variants"
            };

            span_lint_and_sugg(
                cx,
                WILDCARD_ENUM_MATCH_ARM,
                wildcard_span,
                message,
                "try this",
                suggestions.join(" | "),
                Applicability::MaybeIncorrect,
            );
        },
    };
}

enum CommonPrefixSearcher<'a> {
    None,
    Path(&'a [PathSegment<'a>]),
    Mixed,
}
impl<'a> CommonPrefixSearcher<'a> {
    fn with_path(&mut self, path: &'a [PathSegment<'a>]) {
        match path {
            [path @ .., _] => self.with_prefix(path),
            [] => (),
        }
    }

    fn with_prefix(&mut self, path: &'a [PathSegment<'a>]) {
        match self {
            Self::None => *self = Self::Path(path),
            Self::Path(self_path)
                if path
                    .iter()
                    .map(|p| p.ident.name)
                    .eq(self_path.iter().map(|p| p.ident.name)) => {},
            Self::Path(_) => *self = Self::Mixed,
            Self::Mixed => (),
        }
    }
}

fn is_hidden_and_external(cx: &LateContext<'_>, variant_def: &VariantDef) -> bool {
    (cx.tcx.is_doc_hidden(variant_def.def_id) || cx.tcx.has_attr(variant_def.def_id, sym::unstable))
        && variant_def.def_id.as_local().is_none()
}
