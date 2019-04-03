//! Some lints that are only useful in the compiler or crates that use compiler internals, such as
//! Clippy.

use crate::hir::{HirId, Path, PathSegment, QPath, Ty, TyKind};
use crate::lint::{
    EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintArray, LintContext, LintPass,
};
use errors::Applicability;
use rustc_data_structures::fx::FxHashMap;
use syntax::ast::Ident;

declare_lint! {
    pub DEFAULT_HASH_TYPES,
    Allow,
    "forbid HashMap and HashSet and suggest the FxHash* variants"
}

pub struct DefaultHashTypes {
    map: FxHashMap<String, String>,
}

impl DefaultHashTypes {
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert("HashMap".to_string(), "FxHashMap".to_string());
        map.insert("HashSet".to_string(), "FxHashSet".to_string());
        Self { map }
    }
}

impl LintPass for DefaultHashTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(DEFAULT_HASH_TYPES)
    }

    fn name(&self) -> &'static str {
        "DefaultHashTypes"
    }
}

impl EarlyLintPass for DefaultHashTypes {
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: Ident) {
        let ident_string = ident.to_string();
        if let Some(replace) = self.map.get(&ident_string) {
            let msg = format!(
                "Prefer {} over {}, it has better performance",
                replace, ident_string
            );
            let mut db = cx.struct_span_lint(DEFAULT_HASH_TYPES, ident.span, &msg);
            db.span_suggestion(
                ident.span,
                "use",
                replace.to_string(),
                Applicability::MaybeIncorrect, // FxHashMap, ... needs another import
            );
            db.note(&format!(
                "a `use rustc_data_structures::fx::{}` may be necessary",
                replace
            ))
            .emit();
        }
    }
}

declare_lint! {
    pub USAGE_OF_TY_TYKIND,
    Allow,
    "Usage of `ty::TyKind` outside of the `ty::sty` module"
}

pub struct TyKindUsage;

impl LintPass for TyKindUsage {
    fn get_lints(&self) -> LintArray {
        lint_array!(USAGE_OF_TY_TYKIND)
    }

    fn name(&self) -> &'static str {
        "TyKindUsage"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TyKindUsage {
    fn check_path(&mut self, cx: &LateContext<'_, '_>, path: &'tcx Path, _: HirId) {
        let segments = path.segments.iter().rev().skip(1).rev();

        if let Some(last) = segments.last() {
            let span = path.span.with_hi(last.ident.span.hi());
            if lint_ty_kind_usage(cx, last) {
                cx.struct_span_lint(USAGE_OF_TY_TYKIND, span, "usage of `ty::TyKind::<kind>`")
                    .span_suggestion(
                        span,
                        "try using ty::<kind> directly",
                        "ty".to_string(),
                        Applicability::MaybeIncorrect, // ty maybe needs an import
                    )
                    .emit();
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_, '_>, ty: &'tcx Ty) {
        if let TyKind::Path(qpath) = &ty.node {
            if let QPath::Resolved(_, path) = qpath {
                if let Some(last) = path.segments.iter().last() {
                    if lint_ty_kind_usage(cx, last) {
                        cx.struct_span_lint(USAGE_OF_TY_TYKIND, path.span, "usage of `ty::TyKind`")
                            .help("try using `ty::Ty` instead")
                            .emit();
                    }
                }
            }
        }
    }
}

fn lint_ty_kind_usage(cx: &LateContext<'_, '_>, segment: &PathSegment) -> bool {
    if segment.ident.as_str() == "TyKind" {
        if let Some(def) = segment.def {
            if let Some(did) = def.opt_def_id() {
                return did.match_path(cx.tcx, &["rustc", "ty", "sty", "TyKind"]);
            }
        }
    }

    false
}
