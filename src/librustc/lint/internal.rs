// Copyright 2012-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some lints that are only useful in the compiler or crates that use compiler internals, such as
//! Clippy.

use errors::Applicability;
use hir::{Expr, ExprKind, PatKind, Path, QPath, Ty, TyKind};
use lint::{
    EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintArray, LintContext, LintPass,
};
use rustc_data_structures::fx::FxHashMap;
use syntax::ast::Ident;

declare_lint! {
    pub DEFAULT_HASH_TYPES,
    Warn,
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
            db.span_suggestion_with_applicability(
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
    Warn,
    "Usage of `ty::TyKind` outside of the `ty::sty` module"
}

pub struct TyKindUsage;

impl LintPass for TyKindUsage {
    fn get_lints(&self) -> LintArray {
        lint_array!(USAGE_OF_TY_TYKIND)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TyKindUsage {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, expr: &'tcx Expr) {
        let qpaths = match &expr.node {
            ExprKind::Match(_, arms, _) => {
                let mut qpaths = vec![];
                for arm in arms {
                    for pat in &arm.pats {
                        match &pat.node {
                            PatKind::Path(qpath) | PatKind::TupleStruct(qpath, ..) => {
                                qpaths.push(qpath)
                            }
                            _ => (),
                        }
                    }
                }
                qpaths
            }
            ExprKind::Path(qpath) => vec![qpath],
            _ => vec![],
        };
        for qpath in qpaths {
            if let QPath::Resolved(_, path) = qpath {
                let segments_iter = path.segments.iter().rev().skip(1).rev();

                if let Some(last) = segments_iter.clone().last() {
                    if last.ident.as_str() == "TyKind" {
                        let path = Path {
                            span: path.span.with_hi(last.ident.span.hi()),
                            def: path.def,
                            segments: segments_iter.cloned().collect(),
                        };

                        if let Some(def) = last.def {
                            if def
                                .def_id()
                                .match_path(cx.tcx, &["rustc", "ty", "sty", "TyKind"])
                            {
                                cx.struct_span_lint(
                                    USAGE_OF_TY_TYKIND,
                                    path.span,
                                    "usage of `ty::TyKind::<kind>`",
                                )
                                .span_suggestion_with_applicability(
                                    path.span,
                                    "try using ty::<kind> directly",
                                    "ty".to_string(),
                                    Applicability::MaybeIncorrect, // ty maybe needs an import
                                ).emit();
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_, '_>, ty: &'tcx Ty) {
        if let TyKind::Path(qpath) = &ty.node {
            if let QPath::Resolved(_, path) = qpath {
                if let Some(last) = path.segments.iter().last() {
                    if last.ident.as_str() == "TyKind" {
                        if let Some(def) = last.def {
                            if def
                                .def_id()
                                .match_path(cx.tcx, &["rustc", "ty", "sty", "TyKind"])
                            {
                                cx.struct_span_lint(
                                    USAGE_OF_TY_TYKIND,
                                    path.span,
                                    "usage of `ty::TyKind`",
                                )
                                .help("try using `ty::Ty` instead")
                                .emit();
                            }
                        }
                    }
                }
            }
        }
    }
}
