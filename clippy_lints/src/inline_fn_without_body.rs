// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! checks for `#[inline]` on trait methods without bodies

use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::syntax::ast::{Attribute, Name};
use crate::utils::span_lint_and_then;
use crate::utils::sugg::DiagnosticBuilderExt;

/// **What it does:** Checks for `#[inline]` on trait methods without bodies
///
/// **Why is this bad?** Only implementations of trait methods may be inlined.
/// The inline attribute is ignored for trait methods without bodies.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// trait Animal {
///     #[inline]
///     fn name(&self) -> &'static str;
/// }
/// ```
declare_clippy_lint! {
    pub INLINE_FN_WITHOUT_BODY,
    correctness,
    "use of `#[inline]` on trait methods without bodies"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INLINE_FN_WITHOUT_BODY)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if let TraitItemKind::Method(_, TraitMethod::Required(_)) = item.node {
            check_attrs(cx, item.ident.name, &item.attrs);
        }
    }
}

fn check_attrs(cx: &LateContext<'_, '_>, name: Name, attrs: &[Attribute]) {
    for attr in attrs {
        if attr.name() != "inline" {
            continue;
        }

        span_lint_and_then(
            cx,
            INLINE_FN_WITHOUT_BODY,
            attr.span,
            &format!("use of `#[inline]` on trait method `{}` which has no body", name),
            |db| {
                db.suggest_remove_item(cx, attr.span, "remove", Applicability::MachineApplicable);
            },
        );
    }
}
