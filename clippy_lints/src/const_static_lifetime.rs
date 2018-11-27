// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::syntax::ast::*;
use crate::utils::{in_macro, snippet, span_lint_and_then};

/// **What it does:** Checks for constants with an explicit `'static` lifetime.
///
/// **Why is this bad?** Adding `'static` to every reference can create very
/// complicated types.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// const FOO: &'static [(&'static str, &'static str, fn(&Bar) -> bool)] =
/// &[...]
/// ```
/// This code can be rewritten as
/// ```rust
///  const FOO: &[(&str, &str, fn(&Bar) -> bool)] = &[...]
/// ```
declare_clippy_lint! {
    pub CONST_STATIC_LIFETIME,
    style,
    "Using explicit `'static` lifetime for constants when elision rules would allow omitting them."
}

pub struct StaticConst;

impl LintPass for StaticConst {
    fn get_lints(&self) -> LintArray {
        lint_array!(CONST_STATIC_LIFETIME)
    }
}

impl StaticConst {
    // Recursively visit types
    fn visit_type(&mut self, ty: &Ty, cx: &EarlyContext<'_>) {
        match ty.node {
            // Be careful of nested structures (arrays and tuples)
            TyKind::Array(ref ty, _) => {
                self.visit_type(&*ty, cx);
            },
            TyKind::Tup(ref tup) => {
                for tup_ty in tup {
                    self.visit_type(&*tup_ty, cx);
                }
            },
            // This is what we are looking for !
            TyKind::Rptr(ref optional_lifetime, ref borrow_type) => {
                // Match the 'static lifetime
                if let Some(lifetime) = *optional_lifetime {
                    match borrow_type.ty.node {
                        TyKind::Path(..) | TyKind::Slice(..) | TyKind::Array(..) | TyKind::Tup(..) => {
                            if lifetime.ident.name == "'static" {
                                let snip = snippet(cx, borrow_type.ty.span, "<type>");
                                let sugg = format!("&{}", snip);
                                span_lint_and_then(
                                    cx,
                                    CONST_STATIC_LIFETIME,
                                    lifetime.ident.span,
                                    "Constants have by default a `'static` lifetime",
                                    |db| {
                                        db.span_suggestion_with_applicability(
                                            ty.span,
                                            "consider removing `'static`",
                                            sugg,
                                            Applicability::MachineApplicable, //snippet
                                        );
                                    },
                                );
                            }
                        },
                        _ => {},
                    }
                }
                self.visit_type(&*borrow_type.ty, cx);
            },
            TyKind::Slice(ref ty) => {
                self.visit_type(ty, cx);
            },
            _ => {},
        }
    }
}

impl EarlyLintPass for StaticConst {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if !in_macro(item.span) {
            // Match only constants...
            if let ItemKind::Const(ref var_type, _) = item.node {
                self.visit_type(var_type, cx);
            }
        }
    }

    // Don't check associated consts because `'static` cannot be elided on those (issue #2438)
}
