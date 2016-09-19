use rustc::lint::*;
use rustc::hir::*;
use rustc::hir::intravisit::{Visitor, walk_expr};
use utils::{paths, match_path, span_lint};
use syntax::parse::token::InternedString;
use syntax::ast::{Name, NodeId, ItemKind, Crate as AstCrate};
use syntax::codemap::Span;
use std::collections::{HashSet, HashMap};


/// **What it does:** Checks for various things we like to keep tidy in clippy.
///
/// **Why is this bad?** We like to pretend we're an example of tidy code.
///
/// **Known problems:** None.
///
/// **Example:** Wrong ordering of the util::paths constants.
declare_lint! {
    pub CLIPPY_LINTS_INTERNAL,
    Allow,
    "various things that will negatively affect your clippy experience"
}


/// **What it does:** Ensures every lint is associated to a `LintPass`.
///
/// **Why is this bad?** The compiler only knows lints via a `LintPass`. Without
/// putting a lint to a `LintPass::get_lints()`'s return, the compiler will not
/// know the name of the lint.
///
/// **Known problems:** Only checks for lints associated using the `lint_array!`
/// macro.
///
/// **Example:**
/// ```rust
/// declare_lint! { pub LINT_1, ... }
/// declare_lint! { pub LINT_2, ... }
/// declare_lint! { pub FORGOTTEN_LINT, ... }
/// // ...
/// pub struct Pass;
/// impl LintPass for Pass {
///     fn get_lints(&self) -> LintArray {
///         lint_array![LINT_1, LINT_2]
///         // missing FORGOTTEN_LINT
///     }
/// }
/// ```
declare_lint! {
    pub LINT_WITHOUT_LINT_PASS,
    Warn,
    "declaring a lint without associating it in a LintPass"
}


#[derive(Copy, Clone)]
pub struct Clippy;

impl LintPass for Clippy {
    fn get_lints(&self) -> LintArray {
        lint_array!(CLIPPY_LINTS_INTERNAL)
    }
}

impl EarlyLintPass for Clippy {
    fn check_crate(&mut self, cx: &EarlyContext, krate: &AstCrate) {
        if let Some(utils) = krate.module.items.iter().find(|item| item.ident.name.as_str() == "utils") {
            if let ItemKind::Mod(ref utils_mod) = utils.node {
                if let Some(paths) = utils_mod.items.iter().find(|item| item.ident.name.as_str() == "paths") {
                    if let ItemKind::Mod(ref paths_mod) = paths.node {
                        let mut last_name: Option<InternedString> = None;
                        for item in &paths_mod.items {
                            let name = item.ident.name.as_str();
                            if let Some(ref last_name) = last_name {
                                if **last_name > *name {
                                    span_lint(cx,
                                              CLIPPY_LINTS_INTERNAL,
                                              item.span,
                                              "this constant should be before the previous constant due to lexical ordering",
                                    );
                                }
                            }
                            last_name = Some(name);
                        }
                    }
                }
            }
        }
    }
}



#[derive(Clone, Debug, Default)]
pub struct LintWithoutLintPass {
    declared_lints: HashMap<Name, Span>,
    registered_lints: HashSet<Name>,
}


impl LintPass for LintWithoutLintPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(LINT_WITHOUT_LINT_PASS)
    }
}


impl LateLintPass for LintWithoutLintPass {
    fn check_item(&mut self, _: &LateContext, item: &Item) {
        if let ItemStatic(ref ty, MutImmutable, ref expr) = item.node {
            if is_lint_ref_type(ty) {
                self.declared_lints.insert(item.name, item.span);
            } else if is_lint_array_type(ty) && item.vis == Visibility::Inherited && item.name.as_str() == "ARRAY" {
                let mut collector = LintCollector { output: &mut self.registered_lints };
                collector.visit_expr(expr);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext, _: &Crate) {
        for (lint_name, &lint_span) in &self.declared_lints {
            // When using the `declare_lint!` macro, the original `lint_span`'s
            // file points to "<rustc macros>".
            // `compiletest-rs` thinks that's an error in a different file and
            // just ignores it. This causes the test in compile-fail/lint_pass
            // not able to capture the error.
            // Therefore, we need to climb the macro expansion tree and find the
            // actual span that invoked `declare_lint!`:
            let lint_span = cx.sess().codemap().source_callsite(lint_span);

            if !self.registered_lints.contains(lint_name) {
                span_lint(cx,
                          LINT_WITHOUT_LINT_PASS,
                          lint_span,
                          &format!("the lint `{}` is not added to any `LintPass`", lint_name));
            }
        }
    }
}


fn is_lint_ref_type(ty: &Ty) -> bool {
    if let TyRptr(Some(_), MutTy { ty: ref inner, mutbl: MutImmutable }) = ty.node {
        if let TyPath(None, ref path) = inner.node {
            return match_path(path, &paths::LINT);
        }
    }
    false
}


fn is_lint_array_type(ty: &Ty) -> bool {
    if let TyPath(None, ref path) = ty.node {
        match_path(path, &paths::LINT_ARRAY)
    } else {
        false
    }
}

struct LintCollector<'a> {
    output: &'a mut HashSet<Name>,
}

impl<'v, 'a: 'v> Visitor<'v> for LintCollector<'a> {
    fn visit_expr(&mut self, expr: &'v Expr) {
        walk_expr(self, expr);
    }

    fn visit_path(&mut self, path: &'v Path, _: NodeId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].name);
        }
    }
}
