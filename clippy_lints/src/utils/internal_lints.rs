use rustc::lint::*;
use rustc::hir::*;
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use utils::{match_qpath, paths, span_lint};
use syntax::symbol::LocalInternedString;
use syntax::ast::{Crate as AstCrate, ItemKind, Name, NodeId};
use syntax::codemap::Span;
use std::collections::{HashMap, HashSet};


/// **What it does:** Checks for various things we like to keep tidy in clippy.
///
/// **Why is this bad?** We like to pretend we're an example of tidy code.
///
/// **Known problems:** None.
///
/// **Example:** Wrong ordering of the util::paths constants.
declare_clippy_lint! {
    pub CLIPPY_LINTS_INTERNAL,
    internal,
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
declare_clippy_lint! {
    pub LINT_WITHOUT_LINT_PASS,
    internal,
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
        if let Some(utils) = krate
            .module
            .items
            .iter()
            .find(|item| item.ident.name == "utils")
        {
            if let ItemKind::Mod(ref utils_mod) = utils.node {
                if let Some(paths) = utils_mod
                    .items
                    .iter()
                    .find(|item| item.ident.name == "paths")
                {
                    if let ItemKind::Mod(ref paths_mod) = paths.node {
                        let mut last_name: Option<LocalInternedString> = None;
                        for item in &paths_mod.items {
                            let name = item.ident.name.as_str();
                            if let Some(ref last_name) = last_name {
                                if **last_name > *name {
                                    span_lint(
                                        cx,
                                        CLIPPY_LINTS_INTERNAL,
                                        item.span,
                                        "this constant should be before the previous constant due to lexical \
                                         ordering",
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


impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let ItemStatic(ref ty, MutImmutable, body_id) = item.node {
            if is_lint_ref_type(ty) {
                self.declared_lints.insert(item.name, item.span);
            } else if is_lint_array_type(ty) && item.vis == Visibility::Inherited && item.name == "ARRAY" {
                let mut collector = LintCollector {
                    output: &mut self.registered_lints,
                    cx,
                };
                collector.visit_expr(&cx.tcx.hir.body(body_id).value);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'a, 'tcx>, _: &'tcx Crate) {
        for (lint_name, &lint_span) in &self.declared_lints {
            // When using the `declare_lint!` macro, the original `lint_span`'s
            // file points to "<rustc macros>".
            // `compiletest-rs` thinks that's an error in a different file and
            // just ignores it. This causes the test in compile-fail/lint_pass
            // not able to capture the error.
            // Therefore, we need to climb the macro expansion tree and find the
            // actual span that invoked `declare_lint!`:
            let lint_span = lint_span
                .ctxt()
                .outer()
                .expn_info()
                .map(|ei| ei.call_site)
                .expect("unable to get call_site");

            if !self.registered_lints.contains(lint_name) {
                span_lint(
                    cx,
                    LINT_WITHOUT_LINT_PASS,
                    lint_span,
                    &format!("the lint `{}` is not added to any `LintPass`", lint_name),
                );
            }
        }
    }
}


fn is_lint_ref_type(ty: &Ty) -> bool {
    if let TyRptr(
        _,
        MutTy {
            ty: ref inner,
            mutbl: MutImmutable,
        },
    ) = ty.node
    {
        if let TyPath(ref path) = inner.node {
            return match_qpath(path, &paths::LINT);
        }
    }
    false
}


fn is_lint_array_type(ty: &Ty) -> bool {
    if let TyPath(ref path) = ty.node {
        match_qpath(path, &paths::LINT_ARRAY)
    } else {
        false
    }
}

struct LintCollector<'a, 'tcx: 'a> {
    output: &'a mut HashSet<Name>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for LintCollector<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        walk_expr(self, expr);
    }

    fn visit_path(&mut self, path: &'tcx Path, _: NodeId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].name);
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir)
    }
}
