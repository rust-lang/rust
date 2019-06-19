use crate::utils::{
    match_def_path, match_type, method_calls, paths, span_help_and_lint, span_lint, span_lint_and_sugg, walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc::hir;
use rustc::hir::def::{DefKind, Res};
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use syntax::ast::{Crate as AstCrate, ItemKind, Name};
use syntax::source_map::Span;
use syntax_pos::symbol::LocalInternedString;

declare_clippy_lint! {
    /// **What it does:** Checks for various things we like to keep tidy in clippy.
    ///
    /// **Why is this bad?** We like to pretend we're an example of tidy code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:** Wrong ordering of the util::paths constants.
    pub CLIPPY_LINTS_INTERNAL,
    internal,
    "various things that will negatively affect your clippy experience"
}

declare_clippy_lint! {
    /// **What it does:** Ensures every lint is associated to a `LintPass`.
    ///
    /// **Why is this bad?** The compiler only knows lints via a `LintPass`. Without
    /// putting a lint to a `LintPass::get_lints()`'s return, the compiler will not
    /// know the name of the lint.
    ///
    /// **Known problems:** Only checks for lints associated using the
    /// `declare_lint_pass!`, `impl_lint_pass!`, and `lint_array!` macros.
    ///
    /// **Example:**
    /// ```rust
    /// declare_lint! { pub LINT_1, ... }
    /// declare_lint! { pub LINT_2, ... }
    /// declare_lint! { pub FORGOTTEN_LINT, ... }
    /// // ...
    /// declare_lint_pass!(Pass => [LINT_1, LINT_2]);
    /// // missing FORGOTTEN_LINT
    /// ```
    pub LINT_WITHOUT_LINT_PASS,
    internal,
    "declaring a lint without associating it in a LintPass"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `cx.span_lint*` and suggests to use the `utils::*`
    /// variant of the function.
    ///
    /// **Why is this bad?** The `utils::*` variants also add a link to the Clippy documentation to the
    /// warning/error messages.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust
    /// cx.span_lint(LINT_NAME, "message");
    /// ```
    ///
    /// Good:
    /// ```rust
    /// utils::span_lint(cx, LINT_NAME, "message");
    /// ```
    pub COMPILER_LINT_FUNCTIONS,
    internal,
    "usage of the lint functions of the compiler instead of the utils::* variant"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `cx.outer().expn_info()` and suggests to use
    /// the `cx.outer_expn_info()`
    ///
    /// **Why is this bad?** `cx.outer_expn_info()` is faster and more concise.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// Bad:
    /// ```rust
    /// expr.span.ctxt().outer().expn_info()
    /// ```
    ///
    /// Good:
    /// ```rust
    /// expr.span.ctxt().outer_expn_info()
    /// ```
    pub OUTER_EXPN_INFO,
    internal,
    "using `cx.outer().expn_info()` instead of `cx.outer_expn_info()`"
}

declare_lint_pass!(ClippyLintsInternal => [CLIPPY_LINTS_INTERNAL]);

impl EarlyLintPass for ClippyLintsInternal {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &AstCrate) {
        if let Some(utils) = krate
            .module
            .items
            .iter()
            .find(|item| item.ident.name.as_str() == "utils")
        {
            if let ItemKind::Mod(ref utils_mod) = utils.node {
                if let Some(paths) = utils_mod.items.iter().find(|item| item.ident.name.as_str() == "paths") {
                    if let ItemKind::Mod(ref paths_mod) = paths.node {
                        let mut last_name: Option<LocalInternedString> = None;
                        for item in &*paths_mod.items {
                            let name = item.ident.as_str();
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
    declared_lints: FxHashMap<Name, Span>,
    registered_lints: FxHashSet<Name>,
}

impl_lint_pass!(LintWithoutLintPass => [LINT_WITHOUT_LINT_PASS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let hir::ItemKind::Static(ref ty, MutImmutable, _) = item.node {
            if is_lint_ref_type(cx, ty) {
                self.declared_lints.insert(item.ident.name, item.span);
            }
        } else if let hir::ItemKind::Impl(.., Some(ref trait_ref), _, ref impl_item_refs) = item.node {
            if_chain! {
                if let hir::TraitRef{path, ..} = trait_ref;
                if let Res::Def(DefKind::Trait, def_id) = path.res;
                if match_def_path(cx, def_id, &paths::LINT_PASS);
                then {
                    let mut collector = LintCollector {
                        output: &mut self.registered_lints,
                        cx,
                    };
                    let body_id = cx.tcx.hir().body_owned_by(
                        impl_item_refs
                            .iter()
                            .find(|iiref| iiref.ident.as_str() == "get_lints")
                            .expect("LintPass needs to implement get_lints")
                            .id.hir_id
                    );
                    collector.visit_expr(&cx.tcx.hir().body(body_id).value);
                }
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'a, 'tcx>, _: &'tcx Crate) {
        for (lint_name, &lint_span) in &self.declared_lints {
            // When using the `declare_tool_lint!` macro, the original `lint_span`'s
            // file points to "<rustc macros>".
            // `compiletest-rs` thinks that's an error in a different file and
            // just ignores it. This causes the test in compile-fail/lint_pass
            // not able to capture the error.
            // Therefore, we need to climb the macro expansion tree and find the
            // actual span that invoked `declare_tool_lint!`:
            let lint_span = lint_span
                .ctxt()
                .outer_expn_info()
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

fn is_lint_ref_type<'tcx>(cx: &LateContext<'_, 'tcx>, ty: &Ty) -> bool {
    if let TyKind::Rptr(
        _,
        MutTy {
            ty: ref inner,
            mutbl: MutImmutable,
        },
    ) = ty.node
    {
        if let TyKind::Path(ref path) = inner.node {
            if let Res::Def(DefKind::Struct, def_id) = cx.tables.qpath_res(path, inner.hir_id) {
                return match_def_path(cx, def_id, &paths::LINT);
            }
        }
    }

    false
}

struct LintCollector<'a, 'tcx> {
    output: &'a mut FxHashSet<Name>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LintCollector<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        walk_expr(self, expr);
    }

    fn visit_path(&mut self, path: &'tcx Path, _: HirId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].ident.name);
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}

#[derive(Clone, Default)]
pub struct CompilerLintFunctions {
    map: FxHashMap<&'static str, &'static str>,
}

impl CompilerLintFunctions {
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert("span_lint", "utils::span_lint");
        map.insert("struct_span_lint", "utils::span_lint");
        map.insert("lint", "utils::span_lint");
        map.insert("span_lint_note", "utils::span_note_and_lint");
        map.insert("span_lint_help", "utils::span_help_and_lint");
        Self { map }
    }
}

impl_lint_pass!(CompilerLintFunctions => [COMPILER_LINT_FUNCTIONS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CompilerLintFunctions {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;
            let fn_name = path.ident;
            if let Some(sugg) = self.map.get(&*fn_name.as_str());
            let ty = walk_ptrs_ty(cx.tables.expr_ty(&args[0]));
            if match_type(cx, ty, &paths::EARLY_CONTEXT)
                || match_type(cx, ty, &paths::LATE_CONTEXT);
            then {
                span_help_and_lint(
                    cx,
                    COMPILER_LINT_FUNCTIONS,
                    path.ident.span,
                    "usage of a compiler lint function",
                    &format!("please use the Clippy variant of this function: `{}`", sugg),
                );
            }
        }
    }
}

pub struct OuterExpnInfoPass;

impl_lint_pass!(OuterExpnInfoPass => [OUTER_EXPN_INFO]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for OuterExpnInfoPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        let (method_names, arg_lists) = method_calls(expr, 2);
        let method_names: Vec<LocalInternedString> = method_names.iter().map(|s| s.as_str()).collect();
        let method_names: Vec<&str> = method_names.iter().map(std::convert::AsRef::as_ref).collect();
        if_chain! {
            if let ["expn_info", "outer"] = method_names.as_slice();
            let args = arg_lists[1];
            if args.len() == 1;
            let self_arg = &args[0];
            let self_ty = walk_ptrs_ty(cx.tables.expr_ty(self_arg));
            if match_type(cx, self_ty, &paths::SYNTAX_CONTEXT);
            then {
                span_lint_and_sugg(
                    cx,
                    OUTER_EXPN_INFO,
                    expr.span.trim_start(self_arg.span).unwrap_or(expr.span),
                    "usage of `outer().expn_info()`",
                    "try",
                    ".outer_expn_info()".to_string(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
