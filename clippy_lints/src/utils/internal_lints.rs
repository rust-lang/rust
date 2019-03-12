use crate::utils::{
    match_def_path, match_type, paths, span_help_and_lint, span_lint, span_lint_and_sugg, walk_ptrs_ty,
};
use if_chain::if_chain;
use rustc::hir;
use rustc::hir::def::Def;
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use syntax::ast::{Crate as AstCrate, Ident, ItemKind, Name};
use syntax::source_map::Span;
use syntax::symbol::LocalInternedString;

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
    pub LINT_WITHOUT_LINT_PASS,
    internal,
    "declaring a lint without associating it in a LintPass"
}

declare_clippy_lint! {
    /// **What it does:** Checks for the presence of the default hash types "HashMap" or "HashSet"
    /// and recommends the FxHash* variants.
    ///
    /// **Why is this bad?** The FxHash variants have better performance
    /// and we don't need any collision prevention in clippy.
    pub DEFAULT_HASH_TYPES,
    internal,
    "forbid HashMap and HashSet and suggest the FxHash* variants"
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

#[derive(Copy, Clone)]
pub struct Clippy;

impl LintPass for Clippy {
    fn get_lints(&self) -> LintArray {
        lint_array!(CLIPPY_LINTS_INTERNAL)
    }

    fn name(&self) -> &'static str {
        "ClippyLintsInternal"
    }
}

impl EarlyLintPass for Clippy {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &AstCrate) {
        if let Some(utils) = krate.module.items.iter().find(|item| item.ident.name == "utils") {
            if let ItemKind::Mod(ref utils_mod) = utils.node {
                if let Some(paths) = utils_mod.items.iter().find(|item| item.ident.name == "paths") {
                    if let ItemKind::Mod(ref paths_mod) = paths.node {
                        let mut last_name: Option<LocalInternedString> = None;
                        for item in &paths_mod.items {
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

impl LintPass for LintWithoutLintPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(LINT_WITHOUT_LINT_PASS)
    }
    fn name(&self) -> &'static str {
        "LintWithoutLintPass"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let hir::ItemKind::Static(ref ty, MutImmutable, _) = item.node {
            if is_lint_ref_type(cx, ty) {
                self.declared_lints.insert(item.ident.name, item.span);
            }
        } else if let hir::ItemKind::Impl(.., Some(ref trait_ref), _, ref impl_item_refs) = item.node {
            if_chain! {
                if let hir::TraitRef{path, ..} = trait_ref;
                if let Def::Trait(def_id) = path.def;
                if match_def_path(cx.tcx, def_id, &paths::LINT_PASS);
                then {
                    let mut collector = LintCollector {
                        output: &mut self.registered_lints,
                        cx,
                    };
                    let body_id = cx.tcx.hir().body_owned_by(impl_item_refs[0].id.hir_id);
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
            if let Def::Struct(def_id) = cx.tables.qpath_def(path, inner.hir_id) {
                return match_def_path(cx.tcx, def_id, &paths::LINT);
            }
        }
    }

    false
}

struct LintCollector<'a, 'tcx: 'a> {
    output: &'a mut FxHashSet<Name>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for LintCollector<'a, 'tcx> {
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

pub struct DefaultHashTypes {
    map: FxHashMap<String, String>,
}

impl DefaultHashTypes {
    pub fn default() -> Self {
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
        "DefaultHashType"
    }
}

impl EarlyLintPass for DefaultHashTypes {
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: Ident) {
        let ident_string = ident.to_string();
        if let Some(replace) = self.map.get(&ident_string) {
            let msg = format!(
                "Prefer {} over {}, it has better performance \
                 and we don't need any collision prevention in clippy",
                replace, ident_string
            );
            span_lint_and_sugg(
                cx,
                DEFAULT_HASH_TYPES,
                ident.span,
                &msg,
                "use",
                replace.to_string(),
                Applicability::MaybeIncorrect, // FxHashMap, ... needs another import
            );
        }
    }
}

#[derive(Clone, Default)]
pub struct CompilerLintFunctions {
    map: FxHashMap<String, String>,
}

impl CompilerLintFunctions {
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert("span_lint".to_string(), "utils::span_lint".to_string());
        map.insert("struct_span_lint".to_string(), "utils::span_lint".to_string());
        map.insert("lint".to_string(), "utils::span_lint".to_string());
        map.insert("span_lint_note".to_string(), "utils::span_note_and_lint".to_string());
        map.insert("span_lint_help".to_string(), "utils::span_help_and_lint".to_string());
        Self { map }
    }
}

impl LintPass for CompilerLintFunctions {
    fn get_lints(&self) -> LintArray {
        lint_array!(COMPILER_LINT_FUNCTIONS)
    }

    fn name(&self) -> &'static str {
        "CompileLintFunctions"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CompilerLintFunctions {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;
            let fn_name = path.ident.as_str().to_string();
            if let Some(sugg) = self.map.get(&fn_name);
            let ty = walk_ptrs_ty(cx.tables.expr_ty(&args[0]));
            if match_type(cx, ty, &paths::EARLY_CONTEXT)
                || match_type(cx, ty, &paths::LATE_CONTEXT);
            then {
                span_help_and_lint(
                    cx,
                    COMPILER_LINT_FUNCTIONS,
                    path.ident.span,
                    "usage of a compiler lint function",
                    &format!("Please use the Clippy variant of this function: `{}`", sugg),
                );
            }
        }
    }
}
