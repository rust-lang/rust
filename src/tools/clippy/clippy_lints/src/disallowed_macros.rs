
use clippy_config::Conf;
use clippy_config::types::{DisallowedPath, create_disallowed_map};
use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::macros::macro_backtrace;
use clippy_utils::paths::PathNS;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{
    AmbigArg, Expr, ExprKind, ForeignItem, HirId, ImplItem, Item, ItemKind, OwnerId, Pat, Path, Stmt, TraitItem, Ty,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::{ExpnId, MacroKind, Span};

use crate::utils::attr_collector::AttrStorage;

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured macros in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// macros are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some macros are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-macros = [
    ///     # Can use a string as the path of the disallowed macro.
    ///     "std::print",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::println" },
    ///     # When using an inline table, can add a `reason` for why the macro
    ///     # is disallowed.
    ///     { path = "serde::Serialize", reason = "no serializing" },
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::invalid_macro", reason = "use alternative instead", allow-invalid = true }
    /// ]
    /// ```
    /// ```no_run
    /// use serde::Serialize;
    ///
    /// println!("warns");
    ///
    /// // The diagnostic will contain the message "no serializing"
    /// #[derive(Serialize)]
    /// struct Data {
    ///     name: String,
    ///     value: usize,
    /// }
    /// ```
    #[clippy::version = "1.66.0"]
    pub DISALLOWED_MACROS,
    style,
    "use of a disallowed macro"
}

pub struct DisallowedMacros {
    disallowed: DefIdMap<(&'static str, &'static DisallowedPath)>,
    seen: FxHashSet<ExpnId>,
    // Track the most recently seen node that can have a `derive` attribute.
    // Needed to use the correct lint level.
    derive_src: Option<OwnerId>,

    // When a macro is disallowed in an early pass, it's stored
    // and emitted during the late pass. This happens for attributes.
    earlies: AttrStorage,
}

impl DisallowedMacros {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf, earlies: AttrStorage) -> Self {
        let (disallowed, _) = create_disallowed_map(
            tcx,
            &conf.disallowed_macros,
            PathNS::Macro,
            |def_kind| matches!(def_kind, DefKind::Macro(_)),
            "macro",
            false,
        );
        Self {
            disallowed,
            seen: FxHashSet::default(),
            derive_src: None,
            earlies,
        }
    }

    fn check(&mut self, cx: &LateContext<'_>, span: Span, derive_src: Option<OwnerId>) {
        if self.disallowed.is_empty() {
            return;
        }

        for mac in macro_backtrace(span) {
            if !self.seen.insert(mac.expn) {
                return;
            }

            if let Some(&(path, disallowed_path)) = self.disallowed.get(&mac.def_id) {
                let msg = format!("use of a disallowed macro `{path}`");
                let add_note = disallowed_path.diag_amendment(mac.span);
                if matches!(mac.kind, MacroKind::Derive)
                    && let Some(derive_src) = derive_src
                {
                    span_lint_hir_and_then(
                        cx,
                        DISALLOWED_MACROS,
                        cx.tcx.local_def_id_to_hir_id(derive_src.def_id),
                        mac.span,
                        msg,
                        add_note,
                    );
                } else {
                    span_lint_and_then(cx, DISALLOWED_MACROS, mac.span, msg, add_note);
                }
            }
        }
    }
}

impl_lint_pass!(DisallowedMacros => [DISALLOWED_MACROS]);

impl LateLintPass<'_> for DisallowedMacros {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        // once we check a crate in the late pass we can emit the early pass lints
        if let Some(attr_spans) = self.earlies.clone().0.get() {
            for span in attr_spans {
                self.check(cx, *span, None);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        self.check(cx, expr.span, None);
        // `$t + $t` can have the context of $t, check also the span of the binary operator
        if let ExprKind::Binary(op, ..) = expr.kind {
            self.check(cx, op.span, None);
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        self.check(cx, stmt.span, None);
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &Ty<'_, AmbigArg>) {
        self.check(cx, ty.span, None);
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        self.check(cx, pat.span, None);
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        self.check(cx, item.span, self.derive_src);
        self.check(cx, item.vis_span, None);

        if matches!(
            item.kind,
            ItemKind::Struct(..) | ItemKind::Enum(..) | ItemKind::Union(..)
        ) && macro_backtrace(item.span).all(|m| !matches!(m.kind, MacroKind::Derive))
        {
            self.derive_src = Some(item.owner_id);
        }
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'_>, item: &ForeignItem<'_>) {
        self.check(cx, item.span, None);
        self.check(cx, item.vis_span, None);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, item: &ImplItem<'_>) {
        self.check(cx, item.span, None);
        self.check(cx, item.vis_span, None);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &TraitItem<'_>) {
        self.check(cx, item.span, None);
    }

    fn check_path(&mut self, cx: &LateContext<'_>, path: &Path<'_>, _: HirId) {
        self.check(cx, path.span, None);
    }
}
