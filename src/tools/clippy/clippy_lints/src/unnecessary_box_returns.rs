use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::approx_ty_size;
use rustc_errors::Applicability;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{FnDecl, FnRetTy, ImplItemKind, Item, ItemKind, Node, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Symbol;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for a return type containing a `Box<T>` where `T` implements `Sized`
    ///
    /// The lint ignores `Box<T>` where `T` is larger than `unnecessary_box_size`,
    /// as returning a large `T` directly may be detrimental to performance.
    ///
    /// ### Why is this bad?
    ///
    /// It's better to just return `T` in these cases. The caller may not need
    /// the value to be boxed, and it's expensive to free the memory once the
    /// `Box<T>` been dropped.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo() -> Box<String> {
    ///     Box::new(String::from("Hello, world!"))
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn foo() -> String {
    ///     String::from("Hello, world!")
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub UNNECESSARY_BOX_RETURNS,
    pedantic,
    "Needlessly returning a Box"
}

pub struct UnnecessaryBoxReturns {
    avoid_breaking_exported_api: bool,
    maximum_size: u64,
}

impl_lint_pass!(UnnecessaryBoxReturns => [UNNECESSARY_BOX_RETURNS]);

impl UnnecessaryBoxReturns {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            maximum_size: conf.unnecessary_box_size,
        }
    }

    fn check_fn_item(&self, cx: &LateContext<'_>, decl: &FnDecl<'_>, def_id: LocalDefId, name: Symbol) {
        // we don't want to tell someone to break an exported function if they ask us not to
        if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(def_id) {
            return;
        }

        // functions which contain the word "box" are exempt from this lint
        if name.as_str().contains("box") {
            return;
        }

        let FnRetTy::Return(return_ty_hir) = &decl.output else {
            return;
        };

        let return_ty = cx
            .tcx
            .instantiate_bound_regions_with_erased(cx.tcx.fn_sig(def_id).skip_binder())
            .output();

        let Some(boxed_ty) = return_ty.boxed_ty() else {
            return;
        };

        // It's sometimes useful to return Box<T> if T is unsized, so don't lint those.
        // Also, don't lint if we know that T is very large, in which case returning
        // a Box<T> may be beneficial.
        if boxed_ty.is_sized(cx.tcx, cx.typing_env()) && approx_ty_size(cx, boxed_ty) <= self.maximum_size {
            span_lint_and_then(
                cx,
                UNNECESSARY_BOX_RETURNS,
                return_ty_hir.span,
                format!("boxed return of the sized type `{boxed_ty}`"),
                |diagnostic| {
                    diagnostic.span_suggestion(
                        return_ty_hir.span,
                        "try",
                        boxed_ty.to_string(),
                        // the return value and function callers also needs to
                        // be changed, so this can't be MachineApplicable
                        Applicability::Unspecified,
                    );
                    diagnostic.help("changing this also requires a change to the return expressions in this function");
                },
            );
        }
    }
}

impl LateLintPass<'_> for UnnecessaryBoxReturns {
    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &TraitItem<'_>) {
        let TraitItemKind::Fn(signature, _) = &item.kind else {
            return;
        };
        self.check_fn_item(cx, signature.decl, item.owner_id.def_id, item.ident.name);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, item: &rustc_hir::ImplItem<'_>) {
        // Ignore implementations of traits, because the lint should be on the
        // trait, not on the implementation of it.
        let Node::Item(parent) = cx.tcx.parent_hir_node(item.hir_id()) else {
            return;
        };
        let ItemKind::Impl(parent) = parent.kind else { return };
        if parent.of_trait.is_some() {
            return;
        }

        let ImplItemKind::Fn(signature, ..) = &item.kind else {
            return;
        };
        self.check_fn_item(cx, signature.decl, item.owner_id.def_id, item.ident.name);
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        let ItemKind::Fn { ident, sig, .. } = &item.kind else {
            return;
        };
        self.check_fn_item(cx, sig.decl, item.owner_id.def_id, ident.name);
    }
}
