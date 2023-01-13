use clippy_utils::diagnostics::span_lint;
use clippy_utils::return_ty;
use clippy_utils::ty::contains_adt_constructor;
use rustc_hir::{Impl, ImplItem, ImplItemKind, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Warns when constructors have the same name as their types.
    ///
    /// ### Why is this bad?
    /// Repeating the name of the type is redundant.
    ///
    /// ### Example
    /// ```rust,ignore
    /// struct Foo {}
    ///
    /// impl Foo {
    ///     pub fn foo() -> Foo {
    ///         Foo {}
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// struct Foo {}
    ///
    /// impl Foo {
    ///     pub fn new() -> Foo {
    ///         Foo {}
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.55.0"]
    pub SELF_NAMED_CONSTRUCTORS,
    style,
    "method should not have the same name as the type it is implemented for"
}

declare_lint_pass!(SelfNamedConstructors => [SELF_NAMED_CONSTRUCTORS]);

impl<'tcx> LateLintPass<'tcx> for SelfNamedConstructors {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        match impl_item.kind {
            ImplItemKind::Fn(ref sig, _) => {
                if sig.decl.implicit_self.has_implicit_self() {
                    return;
                }
            },
            _ => return,
        }

        let parent = cx.tcx.hir().get_parent_item(impl_item.hir_id()).def_id;
        let item = cx.tcx.hir().expect_item(parent);
        let self_ty = cx.tcx.type_of(item.owner_id);
        let ret_ty = return_ty(cx, impl_item.hir_id());

        // Do not check trait impls
        if matches!(item.kind, ItemKind::Impl(Impl { of_trait: Some(_), .. })) {
            return;
        }

        // Ensure method is constructor-like
        if let Some(self_adt) = self_ty.ty_adt_def() {
            if !contains_adt_constructor(ret_ty, self_adt) {
                return;
            }
        } else if !ret_ty.contains(self_ty) {
            return;
        }

        if_chain! {
            if let Some(self_def) = self_ty.ty_adt_def();
            if let Some(self_local_did) = self_def.did().as_local();
            let self_id = cx.tcx.hir().local_def_id_to_hir_id(self_local_did);
            if let Some(Node::Item(x)) = cx.tcx.hir().find(self_id);
            let type_name = x.ident.name.as_str().to_lowercase();
            if impl_item.ident.name.as_str() == type_name || impl_item.ident.name.as_str().replace('_', "") == type_name;

            then {
                span_lint(
                    cx,
                    SELF_NAMED_CONSTRUCTORS,
                    impl_item.span,
                    &format!("constructor `{}` has the same name as the type", impl_item.ident.name),
                );
            }
        }
    }
}
