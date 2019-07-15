use if_chain::if_chain;
use rustc::hir::{ImplItem, ImplItemKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

use crate::utils::{
    get_trait_def_id, implements_trait, in_macro_or_desugar, match_type, paths, return_ty, span_help_and_lint,
    trait_ref_of_method, walk_ptrs_ty,
};

declare_clippy_lint! {
    /// **What id does:** Checks for the definition of inherent methods with a signature of `to_string(&self) -> String`.
    ///
    /// **Why is this bad?** This method is also implicitly defined if a type implements the `Display` trait. As the functionality of `Display` is much more versatile, it should be preferred.
    ///
    /// **Known problems:** None
    ///
    /// ** Example:**
    ///
    /// ```rust
    /// // Bad
    /// pub struct A;
    ///
    /// impl A {
    ///     pub fn to_string(&self) -> String {
    ///         "I am A".to_string()
    ///     }
    /// }
    /// ```
    ///
    /// ```rust
    /// // Good
    /// use std::fmt;
    ///
    /// pub struct A;
    ///
    /// impl fmt::Display for A {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "I am A")
    ///     }
    /// }
    /// ```
    pub INHERENT_TO_STRING,
    style,
    "type implements inherent method `to_string()`, but should instead implement the `Display` trait"
}

declare_clippy_lint! {
    /// **What id does:** Checks for the definition of inherent methods with a signature of `to_string(&self) -> String` and if the type implementing this method also implements the `Display` trait.
    ///
    /// **Why is this bad?** This method is also implicitly defined if a type implements the `Display` trait. The less versatile inherent method will then shadow the implementation introduced by `Display`.
    ///
    /// **Known problems:** None
    ///
    /// ** Example:**
    ///
    /// ```rust
    /// // Bad
    /// use std::fmt;
    ///
    /// pub struct A;
    ///
    /// impl A {
    ///     pub fn to_string(&self) -> String {
    ///         "I am A".to_string()
    ///     }
    /// }
    ///
    /// impl fmt::Display for A {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "I am A, too")
    ///     }
    /// }
    /// ```
    ///
    /// ```rust
    /// // Good
    /// use std::fmt;
    ///
    /// pub struct A;
    ///
    /// impl fmt::Display for A {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    ///         write!(f, "I am A")
    ///     }
    /// }
    /// ```
    pub INHERENT_TO_STRING_SHADOW_DISPLAY,
    correctness,
    "type implements inherent method `to_string()`, which gets shadowed by the implementation of the `Display` trait "
}

declare_lint_pass!(InherentToString => [INHERENT_TO_STRING, INHERENT_TO_STRING_SHADOW_DISPLAY]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for InherentToString {
    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, impl_item: &'tcx ImplItem) {
        if in_macro_or_desugar(impl_item.span) {
            return;
        }

        if_chain! {
            // Check if item is a method, called to_string and has a parameter 'self'
            if let ImplItemKind::Method(ref signature, _) = impl_item.node;
            if impl_item.ident.name.as_str() == "to_string";
            let decl = &signature.decl;
            if decl.implicit_self.has_implicit_self();

            // Check if return type is String
            if match_type(cx, return_ty(cx, impl_item.hir_id), &paths::STRING);

            // Filters instances of to_string which are required by a trait
            if trait_ref_of_method(cx, impl_item.hir_id).is_none();

            then {
                show_lint(cx, impl_item);
            }
        }
    }
}

fn show_lint(cx: &LateContext<'_, '_>, item: &ImplItem) {
    let display_trait_id =
        get_trait_def_id(cx, &["core", "fmt", "Display"]).expect("Failed to get trait ID of `Display`!");

    // Get the real type of 'self'
    let fn_def_id = cx.tcx.hir().local_def_id(item.hir_id);
    let self_type = cx.tcx.fn_sig(fn_def_id).input(0);
    let self_type = walk_ptrs_ty(self_type.skip_binder());

    // Emit either a warning or an error
    if implements_trait(cx, self_type, display_trait_id, &[]) {
        span_help_and_lint(
            cx,
            INHERENT_TO_STRING_SHADOW_DISPLAY,
            item.span,
            &format!(
                "type `{}` implements inherent method `to_string(&self) -> String` which shadows the implementation of `Display`",
                self_type.to_string()
            ),
            &format!("remove the inherent method from type `{}`", self_type.to_string())
        );
    } else {
        span_help_and_lint(
            cx,
            INHERENT_TO_STRING,
            item.span,
            &format!(
                "implementation of inherent method `to_string(&self) -> String` for type `{}`",
                self_type.to_string()
            ),
            &format!("implement trait `Display` for type `{}` instead", self_type.to_string()),
        );
    }
}
