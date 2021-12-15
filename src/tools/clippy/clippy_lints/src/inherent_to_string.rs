use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use clippy_utils::{get_trait_def_id, paths, return_ty, trait_ref_of_method};
use if_chain::if_chain;
use rustc_hir::{ImplItem, ImplItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the definition of inherent methods with a signature of `to_string(&self) -> String`.
    ///
    /// ### Why is this bad?
    /// This method is also implicitly defined if a type implements the `Display` trait. As the functionality of `Display` is much more versatile, it should be preferred.
    ///
    /// ### Known problems
    /// None
    ///
    /// ### Example
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
    #[clippy::version = "1.38.0"]
    pub INHERENT_TO_STRING,
    style,
    "type implements inherent method `to_string()`, but should instead implement the `Display` trait"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the definition of inherent methods with a signature of `to_string(&self) -> String` and if the type implementing this method also implements the `Display` trait.
    ///
    /// ### Why is this bad?
    /// This method is also implicitly defined if a type implements the `Display` trait. The less versatile inherent method will then shadow the implementation introduced by `Display`.
    ///
    /// ### Known problems
    /// None
    ///
    /// ### Example
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
    #[clippy::version = "1.38.0"]
    pub INHERENT_TO_STRING_SHADOW_DISPLAY,
    correctness,
    "type implements inherent method `to_string()`, which gets shadowed by the implementation of the `Display` trait"
}

declare_lint_pass!(InherentToString => [INHERENT_TO_STRING, INHERENT_TO_STRING_SHADOW_DISPLAY]);

impl<'tcx> LateLintPass<'tcx> for InherentToString {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        if impl_item.span.from_expansion() {
            return;
        }

        if_chain! {
            // Check if item is a method, called to_string and has a parameter 'self'
            if let ImplItemKind::Fn(ref signature, _) = impl_item.kind;
            if impl_item.ident.name.as_str() == "to_string";
            let decl = &signature.decl;
            if decl.implicit_self.has_implicit_self();
            if decl.inputs.len() == 1;
            if impl_item.generics.params.is_empty();

            // Check if return type is String
            if is_type_diagnostic_item(cx, return_ty(cx, impl_item.hir_id()), sym::String);

            // Filters instances of to_string which are required by a trait
            if trait_ref_of_method(cx, impl_item.hir_id()).is_none();

            then {
                show_lint(cx, impl_item);
            }
        }
    }
}

fn show_lint(cx: &LateContext<'_>, item: &ImplItem<'_>) {
    let display_trait_id = get_trait_def_id(cx, &paths::DISPLAY_TRAIT).expect("Failed to get trait ID of `Display`!");

    // Get the real type of 'self'
    let self_type = cx.tcx.fn_sig(item.def_id).input(0);
    let self_type = self_type.skip_binder().peel_refs();

    // Emit either a warning or an error
    if implements_trait(cx, self_type, display_trait_id, &[]) {
        span_lint_and_help(
            cx,
            INHERENT_TO_STRING_SHADOW_DISPLAY,
            item.span,
            &format!(
                "type `{}` implements inherent method `to_string(&self) -> String` which shadows the implementation of `Display`",
                self_type
            ),
            None,
            &format!("remove the inherent method from type `{}`", self_type),
        );
    } else {
        span_lint_and_help(
            cx,
            INHERENT_TO_STRING,
            item.span,
            &format!(
                "implementation of inherent method `to_string(&self) -> String` for type `{}`",
                self_type
            ),
            None,
            &format!("implement trait `Display` for type `{}` instead", self_type),
        );
    }
}
