use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{get_parent_as_impl, has_repr_attr, is_bool};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId, Item, ItemKind, TraitFn, TraitItem, TraitItemKind, Ty};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for excessive
    /// use of bools in structs.
    ///
    /// ### Why is this bad?
    /// Excessive bools in a struct
    /// is often a sign that it's used as a state machine,
    /// which is much better implemented as an enum.
    /// If it's not the case, excessive bools usually benefit
    /// from refactoring into two-variant enums for better
    /// readability and API.
    ///
    /// ### Example
    /// ```rust
    /// struct S {
    ///     is_pending: bool,
    ///     is_processing: bool,
    ///     is_finished: bool,
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// enum S {
    ///     Pending,
    ///     Processing,
    ///     Finished,
    /// }
    /// ```
    #[clippy::version = "1.43.0"]
    pub STRUCT_EXCESSIVE_BOOLS,
    pedantic,
    "using too many bools in a struct"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for excessive use of
    /// bools in function definitions.
    ///
    /// ### Why is this bad?
    /// Calls to such functions
    /// are confusing and error prone, because it's
    /// hard to remember argument order and you have
    /// no type system support to back you up. Using
    /// two-variant enums instead of bools often makes
    /// API easier to use.
    ///
    /// ### Example
    /// ```rust,ignore
    /// fn f(is_round: bool, is_hot: bool) { ... }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// enum Shape {
    ///     Round,
    ///     Spiky,
    /// }
    ///
    /// enum Temperature {
    ///     Hot,
    ///     IceCold,
    /// }
    ///
    /// fn f(shape: Shape, temperature: Temperature) { ... }
    /// ```
    #[clippy::version = "1.43.0"]
    pub FN_PARAMS_EXCESSIVE_BOOLS,
    pedantic,
    "using too many bools in function parameters"
}

pub struct ExcessiveBools {
    max_struct_bools: u64,
    max_fn_params_bools: u64,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
enum Kind {
    Struct,
    Fn,
}

impl ExcessiveBools {
    #[must_use]
    pub fn new(max_struct_bools: u64, max_fn_params_bools: u64) -> Self {
        Self {
            max_struct_bools,
            max_fn_params_bools,
        }
    }

    fn too_many_bools<'tcx>(&self, tys: impl Iterator<Item = &'tcx Ty<'tcx>>, kind: Kind) -> bool {
        if let Ok(bools) = tys.filter(|ty| is_bool(ty)).count().try_into() {
            (if Kind::Fn == kind {
                self.max_fn_params_bools
            } else {
                self.max_struct_bools
            }) < bools
        } else {
            false
        }
    }

    fn check_fn_sig(&self, cx: &LateContext<'_>, fn_decl: &FnDecl<'_>, span: Span) {
        if !span.from_expansion() && self.too_many_bools(fn_decl.inputs.iter(), Kind::Fn) {
            span_lint_and_help(
                cx,
                FN_PARAMS_EXCESSIVE_BOOLS,
                span,
                &format!("more than {} bools in function parameters", self.max_fn_params_bools),
                None,
                "consider refactoring bools into two-variant enums",
            );
        }
    }
}

impl_lint_pass!(ExcessiveBools => [STRUCT_EXCESSIVE_BOOLS, FN_PARAMS_EXCESSIVE_BOOLS]);

impl<'tcx> LateLintPass<'tcx> for ExcessiveBools {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if item.span.from_expansion() {
            return;
        }
        if let ItemKind::Struct(variant_data, _) = &item.kind {
            if has_repr_attr(cx, item.hir_id()) {
                return;
            }

            if self.too_many_bools(variant_data.fields().iter().map(|field| field.ty), Kind::Struct) {
                span_lint_and_help(
                    cx,
                    STRUCT_EXCESSIVE_BOOLS,
                    item.span,
                    &format!("more than {} bools in a struct", self.max_struct_bools),
                    None,
                    "consider using a state machine or refactoring bools into two-variant enums",
                );
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx TraitItem<'tcx>) {
        // functions with a body are already checked by `check_fn`
        if let TraitItemKind::Fn(fn_sig, TraitFn::Required(_)) = &trait_item.kind
            && fn_sig.header.abi == Abi::Rust
            {
            self.check_fn_sig(cx, fn_sig.decl, fn_sig.span);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        span: Span,
        hir_id: HirId,
    ) {
        if let Some(fn_header) = fn_kind.header()
            && fn_header.abi == Abi::Rust
            && get_parent_as_impl(cx.tcx, hir_id)
                .map_or(true,
                    |impl_item| impl_item.of_trait.is_none()
                )
            {
            self.check_fn_sig(cx, fn_decl, span);
        }
    }
}
