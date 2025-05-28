use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{get_parent_as_impl, has_repr_attr, is_bool};
use rustc_abi::ExternAbi;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, Item, ItemKind, TraitFn, TraitItem, TraitItemKind, Ty};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for excessive
    /// use of bools in structs.
    ///
    /// ### Why is this bad?
    /// Excessive bools in a struct is often a sign that
    /// the type is being used to represent a state
    /// machine, which is much better implemented as an
    /// enum.
    ///
    /// The reason an enum is better for state machines
    /// over structs is that enums more easily forbid
    /// invalid states.
    ///
    /// Structs with too many booleans may benefit from refactoring
    /// into multi variant enums for better readability and API.
    ///
    /// ### Example
    /// ```no_run
    /// struct S {
    ///     is_pending: bool,
    ///     is_processing: bool,
    ///     is_finished: bool,
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
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

impl ExcessiveBools {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            max_struct_bools: conf.max_struct_bools,
            max_fn_params_bools: conf.max_fn_params_bools,
        }
    }
}

impl_lint_pass!(ExcessiveBools => [STRUCT_EXCESSIVE_BOOLS, FN_PARAMS_EXCESSIVE_BOOLS]);

fn has_n_bools<'tcx>(iter: impl Iterator<Item = &'tcx Ty<'tcx>>, mut count: u64) -> bool {
    iter.filter(|ty| is_bool(ty)).any(|_| {
        let (x, overflow) = count.overflowing_sub(1);
        count = x;
        overflow
    })
}

fn check_fn_decl(cx: &LateContext<'_>, decl: &FnDecl<'_>, sp: Span, max: u64) {
    if has_n_bools(decl.inputs.iter(), max) && !sp.from_expansion() {
        span_lint_and_help(
            cx,
            FN_PARAMS_EXCESSIVE_BOOLS,
            sp,
            format!("more than {max} bools in function parameters"),
            None,
            "consider refactoring bools into two-variant enums",
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for ExcessiveBools {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Struct(_, _, variant_data) = &item.kind
            && variant_data.fields().len() as u64 > self.max_struct_bools
            && has_n_bools(
                variant_data.fields().iter().map(|field| field.ty),
                self.max_struct_bools,
            )
            && !has_repr_attr(cx, item.hir_id())
            && !item.span.from_expansion()
        {
            span_lint_and_help(
                cx,
                STRUCT_EXCESSIVE_BOOLS,
                item.span,
                format!("more than {} bools in a struct", self.max_struct_bools),
                None,
                "consider using a state machine or refactoring bools into two-variant enums",
            );
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, trait_item: &'tcx TraitItem<'tcx>) {
        // functions with a body are already checked by `check_fn`
        if let TraitItemKind::Fn(fn_sig, TraitFn::Required(_)) = &trait_item.kind
            && fn_sig.header.abi == ExternAbi::Rust
            && fn_sig.decl.inputs.len() as u64 > self.max_fn_params_bools
        {
            check_fn_decl(cx, fn_sig.decl, fn_sig.span, self.max_fn_params_bools);
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if let Some(fn_header) = fn_kind.header()
            && fn_header.abi == ExternAbi::Rust
            && fn_decl.inputs.len() as u64 > self.max_fn_params_bools
            && get_parent_as_impl(cx.tcx, cx.tcx.local_def_id_to_hir_id(def_id))
                .is_none_or(|impl_item| impl_item.of_trait.is_none())
        {
            check_fn_decl(cx, fn_decl, span, self.max_fn_params_bools);
        }
    }
}
