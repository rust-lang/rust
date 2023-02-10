use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::trait_ref_of_method;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::MultiSpan;
use rustc_hir::intravisit::{walk_impl_item, walk_item, walk_param_bound, walk_ty, Visitor};
use rustc_hir::{
    GenericParamKind, Generics, ImplItem, ImplItemKind, Item, ItemKind, PredicateOrigin, Ty, TyKind, WherePredicate,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{def_id::DefId, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for type parameters in generics that are never used anywhere else.
    ///
    /// ### Why is this bad?
    /// Functions cannot infer the value of unused type parameters; therefore, calling them
    /// requires using a turbofish, which serves no purpose but to satisfy the compiler.
    ///
    /// ### Example
    /// ```rust
    /// // unused type parameters
    /// fn unused_ty<T>(x: u8) {
    ///     // ..
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn no_unused_ty(x: u8) {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "1.69.0"]
    pub EXTRA_UNUSED_TYPE_PARAMETERS,
    complexity,
    "unused type parameters in function definitions"
}
declare_lint_pass!(ExtraUnusedTypeParameters => [EXTRA_UNUSED_TYPE_PARAMETERS]);

/// A visitor struct that walks a given function and gathers generic type parameters, plus any
/// trait bounds those parameters have.
struct TypeWalker<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    /// Collection of all the type parameters and their spans.
    ty_params: FxHashMap<DefId, Span>,
    /// Collection of any (inline) trait bounds corresponding to each type parameter.
    bounds: FxHashMap<DefId, Span>,
    /// The entire `Generics` object of the function, useful for querying purposes.
    generics: &'tcx Generics<'tcx>,
    /// The value of this will remain `true` if *every* parameter:
    ///   1. Is a type parameter, and
    ///   2. Goes unused in the function.
    /// Otherwise, if any type parameters end up being used, or if any lifetime or const-generic
    /// parameters are present, this will be set to `false`.
    all_params_unused: bool,
}

impl<'cx, 'tcx> TypeWalker<'cx, 'tcx> {
    fn new(cx: &'cx LateContext<'tcx>, generics: &'tcx Generics<'tcx>) -> Self {
        let mut all_params_unused = true;
        let ty_params = generics
            .params
            .iter()
            .filter_map(|param| {
                if let GenericParamKind::Type { .. } = param.kind {
                    Some((param.def_id.into(), param.span))
                } else {
                    if !param.is_elided_lifetime() {
                        all_params_unused = false;
                    }
                    None
                }
            })
            .collect();
        Self {
            cx,
            ty_params,
            bounds: FxHashMap::default(),
            generics,
            all_params_unused,
        }
    }

    fn emit_lint(&self) {
        let (msg, help) = match self.ty_params.len() {
            0 => return,
            1 => (
                "type parameter goes unused in function definition",
                "consider removing the parameter",
            ),
            _ => (
                "type parameters go unused in function definition",
                "consider removing the parameters",
            ),
        };

        let source_map = self.cx.tcx.sess.source_map();
        let span = if self.all_params_unused {
            self.generics.span.into() // Remove the entire list of generics
        } else {
            MultiSpan::from_spans(
                self.ty_params
                    .iter()
                    .map(|(def_id, &span)| {
                        // Extend the span past any trait bounds, and include the comma at the end.
                        let span_to_extend = self.bounds.get(def_id).copied().map_or(span, Span::shrink_to_hi);
                        let comma_range = source_map.span_extend_to_next_char(span_to_extend, '>', false);
                        let comma_span = source_map.span_through_char(comma_range, ',');
                        span.with_hi(comma_span.hi())
                    })
                    .collect(),
            )
        };

        span_lint_and_help(self.cx, EXTRA_UNUSED_TYPE_PARAMETERS, span, msg, None, help);
    }
}

impl<'cx, 'tcx> Visitor<'tcx> for TypeWalker<'cx, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_ty(&mut self, t: &'tcx Ty<'tcx>) {
        if let Some((def_id, _)) = t.peel_refs().as_generic_param() {
            if self.ty_params.remove(&def_id).is_some() {
                self.all_params_unused = false;
            }
        } else if let TyKind::OpaqueDef(id, _, _) = t.kind {
            // Explicitly walk OpaqueDef. Normally `walk_ty` would do the job, but it calls
            // `visit_nested_item`, which checks that `Self::NestedFilter::INTER` is set. We're
            // using `OnlyBodies`, so the check ends up failing and the type isn't fully walked.
            let item = self.nested_visit_map().item(id);
            walk_item(self, item);
        } else {
            walk_ty(self, t);
        }
    }

    fn visit_where_predicate(&mut self, predicate: &'tcx WherePredicate<'tcx>) {
        if let WherePredicate::BoundPredicate(predicate) = predicate {
            // Collect spans for bounds that appear in the list of generics (not in a where-clause)
            // for use in forming the help message
            if let Some((def_id, _)) = predicate.bounded_ty.peel_refs().as_generic_param()
                && let PredicateOrigin::GenericParam = predicate.origin
            {
                self.bounds.insert(def_id, predicate.span);
            }
            // Only walk the right-hand side of where-bounds
            for bound in predicate.bounds {
                walk_param_bound(self, bound);
            }
        }
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

impl<'tcx> LateLintPass<'tcx> for ExtraUnusedTypeParameters {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn(_, generics, _) = item.kind {
            let mut walker = TypeWalker::new(cx, generics);
            walk_item(&mut walker, item);
            walker.emit_lint();
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'tcx>) {
        // Only lint on inherent methods, not trait methods.
        if let ImplItemKind::Fn(..) = item.kind && trait_ref_of_method(cx, item.owner_id.def_id).is_none() {
            let mut walker = TypeWalker::new(cx, item.generics);
            walk_impl_item(&mut walker, item);
            walker.emit_lint();
        }
    }
}
