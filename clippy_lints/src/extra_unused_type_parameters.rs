use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::is_from_proc_macro;
use clippy_utils::trait_ref_of_method;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_impl_item, walk_item, walk_param_bound, walk_ty, Visitor};
use rustc_hir::{
    BodyId, ExprKind, GenericBound, GenericParam, GenericParamKind, Generics, ImplItem, ImplItemKind, Item, ItemKind,
    PredicateOrigin, Ty, TyKind, WherePredicate,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::nested_filter;
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{
    def_id::{DefId, LocalDefId},
    Span,
};

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

pub struct ExtraUnusedTypeParameters {
    avoid_breaking_exported_api: bool,
}

impl ExtraUnusedTypeParameters {
    pub fn new(avoid_breaking_exported_api: bool) -> Self {
        Self {
            avoid_breaking_exported_api,
        }
    }

    /// Don't lint external macros or functions with empty bodies. Also, don't lint exported items
    /// if the `avoid_breaking_exported_api` config option is set.
    fn is_empty_exported_or_macro(
        &self,
        cx: &LateContext<'_>,
        span: Span,
        def_id: LocalDefId,
        body_id: BodyId,
    ) -> bool {
        let body = cx.tcx.hir().body(body_id).value;
        let fn_empty = matches!(&body.kind, ExprKind::Block(blk, None) if blk.stmts.is_empty() && blk.expr.is_none());
        let is_exported = cx.effective_visibilities.is_exported(def_id);
        in_external_macro(cx.sess(), span) || fn_empty || (is_exported && self.avoid_breaking_exported_api)
    }
}

impl_lint_pass!(ExtraUnusedTypeParameters => [EXTRA_UNUSED_TYPE_PARAMETERS]);

/// A visitor struct that walks a given function and gathers generic type parameters, plus any
/// trait bounds those parameters have.
struct TypeWalker<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    /// Collection of the function's type parameters. Once the function has been walked, this will
    /// contain only unused type parameters.
    ty_params: FxHashMap<DefId, Span>,
    /// Collection of any inline trait bounds corresponding to each type parameter.
    inline_bounds: FxHashMap<DefId, Span>,
    /// Collection of any type parameters with trait bounds that appear in a where clause.
    where_bounds: FxHashSet<DefId>,
    /// The entire `Generics` object of the function, useful for querying purposes.
    generics: &'tcx Generics<'tcx>,
}

impl<'cx, 'tcx> TypeWalker<'cx, 'tcx> {
    fn new(cx: &'cx LateContext<'tcx>, generics: &'tcx Generics<'tcx>) -> Self {
        let ty_params = generics
            .params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Type { synthetic, .. } if !synthetic => Some((param.def_id.into(), param.span)),
                _ => None,
            })
            .collect();

        Self {
            cx,
            ty_params,
            inline_bounds: FxHashMap::default(),
            where_bounds: FxHashSet::default(),
            generics,
        }
    }

    fn get_bound_span(&self, param: &'tcx GenericParam<'tcx>) -> Span {
        self.inline_bounds
            .get(&param.def_id.to_def_id())
            .map_or(param.span, |bound_span| param.span.with_hi(bound_span.hi()))
    }

    fn emit_help(&self, spans: Vec<Span>, msg: &str, help: &'static str) {
        span_lint_and_help(self.cx, EXTRA_UNUSED_TYPE_PARAMETERS, spans, msg, None, help);
    }

    fn emit_sugg(&self, spans: Vec<Span>, msg: &str, help: &'static str) {
        let suggestions: Vec<(Span, String)> = spans.iter().copied().zip(std::iter::repeat(String::new())).collect();
        span_lint_and_then(self.cx, EXTRA_UNUSED_TYPE_PARAMETERS, spans, msg, |diag| {
            diag.multipart_suggestion(help, suggestions, Applicability::MachineApplicable);
        });
    }

    fn emit_lint(&self) {
        let explicit_params = self
            .generics
            .params
            .iter()
            .filter(|param| !param.is_elided_lifetime() && !param.is_impl_trait())
            .collect::<Vec<_>>();

        let extra_params = explicit_params
            .iter()
            .enumerate()
            .filter(|(_, param)| self.ty_params.contains_key(&param.def_id.to_def_id()))
            .collect::<Vec<_>>();

        let (msg, help) = match extra_params.len() {
            0 => return,
            1 => (
                format!(
                    "type parameter `{}` goes unused in function definition",
                    extra_params[0].1.name.ident()
                ),
                "consider removing the parameter",
            ),
            _ => (
                format!(
                    "type parameters go unused in function definition: {}",
                    extra_params
                        .iter()
                        .map(|(_, param)| param.name.ident().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                "consider removing the parameters",
            ),
        };

        // If any parameters are bounded in where clauses, don't try to form a suggestion.
        // Otherwise, the leftover where bound would produce code that wouldn't compile.
        if extra_params
            .iter()
            .any(|(_, param)| self.where_bounds.contains(&param.def_id.to_def_id()))
        {
            let spans = extra_params
                .iter()
                .map(|(_, param)| self.get_bound_span(param))
                .collect::<Vec<_>>();
            self.emit_help(spans, &msg, help);
        } else {
            let spans = if explicit_params.len() == extra_params.len() {
                vec![self.generics.span] // Remove the entire list of generics
            } else {
                let mut end: Option<LocalDefId> = None;
                extra_params
                    .iter()
                    .rev()
                    .map(|(idx, param)| {
                        if let Some(next) = explicit_params.get(idx + 1) && end != Some(next.def_id) {
                        // Extend the current span forward, up until the next param in the list.
                        param.span.until(next.span)
                    } else {
                        // Extend the current span back to include the comma following the previous
                        // param. If the span of the next param in the list has already been
                        // extended, we continue the chain. This is why we're iterating in reverse.
                        end = Some(param.def_id);

                        // idx will never be 0, else we'd be removing the entire list of generics
                        let prev = explicit_params[idx - 1];
                        let prev_span = self.get_bound_span(prev);
                        self.get_bound_span(param).with_lo(prev_span.hi())
                    }
                    })
                    .collect()
            };
            self.emit_sugg(spans, &msg, help);
        };
    }
}

/// Given a generic bound, if the bound is for a trait that's not a `LangItem`, return the
/// `LocalDefId` for that trait.
fn bound_to_trait_def_id(bound: &GenericBound<'_>) -> Option<LocalDefId> {
    bound.trait_ref()?.trait_def_id()?.as_local()
}

impl<'cx, 'tcx> Visitor<'tcx> for TypeWalker<'cx, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_ty(&mut self, t: &'tcx Ty<'tcx>) {
        if let Some((def_id, _)) = t.peel_refs().as_generic_param() {
            self.ty_params.remove(&def_id);
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
            // Collect spans for any bounds on type parameters.
            if let Some((def_id, _)) = predicate.bounded_ty.peel_refs().as_generic_param() {
                match predicate.origin {
                    PredicateOrigin::GenericParam => {
                        self.inline_bounds.insert(def_id, predicate.span);
                    },
                    PredicateOrigin::WhereClause => {
                        self.where_bounds.insert(def_id);
                    },
                    PredicateOrigin::ImplTrait => (),
                }

                // If the bound contains non-public traits, err on the safe side and don't lint the
                // corresponding parameter.
                if !predicate
                    .bounds
                    .iter()
                    .filter_map(bound_to_trait_def_id)
                    .all(|id| self.cx.effective_visibilities.is_exported(id))
                {
                    self.ty_params.remove(&def_id);
                }
            }
            // Only walk the right-hand side of where bounds
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
        if let ItemKind::Fn(_, generics, body_id) = item.kind
            && !self.is_empty_exported_or_macro(cx, item.span, item.owner_id.def_id, body_id)
            && !is_from_proc_macro(cx, item)
        {
            let mut walker = TypeWalker::new(cx, generics);
            walk_item(&mut walker, item);
            walker.emit_lint();
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'tcx>) {
        // Only lint on inherent methods, not trait methods.
        if let ImplItemKind::Fn(.., body_id) = item.kind
            && trait_ref_of_method(cx, item.owner_id.def_id).is_none()
            && !self.is_empty_exported_or_macro(cx, item.span, item.owner_id.def_id, body_id)
        {
            let mut walker = TypeWalker::new(cx, item.generics);
            walk_impl_item(&mut walker, item);
            walker.emit_lint();
        }
    }
}
