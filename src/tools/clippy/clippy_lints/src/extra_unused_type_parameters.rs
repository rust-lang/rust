use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::{is_from_proc_macro, trait_ref_of_method};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{Visitor, walk_impl_item, walk_item, walk_param_bound, walk_ty, walk_unambig_ty};
use rustc_hir::{
    AmbigArg, BodyId, ExprKind, GenericBound, GenericParam, GenericParamKind, Generics, ImplItem, ImplItemKind, Item,
    ItemKind, PredicateOrigin, Ty, WherePredicate, WherePredicateKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::nested_filter;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::{DefId, LocalDefId};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for type parameters in generics that are never used anywhere else.
    ///
    /// ### Why is this bad?
    /// Functions cannot infer the value of unused type parameters; therefore, calling them
    /// requires using a turbofish, which serves no purpose but to satisfy the compiler.
    ///
    /// ### Example
    /// ```no_run
    /// fn unused_ty<T>(x: u8) {
    ///     // ..
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
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
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
        }
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

    fn emit_help(&self, spans: Vec<Span>, msg: String, help: &'static str) {
        span_lint_and_help(self.cx, EXTRA_UNUSED_TYPE_PARAMETERS, spans, msg, None, help);
    }

    fn emit_sugg(&self, spans: Vec<Span>, msg: String, help: &'static str) {
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
            self.emit_help(spans, msg, help);
        } else {
            let spans = if explicit_params.len() == extra_params.len() {
                vec![self.generics.span] // Remove the entire list of generics
            } else {
                let mut end: Option<LocalDefId> = None;
                extra_params
                    .iter()
                    .rev()
                    .map(|(idx, param)| {
                        if let Some(next) = explicit_params.get(idx + 1)
                            && end != Some(next.def_id)
                        {
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
            self.emit_sugg(spans, msg, help);
        }
    }
}

/// Given a generic bound, if the bound is for a trait that's not a `LangItem`, return the
/// `LocalDefId` for that trait.
fn bound_to_trait_def_id(bound: &GenericBound<'_>) -> Option<LocalDefId> {
    bound.trait_ref()?.trait_def_id()?.as_local()
}

impl<'tcx> Visitor<'tcx> for TypeWalker<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_ty(&mut self, t: &'tcx Ty<'tcx, AmbigArg>) {
        if let Some((def_id, _)) = t.peel_refs().as_generic_param() {
            self.ty_params.remove(&def_id);
        } else {
            walk_ty(self, t);
        }
    }

    fn visit_where_predicate(&mut self, predicate: &'tcx WherePredicate<'tcx>) {
        let span = predicate.span;
        if let WherePredicateKind::BoundPredicate(predicate) = predicate.kind {
            // Collect spans for any bounds on type parameters.
            if let Some((def_id, _)) = predicate.bounded_ty.peel_refs().as_generic_param() {
                match predicate.origin {
                    PredicateOrigin::GenericParam => {
                        self.inline_bounds.insert(def_id, span);
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
            } else {
                // If the bounded type isn't a generic param, but is instead a concrete generic
                // type, any params we find nested inside of it are being used as concrete types,
                // and can therefore can be considered used. So, we're fine to walk the left-hand
                // side of the where bound.
                walk_unambig_ty(self, predicate.bounded_ty);
            }
            for bound in predicate.bounds {
                walk_param_bound(self, bound);
            }
        }
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

fn is_empty_body(cx: &LateContext<'_>, body: BodyId) -> bool {
    matches!(cx.tcx.hir_body(body).value.kind, ExprKind::Block(b, _) if b.stmts.is_empty() && b.expr.is_none())
}

impl<'tcx> LateLintPass<'tcx> for ExtraUnusedTypeParameters {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Fn {
            generics,
            body: body_id,
            ..
        } = item.kind
            && !generics.params.is_empty()
            && !is_empty_body(cx, body_id)
            && (!self.avoid_breaking_exported_api || !cx.effective_visibilities.is_exported(item.owner_id.def_id))
            && !item.span.in_external_macro(cx.sess().source_map())
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
            && !item.generics.params.is_empty()
            && trait_ref_of_method(cx, item.owner_id).is_none()
            && !is_empty_body(cx, body_id)
            && (!self.avoid_breaking_exported_api || !cx.effective_visibilities.is_exported(item.owner_id.def_id))
            && !item.span.in_external_macro(cx.sess().source_map())
            && !is_from_proc_macro(cx, item)
        {
            let mut walker = TypeWalker::new(cx, item.generics);
            walk_impl_item(&mut walker, item);
            walker.emit_lint();
        }
    }
}
