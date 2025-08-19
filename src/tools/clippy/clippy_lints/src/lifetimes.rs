use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::trait_ref_of_method;
use itertools::Itertools;
use rustc_ast::visit::{try_visit, walk_list};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::Applicability;
use rustc_hir::FnRetTy::Return;
use rustc_hir::intravisit::nested_filter::{self as hir_nested_filter, NestedFilter};
use rustc_hir::intravisit::{
    Visitor, VisitorExt, walk_fn_decl, walk_generic_args, walk_generics, walk_impl_item_ref, walk_param_bound,
    walk_poly_trait_ref, walk_trait_ref, walk_ty, walk_unambig_ty, walk_where_predicate,
};
use rustc_hir::{
    AmbigArg, BodyId, FnDecl, FnPtrTy, FnSig, GenericArg, GenericArgs, GenericBound, GenericParam, GenericParamKind,
    Generics, HirId, Impl, ImplItem, ImplItemKind, Item, ItemKind, Lifetime, LifetimeKind, LifetimeParamKind, Node,
    PolyTraitRef, PredicateOrigin, TraitFn, TraitItem, TraitItemKind, Ty, TyKind, WhereBoundPredicate, WherePredicate,
    WherePredicateKind, lang_items,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::nested_filter as middle_nested_filter;
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::{Ident, kw};
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for lifetime annotations which can be removed by
    /// relying on lifetime elision.
    ///
    /// ### Why is this bad?
    /// The additional lifetimes make the code look more
    /// complicated, while there is nothing out of the ordinary going on. Removing
    /// them leads to more readable code.
    ///
    /// ### Known problems
    /// This lint ignores functions with `where` clauses that reference
    /// lifetimes to prevent false positives.
    ///
    /// ### Example
    /// ```no_run
    /// // Unnecessary lifetime annotations
    /// fn in_and_out<'a>(x: &'a u8, y: u8) -> &'a u8 {
    ///     x
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn elided(x: &u8, y: u8) -> &u8 {
    ///     x
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_LIFETIMES,
    complexity,
    "using explicit lifetimes for references in function arguments when elision rules \
     would allow omitting them"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for lifetime annotations which can be replaced with anonymous lifetimes (`'_`).
    ///
    /// ### Why is this bad?
    /// The additional lifetimes can make the code look more complicated.
    ///
    /// ### Known problems
    /// This lint ignores functions with `where` clauses that reference
    /// lifetimes to prevent false positives.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::str::Chars;
    /// fn f<'a>(x: &'a str) -> Chars<'a> {
    ///     x.chars()
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::str::Chars;
    /// fn f(x: &str) -> Chars<'_> {
    ///     x.chars()
    /// }
    /// ```
    #[clippy::version = "1.87.0"]
    pub ELIDABLE_LIFETIME_NAMES,
    pedantic,
    "lifetime name that can be replaced with the anonymous lifetime"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for lifetimes in generics that are never used
    /// anywhere else.
    ///
    /// ### Why is this bad?
    /// The additional lifetimes make the code look more
    /// complicated, while there is nothing out of the ordinary going on. Removing
    /// them leads to more readable code.
    ///
    /// ### Example
    /// ```no_run
    /// // unnecessary lifetimes
    /// fn unused_lifetime<'a>(x: u8) {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn no_lifetime(x: u8) {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EXTRA_UNUSED_LIFETIMES,
    complexity,
    "unused lifetimes in function definitions"
}

pub struct Lifetimes {
    msrv: Msrv,
}

impl Lifetimes {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(Lifetimes => [
    NEEDLESS_LIFETIMES,
    ELIDABLE_LIFETIME_NAMES,
    EXTRA_UNUSED_LIFETIMES,
]);

impl<'tcx> LateLintPass<'tcx> for Lifetimes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Fn {
            ref sig,
            generics,
            body: id,
            ..
        } = item.kind
        {
            check_fn_inner(cx, sig, Some(id), None, generics, item.span, true, self.msrv);
        } else if let ItemKind::Impl(impl_) = &item.kind
            && !item.span.from_expansion()
        {
            report_extra_impl_lifetimes(cx, impl_);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(ref sig, id) = item.kind {
            let report_extra_lifetimes = trait_ref_of_method(cx, item.owner_id).is_none();
            check_fn_inner(
                cx,
                sig,
                Some(id),
                None,
                item.generics,
                item.span,
                report_extra_lifetimes,
                self.msrv,
            );
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(ref sig, ref body) = item.kind {
            let (body, trait_sig) = match *body {
                TraitFn::Required(sig) => (None, Some(sig)),
                TraitFn::Provided(id) => (Some(id), None),
            };
            check_fn_inner(cx, sig, body, trait_sig, item.generics, item.span, true, self.msrv);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_fn_inner<'tcx>(
    cx: &LateContext<'tcx>,
    sig: &'tcx FnSig<'_>,
    body: Option<BodyId>,
    trait_sig: Option<&[Option<Ident>]>,
    generics: &'tcx Generics<'_>,
    span: Span,
    report_extra_lifetimes: bool,
    msrv: Msrv,
) {
    if span.in_external_macro(cx.sess().source_map()) || has_where_lifetimes(cx, generics) {
        return;
    }

    let types = generics
        .params
        .iter()
        .filter(|param| matches!(param.kind, GenericParamKind::Type { .. }));

    for typ in types {
        if !typ.span.eq_ctxt(span) {
            return;
        }

        for pred in generics.bounds_for_param(typ.def_id) {
            if pred.origin == PredicateOrigin::WhereClause {
                // has_where_lifetimes checked that this predicate contains no lifetime.
                continue;
            }

            for bound in pred.bounds {
                let mut visitor = RefVisitor::new(cx);
                walk_param_bound(&mut visitor, bound);
                if visitor.lts.iter().any(|lt| matches!(lt.kind, LifetimeKind::Param(_))) {
                    return;
                }
                if let GenericBound::Trait(ref trait_ref) = *bound {
                    let params = &trait_ref
                        .trait_ref
                        .path
                        .segments
                        .last()
                        .expect("a path must have at least one segment")
                        .args;
                    if let Some(params) = *params {
                        let lifetimes = params.args.iter().filter_map(|arg| match arg {
                            GenericArg::Lifetime(lt) => Some(lt),
                            _ => None,
                        });
                        for bound in lifetimes {
                            if bound.kind != LifetimeKind::Static && !bound.is_elided() {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((elidable_lts, usages)) = could_use_elision(cx, sig.decl, body, trait_sig, generics.params, msrv) {
        if usages.iter().any(|usage| !usage.ident.span.eq_ctxt(span)) {
            return;
        }
        // async functions have usages whose spans point at the lifetime declaration which messes up
        // suggestions
        let include_suggestions = !sig.header.is_async();
        report_elidable_lifetimes(cx, generics, &elidable_lts, &usages, include_suggestions);
    }

    if report_extra_lifetimes {
        self::report_extra_lifetimes(cx, sig.decl, generics);
    }
}

fn could_use_elision<'tcx>(
    cx: &LateContext<'tcx>,
    func: &'tcx FnDecl<'_>,
    body: Option<BodyId>,
    trait_sig: Option<&[Option<Ident>]>,
    named_generics: &'tcx [GenericParam<'_>],
    msrv: Msrv,
) -> Option<(Vec<LocalDefId>, Vec<Lifetime>)> {
    // There are two scenarios where elision works:
    // * no output references, all input references have different LT
    // * output references, exactly one input reference with same LT
    // All lifetimes must be unnamed, 'static or defined without bounds on the
    // level of the current item.

    // check named LTs
    let allowed_lts = allowed_lts_from(named_generics);

    // these will collect all the lifetimes for references in arg/return types
    let mut input_visitor = RefVisitor::new(cx);
    let mut output_visitor = RefVisitor::new(cx);

    // extract lifetimes in input argument types
    for arg in func.inputs {
        input_visitor.visit_ty_unambig(arg);
    }
    // extract lifetimes in output type
    if let Return(ty) = func.output {
        output_visitor.visit_ty_unambig(ty);
    }
    for lt in named_generics {
        input_visitor.visit_generic_param(lt);
    }

    if input_visitor.abort() || output_visitor.abort() {
        return None;
    }

    let input_lts = input_visitor.lts;
    let output_lts = output_visitor.lts;

    if let Some(&[trait_sig]) = trait_sig
        && non_elidable_self_type(cx, func, trait_sig, msrv)
    {
        return None;
    }

    if let Some(body_id) = body {
        let body = cx.tcx.hir_body(body_id);

        let first_ident = body.params.first().and_then(|param| param.pat.simple_ident());
        if non_elidable_self_type(cx, func, first_ident, msrv) {
            return None;
        }

        let mut checker = BodyLifetimeChecker::new(cx);
        if checker.visit_expr(body.value).is_break() {
            return None;
        }
    }

    // check for lifetimes from higher scopes
    for lt in input_lts.iter().chain(output_lts.iter()) {
        if let Some(id) = named_lifetime(lt)
            && !allowed_lts.contains(&id)
        {
            return None;
        }
    }

    // check for higher-ranked trait bounds
    if !input_visitor.nested_elision_site_lts.is_empty() || !output_visitor.nested_elision_site_lts.is_empty() {
        let allowed_lts: FxHashSet<_> = allowed_lts.iter().map(|id| cx.tcx.item_name(id.to_def_id())).collect();
        for lt in input_visitor.nested_elision_site_lts {
            if allowed_lts.contains(&lt.ident.name) {
                return None;
            }
        }
        for lt in output_visitor.nested_elision_site_lts {
            if allowed_lts.contains(&lt.ident.name) {
                return None;
            }
        }
    }

    // A lifetime can be newly elided if:
    // - It occurs only once among the inputs.
    // - If there are multiple input lifetimes, then the newly elided lifetime does not occur among the
    //   outputs (because eliding such an lifetime would create an ambiguity).
    let elidable_lts = named_lifetime_occurrences(&input_lts)
        .into_iter()
        .filter_map(|(def_id, occurrences)| {
            if occurrences == 1
                && (input_lts.len() == 1 || !output_lts.iter().any(|lt| named_lifetime(lt) == Some(def_id)))
            {
                Some(def_id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if elidable_lts.is_empty() {
        return None;
    }

    let usages = itertools::chain(input_lts, output_lts).collect();

    Some((elidable_lts, usages))
}

fn allowed_lts_from(named_generics: &[GenericParam<'_>]) -> FxIndexSet<LocalDefId> {
    named_generics
        .iter()
        .filter_map(|par| {
            if let GenericParamKind::Lifetime { .. } = par.kind {
                Some(par.def_id)
            } else {
                None
            }
        })
        .collect()
}

// elision doesn't work for explicit self types before Rust 1.81, see rust-lang/rust#69064
fn non_elidable_self_type<'tcx>(cx: &LateContext<'tcx>, func: &FnDecl<'tcx>, ident: Option<Ident>, msrv: Msrv) -> bool {
    if let Some(ident) = ident
        && ident.name == kw::SelfLower
        && !func.implicit_self.has_implicit_self()
        && let Some(self_ty) = func.inputs.first()
        && !msrv.meets(cx, msrvs::EXPLICIT_SELF_TYPE_ELISION)
    {
        let mut visitor = RefVisitor::new(cx);
        visitor.visit_ty_unambig(self_ty);

        !visitor.all_lts().is_empty()
    } else {
        false
    }
}

/// Number of times each named lifetime occurs in the given slice. Returns a vector to preserve
/// relative order.
#[must_use]
fn named_lifetime_occurrences(lts: &[Lifetime]) -> Vec<(LocalDefId, usize)> {
    let mut occurrences = Vec::new();
    for lt in lts {
        if let Some(curr_def_id) = named_lifetime(lt) {
            if let Some(pair) = occurrences
                .iter_mut()
                .find(|(prev_def_id, _)| *prev_def_id == curr_def_id)
            {
                pair.1 += 1;
            } else {
                occurrences.push((curr_def_id, 1));
            }
        }
    }
    occurrences
}

fn named_lifetime(lt: &Lifetime) -> Option<LocalDefId> {
    match lt.kind {
        LifetimeKind::Param(id) if !lt.is_anonymous() => Some(id),
        _ => None,
    }
}

struct RefVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    lts: Vec<Lifetime>,
    nested_elision_site_lts: Vec<Lifetime>,
    unelided_trait_object_lifetime: bool,
}

impl<'a, 'tcx> RefVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            lts: Vec::new(),
            nested_elision_site_lts: Vec::new(),
            unelided_trait_object_lifetime: false,
        }
    }

    fn all_lts(&self) -> Vec<Lifetime> {
        self.lts
            .iter()
            .chain(self.nested_elision_site_lts.iter())
            .copied()
            .collect::<Vec<_>>()
    }

    fn abort(&self) -> bool {
        self.unelided_trait_object_lifetime
    }
}

impl<'tcx> Visitor<'tcx> for RefVisitor<'_, 'tcx> {
    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        self.lts.push(*lifetime);
    }

    fn visit_poly_trait_ref(&mut self, poly_tref: &'tcx PolyTraitRef<'tcx>) {
        let trait_ref = &poly_tref.trait_ref;
        if let Some(id) = trait_ref.trait_def_id()
            && lang_items::FN_TRAITS
                .iter()
                .any(|&item| self.cx.tcx.lang_items().get(item) == Some(id))
        {
            let mut sub_visitor = RefVisitor::new(self.cx);
            sub_visitor.visit_trait_ref(trait_ref);
            self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
        } else {
            walk_poly_trait_ref(self, poly_tref);
        }
    }

    fn visit_ty(&mut self, ty: &'tcx Ty<'_, AmbigArg>) {
        match ty.kind {
            TyKind::FnPtr(&FnPtrTy { decl, .. }) => {
                let mut sub_visitor = RefVisitor::new(self.cx);
                sub_visitor.visit_fn_decl(decl);
                self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
            },
            TyKind::TraitObject(bounds, lt) => {
                if !lt.is_elided() {
                    self.unelided_trait_object_lifetime = true;
                }
                for bound in bounds {
                    self.visit_poly_trait_ref(bound);
                }
            },
            _ => walk_ty(self, ty),
        }
    }
}

/// Are any lifetimes mentioned in the `where` clause? If so, we don't try to
/// reason about elision.
fn has_where_lifetimes<'tcx>(cx: &LateContext<'tcx>, generics: &'tcx Generics<'_>) -> bool {
    for predicate in generics.predicates {
        match *predicate.kind {
            WherePredicateKind::RegionPredicate(..) => return true,
            WherePredicateKind::BoundPredicate(ref pred) => {
                // a predicate like F: Trait or F: for<'a> Trait<'a>
                let mut visitor = RefVisitor::new(cx);
                // walk the type F, it may not contain LT refs
                walk_unambig_ty(&mut visitor, pred.bounded_ty);
                if !visitor.all_lts().is_empty() {
                    return true;
                }
                // if the bounds define new lifetimes, they are fine to occur
                let allowed_lts = allowed_lts_from(pred.bound_generic_params);
                // now walk the bounds
                for bound in pred.bounds {
                    walk_param_bound(&mut visitor, bound);
                }
                // and check that all lifetimes are allowed
                for lt in visitor.all_lts() {
                    if let Some(id) = named_lifetime(&lt)
                        && !allowed_lts.contains(&id)
                    {
                        return true;
                    }
                }
            },
            WherePredicateKind::EqPredicate(ref pred) => {
                let mut visitor = RefVisitor::new(cx);
                walk_unambig_ty(&mut visitor, pred.lhs_ty);
                walk_unambig_ty(&mut visitor, pred.rhs_ty);
                if !visitor.lts.is_empty() {
                    return true;
                }
            },
        }
    }
    false
}

#[allow(clippy::struct_excessive_bools)]
struct Usage {
    lifetime: Lifetime,
    in_where_predicate: bool,
    in_bounded_ty: bool,
    in_generics_arg: bool,
    lifetime_elision_impossible: bool,
}

struct LifetimeChecker<'cx, 'tcx, F> {
    cx: &'cx LateContext<'tcx>,
    map: FxIndexMap<LocalDefId, Vec<Usage>>,
    where_predicate_depth: usize,
    bounded_ty_depth: usize,
    generic_args_depth: usize,
    lifetime_elision_impossible: bool,
    phantom: std::marker::PhantomData<F>,
}

impl<'cx, 'tcx, F> LifetimeChecker<'cx, 'tcx, F>
where
    F: NestedFilter<'tcx>,
{
    fn new(cx: &'cx LateContext<'tcx>, generics: &'tcx Generics<'_>) -> LifetimeChecker<'cx, 'tcx, F> {
        let map = generics
            .params
            .iter()
            .filter_map(|par| match par.kind {
                GenericParamKind::Lifetime {
                    kind: LifetimeParamKind::Explicit,
                } => Some((par.def_id, Vec::new())),
                _ => None,
            })
            .collect();
        Self {
            cx,
            map,
            where_predicate_depth: 0,
            bounded_ty_depth: 0,
            generic_args_depth: 0,
            lifetime_elision_impossible: false,
            phantom: std::marker::PhantomData,
        }
    }

    // `visit_where_bound_predicate` is based on:
    // https://github.com/rust-lang/rust/blob/864cee3ea383cc8254ba394ba355e648faa9cfa5/compiler/rustc_hir/src/intravisit.rs#L936-L939
    fn visit_where_bound_predicate(
        &mut self,
        hir_id: HirId,
        bounded_ty: &'tcx Ty<'tcx>,
        bounds: &'tcx [GenericBound<'tcx>],
        bound_generic_params: &'tcx [GenericParam<'tcx>],
    ) {
        try_visit!(self.visit_id(hir_id));

        self.bounded_ty_depth += 1;
        try_visit!(self.visit_ty_unambig(bounded_ty));
        self.bounded_ty_depth -= 1;

        walk_list!(self, visit_param_bound, bounds);
        walk_list!(self, visit_generic_param, bound_generic_params);
    }
}

impl<'tcx, F> Visitor<'tcx> for LifetimeChecker<'_, 'tcx, F>
where
    F: NestedFilter<'tcx>,
{
    type MaybeTyCtxt = TyCtxt<'tcx>;
    type NestedFilter = F;

    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        if let LifetimeKind::Param(def_id) = lifetime.kind
            && let Some(usages) = self.map.get_mut(&def_id)
        {
            usages.push(Usage {
                lifetime: *lifetime,
                in_where_predicate: self.where_predicate_depth != 0,
                in_bounded_ty: self.bounded_ty_depth != 0,
                in_generics_arg: self.generic_args_depth != 0,
                lifetime_elision_impossible: self.lifetime_elision_impossible,
            });
        }
    }

    fn visit_where_predicate(&mut self, predicate: &'tcx WherePredicate<'tcx>) {
        self.where_predicate_depth += 1;
        if let &WherePredicateKind::BoundPredicate(WhereBoundPredicate {
            bounded_ty,
            bounds,
            bound_generic_params,
            origin: _,
        }) = predicate.kind
        {
            self.visit_where_bound_predicate(predicate.hir_id, bounded_ty, bounds, bound_generic_params);
        } else {
            walk_where_predicate(self, predicate);
        }
        self.where_predicate_depth -= 1;
    }

    fn visit_generic_args(&mut self, generic_args: &'tcx GenericArgs<'tcx>) -> Self::Result {
        self.generic_args_depth += 1;
        walk_generic_args(self, generic_args);
        self.generic_args_depth -= 1;
    }

    fn visit_fn_decl(&mut self, fd: &'tcx FnDecl<'tcx>) -> Self::Result {
        self.lifetime_elision_impossible = !is_candidate_for_elision(fd);
        walk_fn_decl(self, fd);
        self.lifetime_elision_impossible = false;
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

/// Check if `fd` supports function elision with an anonymous (or elided) lifetime,
/// and has a lifetime somewhere in its output type.
fn is_candidate_for_elision(fd: &FnDecl<'_>) -> bool {
    struct V;

    impl Visitor<'_> for V {
        type Result = ControlFlow<bool>;

        fn visit_lifetime(&mut self, lifetime: &Lifetime) -> Self::Result {
            ControlFlow::Break(lifetime.is_elided() || lifetime.is_anonymous())
        }
    }

    if fd.lifetime_elision_allowed
        && let Return(ret_ty) = fd.output
        && walk_unambig_ty(&mut V, ret_ty).is_break()
    {
        // The first encountered input lifetime will either be one on `self`, or will be the only lifetime.
        fd.inputs
            .iter()
            .find_map(|ty| walk_unambig_ty(&mut V, ty).break_value())
            .unwrap()
    } else {
        false
    }
}

fn report_extra_lifetimes<'tcx>(cx: &LateContext<'tcx>, func: &'tcx FnDecl<'_>, generics: &'tcx Generics<'_>) {
    let mut checker = LifetimeChecker::<hir_nested_filter::None>::new(cx, generics);

    walk_generics(&mut checker, generics);
    walk_fn_decl(&mut checker, func);

    for (def_id, usages) in checker.map {
        if usages
            .iter()
            .all(|usage| usage.in_where_predicate && !usage.in_bounded_ty && !usage.in_generics_arg)
        {
            span_lint(
                cx,
                EXTRA_UNUSED_LIFETIMES,
                cx.tcx.def_span(def_id),
                "this lifetime isn't used in the function definition",
            );
        }
    }
}

fn report_extra_impl_lifetimes<'tcx>(cx: &LateContext<'tcx>, impl_: &'tcx Impl<'_>) {
    let mut checker = LifetimeChecker::<middle_nested_filter::All>::new(cx, impl_.generics);

    walk_generics(&mut checker, impl_.generics);
    if let Some(of_trait) = impl_.of_trait {
        walk_trait_ref(&mut checker, &of_trait.trait_ref);
    }
    walk_unambig_ty(&mut checker, impl_.self_ty);
    for &item in impl_.items {
        walk_impl_item_ref(&mut checker, item);
    }

    for (&def_id, usages) in &checker.map {
        if usages
            .iter()
            .all(|usage| usage.in_where_predicate && !usage.in_bounded_ty && !usage.in_generics_arg)
        {
            span_lint(
                cx,
                EXTRA_UNUSED_LIFETIMES,
                cx.tcx.def_span(def_id),
                "this lifetime isn't used in the impl",
            );
        }
    }

    report_elidable_impl_lifetimes(cx, impl_, &checker.map);
}

// An `impl` lifetime is elidable if it satisfies the following conditions:
// - It is used exactly once.
// - That single use is not in a `WherePredicate`.
fn report_elidable_impl_lifetimes<'tcx>(
    cx: &LateContext<'tcx>,
    impl_: &'tcx Impl<'_>,
    map: &FxIndexMap<LocalDefId, Vec<Usage>>,
) {
    let single_usages = map
        .iter()
        .filter_map(|(def_id, usages)| {
            if let [
                Usage {
                    lifetime,
                    in_where_predicate: false,
                    lifetime_elision_impossible: false,
                    ..
                },
            ] = usages.as_slice()
            {
                Some((def_id, lifetime))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if single_usages.is_empty() {
        return;
    }

    let (elidable_lts, usages): (Vec<_>, Vec<_>) = single_usages.into_iter().unzip();

    report_elidable_lifetimes(cx, impl_.generics, &elidable_lts, &usages, true);
}

#[derive(Copy, Clone)]
enum ElidableUsage {
    /// Used in a ref (`&'a T`), can be removed
    Ref(Span),
    /// Used as a generic param (`T<'a>`) or an impl lifetime (`impl T + 'a`), can be replaced
    /// with `'_`
    Other(Span),
}

/// Generate diagnostic messages for elidable lifetimes.
fn report_elidable_lifetimes(
    cx: &LateContext<'_>,
    generics: &Generics<'_>,
    elidable_lts: &[LocalDefId],
    usages: &[Lifetime],
    include_suggestions: bool,
) {
    let lts = elidable_lts
        .iter()
        // In principle, the result of the call to `Node::ident` could be `unwrap`ped, as `DefId` should refer to a
        // `Node::GenericParam`.
        .filter_map(|&def_id| cx.tcx.hir_node_by_def_id(def_id).ident())
        .map(|ident| ident.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let elidable_usages: Vec<ElidableUsage> = usages
        .iter()
        .filter(|usage| named_lifetime(usage).is_some_and(|id| elidable_lts.contains(&id)))
        .map(|usage| match cx.tcx.parent_hir_node(usage.hir_id) {
            Node::Ty(Ty {
                kind: TyKind::Ref(..), ..
            }) => ElidableUsage::Ref(usage.ident.span),
            _ => ElidableUsage::Other(usage.ident.span),
        })
        .collect();

    let lint = if elidable_usages
        .iter()
        .any(|usage| matches!(usage, ElidableUsage::Other(_)))
    {
        ELIDABLE_LIFETIME_NAMES
    } else {
        NEEDLESS_LIFETIMES
    };

    span_lint_and_then(
        cx,
        lint,
        elidable_lts
            .iter()
            .map(|&lt| cx.tcx.def_span(lt))
            .chain(usages.iter().filter_map(|usage| {
                if let LifetimeKind::Param(def_id) = usage.kind
                    && elidable_lts.contains(&def_id)
                {
                    return Some(usage.ident.span);
                }

                None
            }))
            .collect_vec(),
        format!("the following explicit lifetimes could be elided: {lts}"),
        |diag| {
            if !include_suggestions {
                return;
            }

            if let Some(suggestions) = elision_suggestions(cx, generics, elidable_lts, &elidable_usages) {
                diag.multipart_suggestion("elide the lifetimes", suggestions, Applicability::MachineApplicable);
            }
        },
    );
}

fn elision_suggestions(
    cx: &LateContext<'_>,
    generics: &Generics<'_>,
    elidable_lts: &[LocalDefId],
    usages: &[ElidableUsage],
) -> Option<Vec<(Span, String)>> {
    let explicit_params = generics
        .params
        .iter()
        .filter(|param| !param.is_elided_lifetime() && !param.is_impl_trait())
        .collect::<Vec<_>>();

    let mut suggestions = if elidable_lts.len() == explicit_params.len() {
        // if all the params are elided remove the whole generic block
        //
        // fn x<'a>() {}
        //     ^^^^
        vec![(generics.span, String::new())]
    } else {
        elidable_lts
            .iter()
            .map(|&id| {
                let pos = explicit_params.iter().position(|param| param.def_id == id)?;
                let param = explicit_params.get(pos)?;

                let span = if let Some(next) = explicit_params.get(pos + 1) {
                    // fn x<'prev, 'a, 'next>() {}
                    //             ^^^^
                    param.span.until(next.span)
                } else {
                    // `pos` should be at least 1 here, because the param in position 0 would either have a `next`
                    // param or would have taken the `elidable_lts.len() == explicit_params.len()` branch.
                    let prev = explicit_params.get(pos - 1)?;

                    // fn x<'prev, 'a>() {}
                    //           ^^^^
                    param.span.with_lo(prev.span.hi())
                };

                Some((span, String::new()))
            })
            .collect::<Option<Vec<_>>>()?
    };

    suggestions.extend(usages.iter().map(|&usage| {
        match usage {
            ElidableUsage::Ref(span) => {
                // expand `&'a T` to `&'a T`
                //          ^^         ^^^
                let span = cx.sess().source_map().span_extend_while_whitespace(span);

                (span, String::new())
            },
            ElidableUsage::Other(span) => {
                // `T<'a>` and `impl Foo + 'a` should be replaced by `'_`
                (span, String::from("'_"))
            },
        }
    }));

    Some(suggestions)
}

struct BodyLifetimeChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> BodyLifetimeChecker<'tcx> {
    fn new(cx: &LateContext<'tcx>) -> Self {
        Self { tcx: cx.tcx }
    }
}

impl<'tcx> Visitor<'tcx> for BodyLifetimeChecker<'tcx> {
    type Result = ControlFlow<()>;
    type NestedFilter = middle_nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }
    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) -> ControlFlow<()> {
        if !lifetime.is_anonymous() && lifetime.ident.name != kw::StaticLifetime {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}
