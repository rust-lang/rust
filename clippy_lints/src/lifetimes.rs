use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::trait_ref_of_method;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir::intravisit::nested_filter::{self as hir_nested_filter, NestedFilter};
use rustc_hir::intravisit::{
    walk_fn_decl, walk_generic_param, walk_generics, walk_impl_item_ref, walk_item, walk_param_bound,
    walk_poly_trait_ref, walk_trait_ref, walk_ty, Visitor,
};
use rustc_hir::FnRetTy::Return;
use rustc_hir::{
    lang_items, BareFnTy, BodyId, FnDecl, FnSig, GenericArg, GenericBound, GenericParam, GenericParamKind, Generics,
    Impl, ImplItem, ImplItemKind, Item, ItemKind, Lifetime, LifetimeName, LifetimeParamKind, Node, PolyTraitRef,
    PredicateOrigin, TraitFn, TraitItem, TraitItemKind, Ty, TyKind, WherePredicate,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::nested_filter as middle_nested_filter;
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::source_map::Span;
use rustc_span::symbol::{kw, Ident, Symbol};

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
    /// - We bail out if the function has a `where` clause where lifetimes
    /// are mentioned due to potential false positives.
    ///
    /// ### Example
    /// ```rust
    /// // Unnecessary lifetime annotations
    /// fn in_and_out<'a>(x: &'a u8, y: u8) -> &'a u8 {
    ///     x
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
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
    /// Checks for lifetimes in generics that are never used
    /// anywhere else.
    ///
    /// ### Why is this bad?
    /// The additional lifetimes make the code look more
    /// complicated, while there is nothing out of the ordinary going on. Removing
    /// them leads to more readable code.
    ///
    /// ### Example
    /// ```rust
    /// // unnecessary lifetimes
    /// fn unused_lifetime<'a>(x: u8) {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// fn no_lifetime(x: u8) {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub EXTRA_UNUSED_LIFETIMES,
    complexity,
    "unused lifetimes in function definitions"
}

declare_lint_pass!(Lifetimes => [NEEDLESS_LIFETIMES, EXTRA_UNUSED_LIFETIMES]);

impl<'tcx> LateLintPass<'tcx> for Lifetimes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Fn(ref sig, generics, id) = item.kind {
            check_fn_inner(cx, sig, Some(id), None, generics, item.span, true);
        } else if let ItemKind::Impl(impl_) = item.kind {
            if !item.span.from_expansion() {
                report_extra_impl_lifetimes(cx, impl_);
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(ref sig, id) = item.kind {
            let report_extra_lifetimes = trait_ref_of_method(cx, item.owner_id.def_id).is_none();
            check_fn_inner(
                cx,
                sig,
                Some(id),
                None,
                item.generics,
                item.span,
                report_extra_lifetimes,
            );
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(ref sig, ref body) = item.kind {
            let (body, trait_sig) = match *body {
                TraitFn::Required(sig) => (None, Some(sig)),
                TraitFn::Provided(id) => (Some(id), None),
            };
            check_fn_inner(cx, sig, body, trait_sig, item.generics, item.span, true);
        }
    }
}

fn check_fn_inner<'tcx>(
    cx: &LateContext<'tcx>,
    sig: &'tcx FnSig<'_>,
    body: Option<BodyId>,
    trait_sig: Option<&[Ident]>,
    generics: &'tcx Generics<'_>,
    span: Span,
    report_extra_lifetimes: bool,
) {
    if in_external_macro(cx.sess(), span) || has_where_lifetimes(cx, generics) {
        return;
    }

    let types = generics
        .params
        .iter()
        .filter(|param| matches!(param.kind, GenericParamKind::Type { .. }));

    for typ in types {
        for pred in generics.bounds_for_param(cx.tcx.hir().local_def_id(typ.hir_id)) {
            if pred.origin == PredicateOrigin::WhereClause {
                // has_where_lifetimes checked that this predicate contains no lifetime.
                continue;
            }

            for bound in pred.bounds {
                let mut visitor = RefVisitor::new(cx);
                walk_param_bound(&mut visitor, bound);
                if visitor.lts.iter().any(|lt| matches!(lt.res, LifetimeName::Param(_))) {
                    return;
                }
                if let GenericBound::Trait(ref trait_ref, _) = *bound {
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
                            if !bound.is_static() && !bound.is_elided() {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((elidable_lts, usages)) = could_use_elision(cx, sig.decl, body, trait_sig, generics.params) {
        let lts = elidable_lts
            .iter()
            // In principle, the result of the call to `Node::ident` could be `unwrap`ped, as `DefId` should refer to a
            // `Node::GenericParam`.
            .filter_map(|&def_id| cx.tcx.hir().get_by_def_id(def_id).ident())
            .map(|ident| ident.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        span_lint_and_then(
            cx,
            NEEDLESS_LIFETIMES,
            span.with_hi(sig.decl.output.span().hi()),
            &format!("the following explicit lifetimes could be elided: {lts}"),
            |diag| {
                if sig.header.is_async() {
                    // async functions have usages whose spans point at the lifetime declaration which messes up
                    // suggestions
                    return;
                };

                if let Some(suggestions) = elision_suggestions(cx, generics, &elidable_lts, &usages) {
                    diag.multipart_suggestion("elide the lifetimes", suggestions, Applicability::MachineApplicable);
                }
            },
        );
    }

    if report_extra_lifetimes {
        self::report_extra_lifetimes(cx, sig.decl, generics);
    }
}

fn elision_suggestions(
    cx: &LateContext<'_>,
    generics: &Generics<'_>,
    elidable_lts: &[LocalDefId],
    usages: &[Lifetime],
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

    suggestions.extend(
        usages
            .iter()
            .filter(|usage| named_lifetime(usage).map_or(false, |id| elidable_lts.contains(&id)))
            .map(|usage| {
                match cx.tcx.hir().get_parent(usage.hir_id) {
                    Node::Ty(Ty {
                        kind: TyKind::Ref(..), ..
                    }) => {
                        // expand `&'a T` to `&'a T`
                        //          ^^         ^^^
                        let span = cx
                            .sess()
                            .source_map()
                            .span_extend_while(usage.ident.span, |ch| ch.is_ascii_whitespace())
                            .unwrap_or(usage.ident.span);

                        (span, String::new())
                    },
                    // `T<'a>` and `impl Foo + 'a` should be replaced by `'_`
                    _ => (usage.ident.span, String::from("'_")),
                }
            }),
    );

    Some(suggestions)
}

// elision doesn't work for explicit self types, see rust-lang/rust#69064
fn explicit_self_type<'tcx>(cx: &LateContext<'tcx>, func: &FnDecl<'tcx>, ident: Option<Ident>) -> bool {
    if_chain! {
        if let Some(ident) = ident;
        if ident.name == kw::SelfLower;
        if !func.implicit_self.has_implicit_self();

        if let Some(self_ty) = func.inputs.first();
        then {
            let mut visitor = RefVisitor::new(cx);
            visitor.visit_ty(self_ty);

            !visitor.all_lts().is_empty()
        } else {
            false
        }
    }
}

fn named_lifetime(lt: &Lifetime) -> Option<LocalDefId> {
    match lt.res {
        LifetimeName::Param(id) if !lt.is_anonymous() => Some(id),
        _ => None,
    }
}

fn could_use_elision<'tcx>(
    cx: &LateContext<'tcx>,
    func: &'tcx FnDecl<'_>,
    body: Option<BodyId>,
    trait_sig: Option<&[Ident]>,
    named_generics: &'tcx [GenericParam<'_>],
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
        input_visitor.visit_ty(arg);
    }
    // extract lifetimes in output type
    if let Return(ty) = func.output {
        output_visitor.visit_ty(ty);
    }
    for lt in named_generics {
        input_visitor.visit_generic_param(lt);
    }

    if input_visitor.abort() || output_visitor.abort() {
        return None;
    }

    let input_lts = input_visitor.lts;
    let output_lts = output_visitor.lts;

    if let Some(trait_sig) = trait_sig {
        if explicit_self_type(cx, func, trait_sig.first().copied()) {
            return None;
        }
    }

    if let Some(body_id) = body {
        let body = cx.tcx.hir().body(body_id);

        let first_ident = body.params.first().and_then(|param| param.pat.simple_ident());
        if explicit_self_type(cx, func, first_ident) {
            return None;
        }

        let mut checker = BodyLifetimeChecker {
            lifetimes_used_in_body: false,
        };
        checker.visit_expr(body.value);
        if checker.lifetimes_used_in_body {
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

fn allowed_lts_from(named_generics: &[GenericParam<'_>]) -> FxHashSet<LocalDefId> {
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

impl<'a, 'tcx> Visitor<'tcx> for RefVisitor<'a, 'tcx> {
    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        self.lts.push(*lifetime);
    }

    fn visit_poly_trait_ref(&mut self, poly_tref: &'tcx PolyTraitRef<'tcx>) {
        let trait_ref = &poly_tref.trait_ref;
        if let Some(id) = trait_ref.trait_def_id() && lang_items::FN_TRAITS.iter().any(|&item| {
            self.cx.tcx.lang_items().get(item) == Some(id)
        }) {
            let mut sub_visitor = RefVisitor::new(self.cx);
            sub_visitor.visit_trait_ref(trait_ref);
            self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
        } else {
            walk_poly_trait_ref(self, poly_tref);
        }
    }

    fn visit_ty(&mut self, ty: &'tcx Ty<'_>) {
        match ty.kind {
            TyKind::OpaqueDef(item, bounds, _) => {
                let map = self.cx.tcx.hir();
                let item = map.item(item);
                let len = self.lts.len();
                walk_item(self, item);
                self.lts.truncate(len);
                self.lts.extend(bounds.iter().filter_map(|bound| match bound {
                    GenericArg::Lifetime(&l) => Some(l),
                    _ => None,
                }));
            },
            TyKind::BareFn(&BareFnTy { decl, .. }) => {
                let mut sub_visitor = RefVisitor::new(self.cx);
                sub_visitor.visit_fn_decl(decl);
                self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
            },
            TyKind::TraitObject(bounds, lt, _) => {
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
        match *predicate {
            WherePredicate::RegionPredicate(..) => return true,
            WherePredicate::BoundPredicate(ref pred) => {
                // a predicate like F: Trait or F: for<'a> Trait<'a>
                let mut visitor = RefVisitor::new(cx);
                // walk the type F, it may not contain LT refs
                walk_ty(&mut visitor, pred.bounded_ty);
                if !visitor.all_lts().is_empty() {
                    return true;
                }
                // if the bounds define new lifetimes, they are fine to occur
                let allowed_lts = allowed_lts_from(pred.bound_generic_params);
                // now walk the bounds
                for bound in pred.bounds.iter() {
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
            WherePredicate::EqPredicate(ref pred) => {
                let mut visitor = RefVisitor::new(cx);
                walk_ty(&mut visitor, pred.lhs_ty);
                walk_ty(&mut visitor, pred.rhs_ty);
                if !visitor.lts.is_empty() {
                    return true;
                }
            },
        }
    }
    false
}

struct LifetimeChecker<'cx, 'tcx, F> {
    cx: &'cx LateContext<'tcx>,
    map: FxHashMap<Symbol, Span>,
    phantom: std::marker::PhantomData<F>,
}

impl<'cx, 'tcx, F> LifetimeChecker<'cx, 'tcx, F> {
    fn new(cx: &'cx LateContext<'tcx>, map: FxHashMap<Symbol, Span>) -> LifetimeChecker<'cx, 'tcx, F> {
        Self {
            cx,
            map,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'cx, 'tcx, F> Visitor<'tcx> for LifetimeChecker<'cx, 'tcx, F>
where
    F: NestedFilter<'tcx>,
{
    type Map = rustc_middle::hir::map::Map<'tcx>;
    type NestedFilter = F;

    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        self.map.remove(&lifetime.ident.name);
    }

    fn visit_generic_param(&mut self, param: &'tcx GenericParam<'_>) {
        // don't actually visit `<'a>` or `<'a: 'b>`
        // we've already visited the `'a` declarations and
        // don't want to spuriously remove them
        // `'b` in `'a: 'b` is useless unless used elsewhere in
        // a non-lifetime bound
        if let GenericParamKind::Type { .. } = param.kind {
            walk_generic_param(self, param);
        }
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

fn report_extra_lifetimes<'tcx>(cx: &LateContext<'tcx>, func: &'tcx FnDecl<'_>, generics: &'tcx Generics<'_>) {
    let hs = generics
        .params
        .iter()
        .filter_map(|par| match par.kind {
            GenericParamKind::Lifetime {
                kind: LifetimeParamKind::Explicit,
            } => Some((par.name.ident().name, par.span)),
            _ => None,
        })
        .collect();
    let mut checker = LifetimeChecker::<hir_nested_filter::None>::new(cx, hs);

    walk_generics(&mut checker, generics);
    walk_fn_decl(&mut checker, func);

    for &v in checker.map.values() {
        span_lint(
            cx,
            EXTRA_UNUSED_LIFETIMES,
            v,
            "this lifetime isn't used in the function definition",
        );
    }
}

fn report_extra_impl_lifetimes<'tcx>(cx: &LateContext<'tcx>, impl_: &'tcx Impl<'_>) {
    let hs = impl_
        .generics
        .params
        .iter()
        .filter_map(|par| match par.kind {
            GenericParamKind::Lifetime {
                kind: LifetimeParamKind::Explicit,
            } => Some((par.name.ident().name, par.span)),
            _ => None,
        })
        .collect();
    let mut checker = LifetimeChecker::<middle_nested_filter::All>::new(cx, hs);

    walk_generics(&mut checker, impl_.generics);
    if let Some(ref trait_ref) = impl_.of_trait {
        walk_trait_ref(&mut checker, trait_ref);
    }
    walk_ty(&mut checker, impl_.self_ty);
    for item in impl_.items {
        walk_impl_item_ref(&mut checker, item);
    }

    for &v in checker.map.values() {
        span_lint(cx, EXTRA_UNUSED_LIFETIMES, v, "this lifetime isn't used in the impl");
    }
}

struct BodyLifetimeChecker {
    lifetimes_used_in_body: bool,
}

impl<'tcx> Visitor<'tcx> for BodyLifetimeChecker {
    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        if !lifetime.is_anonymous() && lifetime.ident.name != kw::StaticLifetime {
            self.lifetimes_used_in_body = true;
        }
    }
}
