use clippy_utils::diagnostics::span_lint;
use clippy_utils::trait_ref_of_method;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::intravisit::nested_filter::{self as hir_nested_filter, NestedFilter};
use rustc_hir::intravisit::{
    walk_fn_decl, walk_generic_param, walk_generics, walk_impl_item_ref, walk_item, walk_param_bound,
    walk_poly_trait_ref, walk_trait_ref, walk_ty, Visitor,
};
use rustc_hir::FnRetTy::Return;
use rustc_hir::{
    BareFnTy, BodyId, FnDecl, GenericArg, GenericBound, GenericParam, GenericParamKind, Generics, Impl, ImplItem,
    ImplItemKind, Item, ItemKind, LangItem, Lifetime, LifetimeName, ParamName, PolyTraitRef, PredicateOrigin,
    TraitBoundModifier, TraitFn, TraitItem, TraitItemKind, Ty, TyKind, WherePredicate,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter as middle_nested_filter;
use rustc_middle::ty::TyCtxt;
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
    /// - Lifetime bounds such as `impl Foo + 'a` and `T: 'a` must be elided with the
    /// placeholder notation `'_` because the fully elided notation leaves the type bound to `'static`.
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
            check_fn_inner(cx, sig.decl, Some(id), None, generics, item.span, true);
        } else if let ItemKind::Impl(impl_) = item.kind {
            if !item.span.from_expansion() {
                report_extra_impl_lifetimes(cx, impl_);
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(ref sig, id) = item.kind {
            let report_extra_lifetimes = trait_ref_of_method(cx, item.def_id).is_none();
            check_fn_inner(
                cx,
                sig.decl,
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
            check_fn_inner(cx, sig.decl, body, trait_sig, item.generics, item.span, true);
        }
    }
}

/// The lifetime of a &-reference.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum RefLt {
    Unnamed,
    Static,
    Named(LocalDefId),
}

fn check_fn_inner<'tcx>(
    cx: &LateContext<'tcx>,
    decl: &'tcx FnDecl<'_>,
    body: Option<BodyId>,
    trait_sig: Option<&[Ident]>,
    generics: &'tcx Generics<'_>,
    span: Span,
    report_extra_lifetimes: bool,
) {
    if span.from_expansion() || has_where_lifetimes(cx, generics) {
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
                if visitor.lts.iter().any(|lt| matches!(lt, RefLt::Named(_))) {
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
                            if bound.name != LifetimeName::Static && !bound.is_elided() {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    if could_use_elision(cx, decl, body, trait_sig, generics.params) {
        span_lint(
            cx,
            NEEDLESS_LIFETIMES,
            span.with_hi(decl.output.span().hi()),
            "explicit lifetimes given in parameter types where they could be elided \
             (or replaced with `'_` if needed by type declaration)",
        );
    }
    if report_extra_lifetimes {
        self::report_extra_lifetimes(cx, decl, generics);
    }
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

fn could_use_elision<'tcx>(
    cx: &LateContext<'tcx>,
    func: &'tcx FnDecl<'_>,
    body: Option<BodyId>,
    trait_sig: Option<&[Ident]>,
    named_generics: &'tcx [GenericParam<'_>],
) -> bool {
    // There are two scenarios where elision works:
    // * no output references, all input references have different LT
    // * output references, exactly one input reference with same LT
    // All lifetimes must be unnamed, 'static or defined without bounds on the
    // level of the current item.

    // check named LTs
    let allowed_lts = allowed_lts_from(cx.tcx, named_generics);

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
        return false;
    }

    let input_lts = input_visitor.lts;
    let output_lts = output_visitor.lts;

    if let Some(trait_sig) = trait_sig {
        if explicit_self_type(cx, func, trait_sig.first().copied()) {
            return false;
        }
    }

    if let Some(body_id) = body {
        let body = cx.tcx.hir().body(body_id);

        let first_ident = body.params.first().and_then(|param| param.pat.simple_ident());
        if explicit_self_type(cx, func, first_ident) {
            return false;
        }

        let mut checker = BodyLifetimeChecker {
            lifetimes_used_in_body: false,
        };
        checker.visit_expr(body.value);
        if checker.lifetimes_used_in_body {
            return false;
        }
    }

    // check for lifetimes from higher scopes
    for lt in input_lts.iter().chain(output_lts.iter()) {
        if !allowed_lts.contains(lt) {
            return false;
        }
    }

    // check for higher-ranked trait bounds
    if !input_visitor.nested_elision_site_lts.is_empty() || !output_visitor.nested_elision_site_lts.is_empty() {
        let allowed_lts: FxHashSet<_> = allowed_lts
            .iter()
            .filter_map(|lt| match lt {
                RefLt::Named(def_id) => Some(cx.tcx.item_name(def_id.to_def_id())),
                _ => None,
            })
            .collect();
        for lt in input_visitor.nested_elision_site_lts {
            if let RefLt::Named(def_id) = lt {
                if allowed_lts.contains(&cx.tcx.item_name(def_id.to_def_id())) {
                    return false;
                }
            }
        }
        for lt in output_visitor.nested_elision_site_lts {
            if let RefLt::Named(def_id) = lt {
                if allowed_lts.contains(&cx.tcx.item_name(def_id.to_def_id())) {
                    return false;
                }
            }
        }
    }

    // no input lifetimes? easy case!
    if input_lts.is_empty() {
        false
    } else if output_lts.is_empty() {
        // no output lifetimes, check distinctness of input lifetimes

        // only unnamed and static, ok
        let unnamed_and_static = input_lts.iter().all(|lt| *lt == RefLt::Unnamed || *lt == RefLt::Static);
        if unnamed_and_static {
            return false;
        }
        // we have no output reference, so we only need all distinct lifetimes
        input_lts.len() == unique_lifetimes(&input_lts)
    } else {
        // we have output references, so we need one input reference,
        // and all output lifetimes must be the same
        if unique_lifetimes(&output_lts) > 1 {
            return false;
        }
        if input_lts.len() == 1 {
            match (&input_lts[0], &output_lts[0]) {
                (&RefLt::Named(n1), &RefLt::Named(n2)) if n1 == n2 => true,
                (&RefLt::Named(_), &RefLt::Unnamed) => true,
                _ => false, /* already elided, different named lifetimes
                             * or something static going on */
            }
        } else {
            false
        }
    }
}

fn allowed_lts_from(tcx: TyCtxt<'_>, named_generics: &[GenericParam<'_>]) -> FxHashSet<RefLt> {
    let mut allowed_lts = FxHashSet::default();
    for par in named_generics.iter() {
        if let GenericParamKind::Lifetime { .. } = par.kind {
            allowed_lts.insert(RefLt::Named(tcx.hir().local_def_id(par.hir_id)));
        }
    }
    allowed_lts.insert(RefLt::Unnamed);
    allowed_lts.insert(RefLt::Static);
    allowed_lts
}

/// Number of unique lifetimes in the given vector.
#[must_use]
fn unique_lifetimes(lts: &[RefLt]) -> usize {
    lts.iter().collect::<FxHashSet<_>>().len()
}

const CLOSURE_TRAIT_BOUNDS: [LangItem; 3] = [LangItem::Fn, LangItem::FnMut, LangItem::FnOnce];

/// A visitor usable for `rustc_front::visit::walk_ty()`.
struct RefVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    lts: Vec<RefLt>,
    nested_elision_site_lts: Vec<RefLt>,
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

    fn record(&mut self, lifetime: &Option<Lifetime>) {
        if let Some(ref lt) = *lifetime {
            if lt.name == LifetimeName::Static {
                self.lts.push(RefLt::Static);
            } else if let LifetimeName::Param(_, ParamName::Fresh) = lt.name {
                // Fresh lifetimes generated should be ignored.
                self.lts.push(RefLt::Unnamed);
            } else if lt.is_elided() {
                self.lts.push(RefLt::Unnamed);
            } else if let LifetimeName::Param(def_id, _) = lt.name {
                self.lts.push(RefLt::Named(def_id));
            } else {
                self.lts.push(RefLt::Unnamed);
            }
        } else {
            self.lts.push(RefLt::Unnamed);
        }
    }

    fn all_lts(&self) -> Vec<RefLt> {
        self.lts
            .iter()
            .chain(self.nested_elision_site_lts.iter())
            .cloned()
            .collect::<Vec<_>>()
    }

    fn abort(&self) -> bool {
        self.unelided_trait_object_lifetime
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RefVisitor<'a, 'tcx> {
    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        self.record(&Some(*lifetime));
    }

    fn visit_poly_trait_ref(&mut self, poly_tref: &'tcx PolyTraitRef<'tcx>, tbm: TraitBoundModifier) {
        let trait_ref = &poly_tref.trait_ref;
        if CLOSURE_TRAIT_BOUNDS.iter().any(|&item| {
            self.cx
                .tcx
                .lang_items()
                .require(item)
                .map_or(false, |id| Some(id) == trait_ref.trait_def_id())
        }) {
            let mut sub_visitor = RefVisitor::new(self.cx);
            sub_visitor.visit_trait_ref(trait_ref);
            self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
        } else {
            walk_poly_trait_ref(self, poly_tref, tbm);
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
                    GenericArg::Lifetime(l) => Some(if let LifetimeName::Param(def_id, _) = l.name {
                        RefLt::Named(def_id)
                    } else {
                        RefLt::Unnamed
                    }),
                    _ => None,
                }));
            },
            TyKind::BareFn(&BareFnTy { decl, .. }) => {
                let mut sub_visitor = RefVisitor::new(self.cx);
                sub_visitor.visit_fn_decl(decl);
                self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
            },
            TyKind::TraitObject(bounds, ref lt, _) => {
                if !lt.is_elided() {
                    self.unelided_trait_object_lifetime = true;
                }
                for bound in bounds {
                    self.visit_poly_trait_ref(bound, TraitBoundModifier::None);
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
                let allowed_lts = allowed_lts_from(cx.tcx, pred.bound_generic_params);
                // now walk the bounds
                for bound in pred.bounds.iter() {
                    walk_param_bound(&mut visitor, bound);
                }
                // and check that all lifetimes are allowed
                if visitor.all_lts().iter().any(|it| !allowed_lts.contains(it)) {
                    return true;
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
        self.map.remove(&lifetime.name.ident().name);
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
            GenericParamKind::Lifetime { .. } => Some((par.name.ident().name, par.span)),
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
            GenericParamKind::Lifetime { .. } => Some((par.name.ident().name, par.span)),
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
        if lifetime.name.ident().name != kw::UnderscoreLifetime && lifetime.name.ident().name != kw::StaticLifetime {
            self.lifetimes_used_in_body = true;
        }
    }
}
