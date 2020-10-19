use crate::utils::paths;
use crate::utils::{get_trait_def_id, in_macro, span_lint, trait_ref_of_method};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::intravisit::{
    walk_fn_decl, walk_generic_param, walk_generics, walk_item, walk_param_bound, walk_poly_trait_ref, walk_ty,
    NestedVisitorMap, Visitor,
};
use rustc_hir::FnRetTy::Return;
use rustc_hir::{
    BareFnTy, BodyId, FnDecl, GenericArg, GenericBound, GenericParam, GenericParamKind, Generics, ImplItem,
    ImplItemKind, Item, ItemKind, Lifetime, LifetimeName, ParamName, PolyTraitRef, TraitBoundModifier, TraitFn,
    TraitItem, TraitItemKind, Ty, TyKind, WhereClause, WherePredicate,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::symbol::{kw, Symbol};
use std::iter::FromIterator;

declare_clippy_lint! {
    /// **What it does:** Checks for lifetime annotations which can be removed by
    /// relying on lifetime elision.
    ///
    /// **Why is this bad?** The additional lifetimes make the code look more
    /// complicated, while there is nothing out of the ordinary going on. Removing
    /// them leads to more readable code.
    ///
    /// **Known problems:**
    /// - We bail out if the function has a `where` clause where lifetimes
    /// are mentioned due to potenial false positives.
    /// - Lifetime bounds such as `impl Foo + 'a` and `T: 'a` must be elided with the
    /// placeholder notation `'_` because the fully elided notation leaves the type bound to `'static`.
    ///
    /// **Example:**
    /// ```rust
    /// // Bad: unnecessary lifetime annotations
    /// fn in_and_out<'a>(x: &'a u8, y: u8) -> &'a u8 {
    ///     x
    /// }
    ///
    /// // Good
    /// fn elided(x: &u8, y: u8) -> &u8 {
    ///     x
    /// }
    /// ```
    pub NEEDLESS_LIFETIMES,
    complexity,
    "using explicit lifetimes for references in function arguments when elision rules \
     would allow omitting them"
}

declare_clippy_lint! {
    /// **What it does:** Checks for lifetimes in generics that are never used
    /// anywhere else.
    ///
    /// **Why is this bad?** The additional lifetimes make the code look more
    /// complicated, while there is nothing out of the ordinary going on. Removing
    /// them leads to more readable code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// // Bad: unnecessary lifetimes
    /// fn unused_lifetime<'a>(x: u8) {
    ///     // ..
    /// }
    ///
    /// // Good
    /// fn no_lifetime(x: u8) {
    ///     // ...
    /// }
    /// ```
    pub EXTRA_UNUSED_LIFETIMES,
    complexity,
    "unused lifetimes in function definitions"
}

declare_lint_pass!(Lifetimes => [NEEDLESS_LIFETIMES, EXTRA_UNUSED_LIFETIMES]);

impl<'tcx> LateLintPass<'tcx> for Lifetimes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Fn(ref sig, ref generics, id) = item.kind {
            check_fn_inner(cx, &sig.decl, Some(id), generics, item.span, true);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if let ImplItemKind::Fn(ref sig, id) = item.kind {
            let report_extra_lifetimes = trait_ref_of_method(cx, item.hir_id).is_none();
            check_fn_inner(
                cx,
                &sig.decl,
                Some(id),
                &item.generics,
                item.span,
                report_extra_lifetimes,
            );
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(ref sig, ref body) = item.kind {
            let body = match *body {
                TraitFn::Required(_) => None,
                TraitFn::Provided(id) => Some(id),
            };
            check_fn_inner(cx, &sig.decl, body, &item.generics, item.span, true);
        }
    }
}

/// The lifetime of a &-reference.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
enum RefLt {
    Unnamed,
    Static,
    Named(Symbol),
}

fn check_fn_inner<'tcx>(
    cx: &LateContext<'tcx>,
    decl: &'tcx FnDecl<'_>,
    body: Option<BodyId>,
    generics: &'tcx Generics<'_>,
    span: Span,
    report_extra_lifetimes: bool,
) {
    if in_macro(span) || has_where_lifetimes(cx, &generics.where_clause) {
        return;
    }

    let types = generics
        .params
        .iter()
        .filter(|param| matches!(param.kind, GenericParamKind::Type { .. }));
    for typ in types {
        for bound in typ.bounds {
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
                if let Some(ref params) = *params {
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
    if could_use_elision(cx, decl, body, &generics.params) {
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

fn could_use_elision<'tcx>(
    cx: &LateContext<'tcx>,
    func: &'tcx FnDecl<'_>,
    body: Option<BodyId>,
    named_generics: &'tcx [GenericParam<'_>],
) -> bool {
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
    if let Return(ref ty) = func.output {
        output_visitor.visit_ty(ty);
    }
    for lt in named_generics {
        input_visitor.visit_generic_param(lt)
    }

    if input_visitor.abort() || output_visitor.abort() {
        return false;
    }

    if allowed_lts
        .intersection(&FxHashSet::from_iter(
            input_visitor
                .nested_elision_site_lts
                .iter()
                .chain(output_visitor.nested_elision_site_lts.iter())
                .cloned()
                .filter(|v| matches!(v, RefLt::Named(_))),
        ))
        .next()
        .is_some()
    {
        return false;
    }

    let input_lts = input_visitor.lts;
    let output_lts = output_visitor.lts;

    if let Some(body_id) = body {
        let mut checker = BodyLifetimeChecker {
            lifetimes_used_in_body: false,
        };
        checker.visit_expr(&cx.tcx.hir().body(body_id).value);
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

fn allowed_lts_from(named_generics: &[GenericParam<'_>]) -> FxHashSet<RefLt> {
    let mut allowed_lts = FxHashSet::default();
    for par in named_generics.iter() {
        if let GenericParamKind::Lifetime { .. } = par.kind {
            if par.bounds.is_empty() {
                allowed_lts.insert(RefLt::Named(par.name.ident().name));
            }
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

const CLOSURE_TRAIT_BOUNDS: [&[&str]; 3] = [&paths::FN, &paths::FN_MUT, &paths::FN_ONCE];

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
            } else if let LifetimeName::Param(ParamName::Fresh(_)) = lt.name {
                // Fresh lifetimes generated should be ignored.
            } else if lt.is_elided() {
                self.lts.push(RefLt::Unnamed);
            } else {
                self.lts.push(RefLt::Named(lt.name.ident().name));
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
    type Map = Map<'tcx>;

    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        self.record(&Some(*lifetime));
    }

    fn visit_poly_trait_ref(&mut self, poly_tref: &'tcx PolyTraitRef<'tcx>, tbm: TraitBoundModifier) {
        let trait_ref = &poly_tref.trait_ref;
        if CLOSURE_TRAIT_BOUNDS
            .iter()
            .any(|trait_path| trait_ref.trait_def_id() == get_trait_def_id(self.cx, trait_path))
        {
            let mut sub_visitor = RefVisitor::new(self.cx);
            sub_visitor.visit_trait_ref(trait_ref);
            self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
        } else {
            walk_poly_trait_ref(self, poly_tref, tbm);
        }
    }

    fn visit_ty(&mut self, ty: &'tcx Ty<'_>) {
        match ty.kind {
            TyKind::OpaqueDef(item, _) => {
                let map = self.cx.tcx.hir();
                let item = map.expect_item(item.id);
                walk_item(self, item);
                walk_ty(self, ty);
            },
            TyKind::BareFn(&BareFnTy { decl, .. }) => {
                let mut sub_visitor = RefVisitor::new(self.cx);
                sub_visitor.visit_fn_decl(decl);
                self.nested_elision_site_lts.append(&mut sub_visitor.all_lts());
                return;
            },
            TyKind::TraitObject(bounds, ref lt) => {
                if !lt.is_elided() {
                    self.unelided_trait_object_lifetime = true;
                }
                for bound in bounds {
                    self.visit_poly_trait_ref(bound, TraitBoundModifier::None);
                }
                return;
            },
            _ => (),
        }
        walk_ty(self, ty);
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Are any lifetimes mentioned in the `where` clause? If so, we don't try to
/// reason about elision.
fn has_where_lifetimes<'tcx>(cx: &LateContext<'tcx>, where_clause: &'tcx WhereClause<'_>) -> bool {
    for predicate in where_clause.predicates {
        match *predicate {
            WherePredicate::RegionPredicate(..) => return true,
            WherePredicate::BoundPredicate(ref pred) => {
                // a predicate like F: Trait or F: for<'a> Trait<'a>
                let mut visitor = RefVisitor::new(cx);
                // walk the type F, it may not contain LT refs
                walk_ty(&mut visitor, &pred.bounded_ty);
                if !visitor.all_lts().is_empty() {
                    return true;
                }
                // if the bounds define new lifetimes, they are fine to occur
                let allowed_lts = allowed_lts_from(&pred.bound_generic_params);
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
                walk_ty(&mut visitor, &pred.lhs_ty);
                walk_ty(&mut visitor, &pred.rhs_ty);
                if !visitor.lts.is_empty() {
                    return true;
                }
            },
        }
    }
    false
}

struct LifetimeChecker {
    map: FxHashMap<Symbol, Span>,
}

impl<'tcx> Visitor<'tcx> for LifetimeChecker {
    type Map = Map<'tcx>;

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
            walk_generic_param(self, param)
        }
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
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
    let mut checker = LifetimeChecker { map: hs };

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

struct BodyLifetimeChecker {
    lifetimes_used_in_body: bool,
}

impl<'tcx> Visitor<'tcx> for BodyLifetimeChecker {
    type Map = Map<'tcx>;

    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'tcx Lifetime) {
        if lifetime.name.ident().name != kw::Invalid && lifetime.name.ident().name != kw::StaticLifetime {
            self.lifetimes_used_in_body = true;
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
