use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{snippet, snippet_opt, snippet_with_applicability};
use clippy_utils::{is_from_proc_macro, SpanlessEq, SpanlessHash};
use core::hash::{Hash, Hasher};
use if_chain::if_chain;
use itertools::Itertools;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::unhash::UnhashMap;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{
    GenericArg, GenericBound, Generics, Item, ItemKind, LangItem, Node, Path, PathSegment, PredicateOrigin, QPath,
    TraitBoundModifier, TraitItem, TraitRef, Ty, TyKind, WherePredicate,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{BytePos, Span};
use std::collections::hash_map::Entry;

declare_clippy_lint! {
    /// ### What it does
    /// This lint warns about unnecessary type repetitions in trait bounds
    ///
    /// ### Why is this bad?
    /// Repeating the type for every bound makes the code
    /// less readable than combining the bounds
    ///
    /// ### Example
    /// ```rust
    /// pub fn foo<T>(t: T) where T: Copy, T: Clone {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// pub fn foo<T>(t: T) where T: Copy + Clone {}
    /// ```
    #[clippy::version = "1.38.0"]
    pub TYPE_REPETITION_IN_BOUNDS,
    nursery,
    "types are repeated unnecessarily in trait bounds, use `+` instead of using `T: _, T: _`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for cases where generics or trait objects are being used and multiple
    /// syntax specifications for trait bounds are used simultaneously.
    ///
    /// ### Why is this bad?
    /// Duplicate bounds makes the code
    /// less readable than specifying them only once.
    ///
    /// ### Example
    /// ```rust
    /// fn func<T: Clone + Default>(arg: T) where T: Clone + Default {}
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # mod hidden {
    /// fn func<T: Clone + Default>(arg: T) {}
    /// # }
    ///
    /// // or
    ///
    /// fn func<T>(arg: T) where T: Clone + Default {}
    /// ```
    ///
    /// ```rust
    /// fn foo<T: Default + Default>(bar: T) {}
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo<T: Default>(bar: T) {}
    /// ```
    ///
    /// ```rust
    /// fn foo<T>(bar: T) where T: Default + Default {}
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo<T>(bar: T) where T: Default {}
    /// ```
    #[clippy::version = "1.47.0"]
    pub TRAIT_DUPLICATION_IN_BOUNDS,
    nursery,
    "check if the same trait bounds are specified more than once during a generic declaration"
}

#[derive(Clone)]
pub struct TraitBounds {
    max_trait_bounds: u64,
    msrv: Msrv,
}

impl TraitBounds {
    #[must_use]
    pub fn new(max_trait_bounds: u64, msrv: Msrv) -> Self {
        Self { max_trait_bounds, msrv }
    }
}

impl_lint_pass!(TraitBounds => [TYPE_REPETITION_IN_BOUNDS, TRAIT_DUPLICATION_IN_BOUNDS]);

impl<'tcx> LateLintPass<'tcx> for TraitBounds {
    fn check_generics(&mut self, cx: &LateContext<'tcx>, gen: &'tcx Generics<'_>) {
        self.check_type_repetition(cx, gen);
        check_trait_bound_duplication(cx, gen);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        // special handling for self trait bounds as these are not considered generics
        // ie. trait Foo: Display {}
        if let Item {
            kind: ItemKind::Trait(_, _, _, bounds, ..),
            ..
        } = item
        {
            rollup_traits(cx, bounds, "these bounds contain repeated elements");
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'tcx>) {
        let mut self_bounds_map = FxHashMap::default();

        for predicate in item.generics.predicates {
            if_chain! {
                if let WherePredicate::BoundPredicate(ref bound_predicate) = predicate;
                if bound_predicate.origin != PredicateOrigin::ImplTrait;
                if !bound_predicate.span.from_expansion();
                if let TyKind::Path(QPath::Resolved(_, Path { segments, .. })) = bound_predicate.bounded_ty.kind;
                if let Some(PathSegment {
                    res: Res::SelfTyParam { trait_: def_id }, ..
                }) = segments.first();
                if let Some(
                    Node::Item(
                        Item {
                            kind: ItemKind::Trait(_, _, _, self_bounds, _),
                            .. }
                        )
                    ) = cx.tcx.hir().get_if_local(*def_id);
                then {
                    if self_bounds_map.is_empty() {
                        for bound in *self_bounds {
                            let Some((self_res, self_segments, _)) = get_trait_info_from_bound(bound) else { continue };
                            self_bounds_map.insert(self_res, self_segments);
                        }
                    }

                    bound_predicate
                        .bounds
                        .iter()
                        .filter_map(get_trait_info_from_bound)
                        .for_each(|(trait_item_res, trait_item_segments, span)| {
                            if let Some(self_segments) = self_bounds_map.get(&trait_item_res) {
                                if SpanlessEq::new(cx).eq_path_segments(self_segments, trait_item_segments) {
                                    span_lint_and_help(
                                        cx,
                                        TRAIT_DUPLICATION_IN_BOUNDS,
                                        span,
                                        "this trait bound is already specified in trait declaration",
                                        None,
                                        "consider removing this trait bound",
                                    );
                                }
                            }
                        });
                }
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx>) {
        if_chain! {
            if let TyKind::Ref(.., mut_ty) = &ty.kind;
            if let TyKind::TraitObject(bounds, ..) = mut_ty.ty.kind;
            if bounds.len() > 2;
            then {

                // Build up a hash of every trait we've seen
                // When we see a trait for the first time, add it to unique_traits
                // so we can later use it to build a string of all traits exactly once, without duplicates

                let mut seen_def_ids = FxHashSet::default();
                let mut unique_traits = Vec::new();

                // Iterate the bounds and add them to our seen hash
                // If we haven't yet seen it, add it to the fixed traits
                for bound in bounds {
                    let Some(def_id) = bound.trait_ref.trait_def_id() else { continue; };

                    let new_trait = seen_def_ids.insert(def_id);

                    if new_trait {
                        unique_traits.push(bound);
                    }
                }

                // If the number of unique traits isn't the same as the number of traits in the bounds,
                // there must be 1 or more duplicates
                if bounds.len() != unique_traits.len() {
                    let mut bounds_span = bounds[0].span;

                    for bound in bounds.iter().skip(1) {
                        bounds_span = bounds_span.to(bound.span);
                    }

                    let fixed_trait_snippet = unique_traits
                        .iter()
                        .filter_map(|b| snippet_opt(cx, b.span))
                        .collect::<Vec<_>>()
                        .join(" + ");

                    span_lint_and_sugg(
                        cx,
                        TRAIT_DUPLICATION_IN_BOUNDS,
                        bounds_span,
                        "this trait bound is already specified in trait declaration",
                        "try",
                        fixed_trait_snippet,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

impl TraitBounds {
    /// Is the given bound a `?Sized` bound, and is combining it (i.e. `T: X + ?Sized`) an error on
    /// this MSRV? See <https://github.com/rust-lang/rust-clippy/issues/8772> for details.
    fn cannot_combine_maybe_bound(&self, cx: &LateContext<'_>, bound: &GenericBound<'_>) -> bool {
        if !self.msrv.meets(msrvs::MAYBE_BOUND_IN_WHERE)
            && let GenericBound::Trait(tr, TraitBoundModifier::Maybe) = bound
        {
            cx.tcx.lang_items().get(LangItem::Sized) == tr.trait_ref.path.res.opt_def_id()
        } else {
            false
        }
    }

    fn check_type_repetition<'tcx>(&self, cx: &LateContext<'tcx>, gen: &'tcx Generics<'_>) {
        struct SpanlessTy<'cx, 'tcx> {
            ty: &'tcx Ty<'tcx>,
            cx: &'cx LateContext<'tcx>,
        }
        impl PartialEq for SpanlessTy<'_, '_> {
            fn eq(&self, other: &Self) -> bool {
                let mut eq = SpanlessEq::new(self.cx);
                eq.inter_expr().eq_ty(self.ty, other.ty)
            }
        }
        impl Hash for SpanlessTy<'_, '_> {
            fn hash<H: Hasher>(&self, h: &mut H) {
                let mut t = SpanlessHash::new(self.cx);
                t.hash_ty(self.ty);
                h.write_u64(t.finish());
            }
        }
        impl Eq for SpanlessTy<'_, '_> {}

        if gen.span.from_expansion() {
            return;
        }
        let mut map: UnhashMap<SpanlessTy<'_, '_>, Vec<&GenericBound<'_>>> = UnhashMap::default();
        let mut applicability = Applicability::MaybeIncorrect;
        for bound in gen.predicates {
            if_chain! {
                if let WherePredicate::BoundPredicate(ref p) = bound;
                if p.origin != PredicateOrigin::ImplTrait;
                if p.bounds.len() as u64 <= self.max_trait_bounds;
                if !p.span.from_expansion();
                let bounds = p.bounds.iter().filter(|b| !self.cannot_combine_maybe_bound(cx, b)).collect::<Vec<_>>();
                if !bounds.is_empty();
                if let Some(ref v) = map.insert(SpanlessTy { ty: p.bounded_ty, cx }, bounds);
                if !is_from_proc_macro(cx, p.bounded_ty);
                then {
                    let trait_bounds = v
                        .iter()
                        .copied()
                        .chain(p.bounds.iter())
                        .filter_map(get_trait_info_from_bound)
                        .map(|(_, _, span)| snippet_with_applicability(cx, span, "..", &mut applicability))
                        .join(" + ");
                    let hint_string = format!(
                        "consider combining the bounds: `{}: {trait_bounds}`",
                        snippet(cx, p.bounded_ty.span, "_"),
                    );
                    span_lint_and_help(
                        cx,
                        TYPE_REPETITION_IN_BOUNDS,
                        p.span,
                        "this type has already been used as a bound predicate",
                        None,
                        &hint_string,
                    );
                }
            }
        }
    }
}

fn check_trait_bound_duplication(cx: &LateContext<'_>, gen: &'_ Generics<'_>) {
    if gen.span.from_expansion() {
        return;
    }

    // Explanation:
    // fn bad_foo<T: Clone + Default, Z: Copy>(arg0: T, arg1: Z)
    // where T: Clone + Default, { unimplemented!(); }
    //       ^^^^^^^^^^^^^^^^^^
    //       |
    // collects each of these where clauses into a set keyed by generic name and comparable trait
    // eg. (T, Clone)
    let where_predicates = gen
        .predicates
        .iter()
        .filter_map(|pred| {
            if_chain! {
                if pred.in_where_clause();
                if let WherePredicate::BoundPredicate(bound_predicate) = pred;
                if let TyKind::Path(QPath::Resolved(_, path)) =  bound_predicate.bounded_ty.kind;
                then {
                    return Some(
                        rollup_traits(cx, bound_predicate.bounds, "these where clauses contain repeated elements")
                        .into_iter().map(|(trait_ref, _)| (path.res, trait_ref)))
                }
            }
            None
        })
        .flatten()
        .collect::<FxHashSet<_>>();

    // Explanation:
    // fn bad_foo<T: Clone + Default, Z: Copy>(arg0: T, arg1: Z) ...
    //            ^^^^^^^^^^^^^^^^^^  ^^^^^^^
    //            |
    // compare trait bounds keyed by generic name and comparable trait to collected where
    // predicates eg. (T, Clone)
    for predicate in gen.predicates.iter().filter(|pred| !pred.in_where_clause()) {
        if_chain! {
            if let WherePredicate::BoundPredicate(bound_predicate) = predicate;
            if bound_predicate.origin != PredicateOrigin::ImplTrait;
            if !bound_predicate.span.from_expansion();
            if let TyKind::Path(QPath::Resolved(_, path)) =  bound_predicate.bounded_ty.kind;
            then {
                let traits = rollup_traits(cx, bound_predicate.bounds, "these bounds contain repeated elements");
                for (trait_ref, span) in traits {
                    let key = (path.res, trait_ref);
                    if where_predicates.contains(&key) {
                        span_lint_and_help(
                            cx,
                            TRAIT_DUPLICATION_IN_BOUNDS,
                            span,
                            "this trait bound is already specified in the where clause",
                            None,
                            "consider removing this trait bound",
                        );
                    }
                }
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ComparableTraitRef(Res, Vec<Res>);
impl Default for ComparableTraitRef {
    fn default() -> Self {
        Self(Res::Err, Vec::new())
    }
}

fn get_trait_info_from_bound<'a>(bound: &'a GenericBound<'_>) -> Option<(Res, &'a [PathSegment<'a>], Span)> {
    if let GenericBound::Trait(t, tbm) = bound {
        let trait_path = t.trait_ref.path;
        let trait_span = {
            let path_span = trait_path.span;
            if let TraitBoundModifier::Maybe = tbm {
                path_span.with_lo(path_span.lo() - BytePos(1)) // include the `?`
            } else {
                path_span
            }
        };
        Some((trait_path.res, trait_path.segments, trait_span))
    } else {
        None
    }
}

// FIXME: ComparableTraitRef does not support nested bounds needed for associated_type_bounds
fn into_comparable_trait_ref(trait_ref: &TraitRef<'_>) -> ComparableTraitRef {
    ComparableTraitRef(
        trait_ref.path.res,
        trait_ref
            .path
            .segments
            .iter()
            .filter_map(|segment| {
                // get trait bound type arguments
                Some(segment.args?.args.iter().filter_map(|arg| {
                    if_chain! {
                        if let GenericArg::Type(ty) = arg;
                        if let TyKind::Path(QPath::Resolved(_, path)) = ty.kind;
                        then { return Some(path.res) }
                    }
                    None
                }))
            })
            .flatten()
            .collect(),
    )
}

fn rollup_traits(cx: &LateContext<'_>, bounds: &[GenericBound<'_>], msg: &str) -> Vec<(ComparableTraitRef, Span)> {
    let mut map = FxHashMap::default();
    let mut repeated_res = false;

    let only_comparable_trait_refs = |bound: &GenericBound<'_>| {
        if let GenericBound::Trait(t, _) = bound {
            Some((into_comparable_trait_ref(&t.trait_ref), t.span))
        } else {
            None
        }
    };

    let mut i = 0usize;
    for bound in bounds.iter().filter_map(only_comparable_trait_refs) {
        let (comparable_bound, span_direct) = bound;
        match map.entry(comparable_bound) {
            Entry::Occupied(_) => repeated_res = true,
            Entry::Vacant(e) => {
                e.insert((span_direct, i));
                i += 1;
            },
        }
    }

    // Put bounds in source order
    let mut comparable_bounds = vec![Default::default(); map.len()];
    for (k, (v, i)) in map {
        comparable_bounds[i] = (k, v);
    }

    if_chain! {
        if repeated_res;
        if let [first_trait, .., last_trait] = bounds;
        then {
            let all_trait_span = first_trait.span().to(last_trait.span());

            let traits = comparable_bounds.iter()
                .filter_map(|&(_, span)| snippet_opt(cx, span))
                .collect::<Vec<_>>();
            let traits = traits.join(" + ");

            span_lint_and_sugg(
                cx,
                TRAIT_DUPLICATION_IN_BOUNDS,
                all_trait_span,
                msg,
                "try",
                traits,
                Applicability::MachineApplicable
            );
        }
    }

    comparable_bounds
}
