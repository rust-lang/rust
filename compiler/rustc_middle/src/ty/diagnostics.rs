//! Diagnostics related methods for `Ty`.

use std::fmt::Write;
use std::ops::ControlFlow;

use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{
    Applicability, Diag, DiagArgValue, IntoDiagArg, into_diag_arg_using_display, listify, pluralize,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, AmbigArg, LangItem, PredicateOrigin, WherePredicateKind};
use rustc_span::{BytePos, Span};
use rustc_type_ir::TyKind::*;

use crate::ty::{
    self, AliasTy, Const, ConstKind, FallibleTypeFolder, InferConst, InferTy, Opaque,
    PolyTraitPredicate, Projection, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitor,
};

impl IntoDiagArg for Ty<'_> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        ty::tls::with(|tcx| {
            let ty = tcx.short_string(self, path);
            rustc_errors::DiagArgValue::Str(std::borrow::Cow::Owned(ty))
        })
    }
}

into_diag_arg_using_display! {
    ty::Region<'_>,
}

impl<'tcx> Ty<'tcx> {
    /// Similar to `Ty::is_primitive`, but also considers inferred numeric values to be primitive.
    pub fn is_primitive_ty(self) -> bool {
        matches!(
            self.kind(),
            Bool | Char
                | Str
                | Int(_)
                | Uint(_)
                | Float(_)
                | Infer(
                    InferTy::IntVar(_)
                        | InferTy::FloatVar(_)
                        | InferTy::FreshIntTy(_)
                        | InferTy::FreshFloatTy(_)
                )
        )
    }

    /// Whether the type is succinctly representable as a type instead of just referred to with a
    /// description in error messages. This is used in the main error message.
    pub fn is_simple_ty(self) -> bool {
        match self.kind() {
            Bool
            | Char
            | Str
            | Int(_)
            | Uint(_)
            | Float(_)
            | Infer(
                InferTy::IntVar(_)
                | InferTy::FloatVar(_)
                | InferTy::FreshIntTy(_)
                | InferTy::FreshFloatTy(_),
            ) => true,
            Ref(_, x, _) | Array(x, _) | Slice(x) => x.peel_refs().is_simple_ty(),
            Tuple(tys) if tys.is_empty() => true,
            _ => false,
        }
    }

    /// Whether the type is succinctly representable as a type instead of just referred to with a
    /// description in error messages. This is used in the primary span label. Beyond what
    /// `is_simple_ty` includes, it also accepts ADTs with no type arguments and references to
    /// ADTs with no type arguments.
    pub fn is_simple_text(self) -> bool {
        match self.kind() {
            Adt(_, args) => args.non_erasable_generics().next().is_none(),
            Ref(_, ty, _) => ty.is_simple_text(),
            _ => self.is_simple_ty(),
        }
    }
}

pub trait IsSuggestable<'tcx>: Sized {
    /// Whether this makes sense to suggest in a diagnostic.
    ///
    /// We filter out certain types and constants since they don't provide
    /// meaningful rendered suggestions when pretty-printed. We leave some
    /// nonsense, such as region vars, since those render as `'_` and are
    /// usually okay to reinterpret as elided lifetimes.
    ///
    /// Only if `infer_suggestable` is true, we consider type and const
    /// inference variables to be suggestable.
    fn is_suggestable(self, tcx: TyCtxt<'tcx>, infer_suggestable: bool) -> bool;

    fn make_suggestable(
        self,
        tcx: TyCtxt<'tcx>,
        infer_suggestable: bool,
        placeholder: Option<Ty<'tcx>>,
    ) -> Option<Self>;
}

impl<'tcx, T> IsSuggestable<'tcx> for T
where
    T: TypeVisitable<TyCtxt<'tcx>> + TypeFoldable<TyCtxt<'tcx>>,
{
    #[tracing::instrument(level = "debug", skip(tcx))]
    fn is_suggestable(self, tcx: TyCtxt<'tcx>, infer_suggestable: bool) -> bool {
        self.visit_with(&mut IsSuggestableVisitor { tcx, infer_suggestable }).is_continue()
    }

    fn make_suggestable(
        self,
        tcx: TyCtxt<'tcx>,
        infer_suggestable: bool,
        placeholder: Option<Ty<'tcx>>,
    ) -> Option<T> {
        self.try_fold_with(&mut MakeSuggestableFolder { tcx, infer_suggestable, placeholder }).ok()
    }
}

pub fn suggest_arbitrary_trait_bound<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: &hir::Generics<'_>,
    err: &mut Diag<'_>,
    trait_pred: PolyTraitPredicate<'tcx>,
    associated_ty: Option<(&'static str, Ty<'tcx>)>,
) -> bool {
    if !trait_pred.is_suggestable(tcx, false) {
        return false;
    }

    let param_name = trait_pred.skip_binder().self_ty().to_string();
    let mut constraint = trait_pred.to_string();

    if let Some((name, term)) = associated_ty {
        // FIXME: this case overlaps with code in TyCtxt::note_and_explain_type_err.
        // That should be extracted into a helper function.
        if let Some(stripped) = constraint.strip_suffix('>') {
            constraint = format!("{stripped}, {name} = {term}>");
        } else {
            constraint.push_str(&format!("<{name} = {term}>"));
        }
    }

    let param = generics.params.iter().find(|p| p.name.ident().as_str() == param_name);

    // Skip, there is a param named Self
    if param.is_some() && param_name == "Self" {
        return false;
    }

    // Suggest a where clause bound for a non-type parameter.
    err.span_suggestion_verbose(
        generics.tail_span_for_predicate_suggestion(),
        format!(
            "consider {} `where` clause, but there might be an alternative better way to express \
             this requirement",
            if generics.where_clause_span.is_empty() { "introducing a" } else { "extending the" },
        ),
        format!("{} {constraint}", generics.add_where_or_trailing_comma()),
        Applicability::MaybeIncorrect,
    );
    true
}

#[derive(Debug, Clone, Copy)]
enum SuggestChangingConstraintsMessage<'a> {
    RestrictBoundFurther,
    RestrictType { ty: &'a str },
    RestrictTypeFurther { ty: &'a str },
    RemoveMaybeUnsized,
    ReplaceMaybeUnsizedWithSized,
}

fn suggest_changing_unsized_bound(
    generics: &hir::Generics<'_>,
    suggestions: &mut Vec<(Span, String, String, SuggestChangingConstraintsMessage<'_>)>,
    param: &hir::GenericParam<'_>,
    def_id: Option<DefId>,
) {
    // See if there's a `?Sized` bound that can be removed to suggest that.
    // First look at the `where` clause because we can have `where T: ?Sized`,
    // then look at params.
    for (where_pos, predicate) in generics.predicates.iter().enumerate() {
        let WherePredicateKind::BoundPredicate(predicate) = predicate.kind else {
            continue;
        };
        if !predicate.is_param_bound(param.def_id.to_def_id()) {
            continue;
        };

        let unsized_bounds = predicate
            .bounds
            .iter()
            .enumerate()
            .filter(|(_, bound)| {
                if let hir::GenericBound::Trait(poly) = bound
                    && let hir::BoundPolarity::Maybe(_) = poly.modifiers.polarity
                    && poly.trait_ref.trait_def_id() == def_id
                {
                    true
                } else {
                    false
                }
            })
            .collect::<Vec<_>>();

        if unsized_bounds.is_empty() {
            continue;
        }

        let mut push_suggestion =
            |sp, msg| suggestions.push((sp, "Sized".to_string(), String::new(), msg));

        if predicate.bounds.len() == unsized_bounds.len() {
            // All the bounds are unsized bounds, e.g.
            // `T: ?Sized + ?Sized` or `_: impl ?Sized + ?Sized`,
            // so in this case:
            // - if it's an impl trait predicate suggest changing the
            //   the first bound to sized and removing the rest
            // - Otherwise simply suggest removing the entire predicate
            if predicate.origin == PredicateOrigin::ImplTrait {
                let first_bound = unsized_bounds[0].1;
                let first_bound_span = first_bound.span();
                if first_bound_span.can_be_used_for_suggestions() {
                    let question_span =
                        first_bound_span.with_hi(first_bound_span.lo() + BytePos(1));
                    push_suggestion(
                        question_span,
                        SuggestChangingConstraintsMessage::ReplaceMaybeUnsizedWithSized,
                    );

                    for (pos, _) in unsized_bounds.iter().skip(1) {
                        let sp = generics.span_for_bound_removal(where_pos, *pos);
                        push_suggestion(sp, SuggestChangingConstraintsMessage::RemoveMaybeUnsized);
                    }
                }
            } else {
                let sp = generics.span_for_predicate_removal(where_pos);
                push_suggestion(sp, SuggestChangingConstraintsMessage::RemoveMaybeUnsized);
            }
        } else {
            // Some of the bounds are other than unsized.
            // So push separate removal suggestion for each unsized bound
            for (pos, _) in unsized_bounds {
                let sp = generics.span_for_bound_removal(where_pos, pos);
                push_suggestion(sp, SuggestChangingConstraintsMessage::RemoveMaybeUnsized);
            }
        }
    }
}

/// Suggest restricting a type param with a new bound.
///
/// If `span_to_replace` is provided, then that span will be replaced with the
/// `constraint`. If one wasn't provided, then the full bound will be suggested.
pub fn suggest_constraining_type_param(
    tcx: TyCtxt<'_>,
    generics: &hir::Generics<'_>,
    err: &mut Diag<'_>,
    param_name: &str,
    constraint: &str,
    def_id: Option<DefId>,
    span_to_replace: Option<Span>,
) -> bool {
    suggest_constraining_type_params(
        tcx,
        generics,
        err,
        [(param_name, constraint, def_id)].into_iter(),
        span_to_replace,
    )
}

/// Suggest restricting a type param with a new bound.
pub fn suggest_constraining_type_params<'a>(
    tcx: TyCtxt<'_>,
    generics: &hir::Generics<'_>,
    err: &mut Diag<'_>,
    param_names_and_constraints: impl Iterator<Item = (&'a str, &'a str, Option<DefId>)>,
    span_to_replace: Option<Span>,
) -> bool {
    let mut grouped = FxHashMap::default();
    let mut unstable_suggestion = false;
    param_names_and_constraints.for_each(|(param_name, constraint, def_id)| {
        let stable = match def_id {
            Some(def_id) => match tcx.lookup_stability(def_id) {
                Some(s) => s.level.is_stable(),
                None => true,
            },
            None => true,
        };
        if stable || tcx.sess.is_nightly_build() {
            grouped.entry(param_name).or_insert(Vec::new()).push((
                constraint,
                def_id,
                if stable { "" } else { "unstable " },
            ));
            if !stable {
                unstable_suggestion = true;
            }
        }
    });

    let mut applicability = Applicability::MachineApplicable;
    let mut suggestions = Vec::new();

    for (param_name, mut constraints) in grouped {
        let param = generics.params.iter().find(|p| p.name.ident().as_str() == param_name);
        let Some(param) = param else { return false };

        {
            let mut sized_constraints = constraints.extract_if(.., |(_, def_id, _)| {
                def_id.is_some_and(|def_id| tcx.is_lang_item(def_id, LangItem::Sized))
            });
            if let Some((_, def_id, _)) = sized_constraints.next() {
                applicability = Applicability::MaybeIncorrect;

                err.span_label(param.span, "this type parameter needs to be `Sized`");
                suggest_changing_unsized_bound(generics, &mut suggestions, param, def_id);
            }
        }
        let bound_message = if constraints.iter().any(|(_, def_id, _)| def_id.is_none()) {
            SuggestChangingConstraintsMessage::RestrictBoundFurther
        } else {
            SuggestChangingConstraintsMessage::RestrictTypeFurther { ty: param_name }
        };

        // in the scenario like impl has stricter requirements than trait,
        // we should not suggest restrict bound on the impl, here we double check
        // the whether the param already has the constraint by checking `def_id`
        let bound_trait_defs: Vec<DefId> = generics
            .bounds_for_param(param.def_id)
            .flat_map(|bound| {
                bound.bounds.iter().flat_map(|b| b.trait_ref().and_then(|t| t.trait_def_id()))
            })
            .collect();

        constraints
            .retain(|(_, def_id, _)| def_id.is_none_or(|def| !bound_trait_defs.contains(&def)));

        if constraints.is_empty() {
            continue;
        }

        let mut constraint = constraints.iter().map(|&(c, _, _)| c).collect::<Vec<_>>();
        constraint.sort();
        constraint.dedup();
        let all_known = constraints.iter().all(|&(_, def_id, _)| def_id.is_some());
        let all_stable = constraints.iter().all(|&(_, _, stable)| stable.is_empty());
        let all_unstable = constraints.iter().all(|&(_, _, stable)| !stable.is_empty());
        let post = if all_stable || all_unstable {
            // Don't redundantly say "trait `X`, trait `Y`", instead "traits `X` and `Y`"
            let mut trait_names = constraints
                .iter()
                .map(|&(c, def_id, _)| match def_id {
                    None => format!("`{c}`"),
                    Some(def_id) => format!("`{}`", tcx.item_name(def_id)),
                })
                .collect::<Vec<_>>();
            trait_names.sort();
            trait_names.dedup();
            let n = trait_names.len();
            let stable = if all_stable { "" } else { "unstable " };
            let trait_ = if all_known { format!("trait{}", pluralize!(n)) } else { String::new() };
            let Some(trait_names) = listify(&trait_names, |n| n.to_string()) else { return false };
            format!("{stable}{trait_} {trait_names}")
        } else {
            // We're more explicit when there's a mix of stable and unstable traits.
            let mut trait_names = constraints
                .iter()
                .map(|&(c, def_id, stable)| match def_id {
                    None => format!("`{c}`"),
                    Some(def_id) => format!("{stable}trait `{}`", tcx.item_name(def_id)),
                })
                .collect::<Vec<_>>();
            trait_names.sort();
            trait_names.dedup();
            match listify(&trait_names, |t| t.to_string()) {
                Some(names) => names,
                None => return false,
            }
        };
        let constraint = constraint.join(" + ");
        let mut suggest_restrict = |span, bound_list_non_empty, open_paren_sp| {
            let suggestion = if span_to_replace.is_some() {
                constraint.clone()
            } else if constraint.starts_with('<') {
                constraint.clone()
            } else if bound_list_non_empty {
                format!(" + {constraint}")
            } else {
                format!(" {constraint}")
            };

            if let Some(open_paren_sp) = open_paren_sp {
                suggestions.push((open_paren_sp, post.clone(), "(".to_string(), bound_message));
                suggestions.push((span, post.clone(), format!("){suggestion}"), bound_message));
            } else {
                suggestions.push((span, post.clone(), suggestion, bound_message));
            }
        };

        if let Some(span) = span_to_replace {
            suggest_restrict(span, true, None);
            continue;
        }

        // When the type parameter has been provided bounds
        //
        //    Message:
        //      fn foo<T>(t: T) where T: Foo { ... }
        //                            ^^^^^^
        //                            |
        //                            help: consider further restricting this bound with `+ Bar`
        //
        //    Suggestion:
        //      fn foo<T>(t: T) where T: Foo { ... }
        //                                  ^
        //                                  |
        //                                  replace with: ` + Bar`
        //
        // Or, if user has provided some bounds, suggest restricting them:
        //
        //   fn foo<T: Foo>(t: T) { ... }
        //             ---
        //             |
        //             help: consider further restricting this bound with `+ Bar`
        //
        // Suggestion for tools in this case is:
        //
        //   fn foo<T: Foo>(t: T) { ... }
        //          --
        //          |
        //          replace with: `T: Bar +`

        if let Some((span, open_paren_sp)) = generics.bounds_span_for_suggestions(param.def_id) {
            suggest_restrict(span, true, open_paren_sp);
            continue;
        }

        if generics.has_where_clause_predicates {
            // This part is a bit tricky, because using the `where` clause user can
            // provide zero, one or many bounds for the same type parameter, so we
            // have following cases to consider:
            //
            // When the type parameter has been provided zero bounds
            //
            //    Message:
            //      fn foo<X, Y>(x: X, y: Y) where Y: Foo { ... }
            //             - help: consider restricting this type parameter with `where X: Bar`
            //
            //    Suggestion:
            //      fn foo<X, Y>(x: X, y: Y) where Y: Foo { ... }
            //                                           - insert: `, X: Bar`
            suggestions.push((
                generics.tail_span_for_predicate_suggestion(),
                post,
                constraints.iter().fold(String::new(), |mut string, &(constraint, _, _)| {
                    write!(string, ", {param_name}: {constraint}").unwrap();
                    string
                }),
                SuggestChangingConstraintsMessage::RestrictTypeFurther { ty: param_name },
            ));
            continue;
        }

        // Additionally, there may be no `where` clause but the generic parameter has a default:
        //
        //    Message:
        //      trait Foo<T=()> {... }
        //                - help: consider further restricting this type parameter with `where T: Zar`
        //
        //    Suggestion:
        //      trait Foo<T=()> {... }
        //                     - insert: `where T: Zar`
        if matches!(param.kind, hir::GenericParamKind::Type { default: Some(_), .. }) {
            // If we are here and the where clause span is of non-zero length
            // it means we're dealing with an empty where clause like this:
            //      fn foo<X>(x: X) where { ... }
            // In that case we don't want to add another "where" (Fixes #120838)
            let where_prefix = if generics.where_clause_span.is_empty() { " where" } else { "" };

            // Suggest a bound, but there is no existing `where` clause *and* the type param has a
            // default (`<T=Foo>`), so we suggest adding `where T: Bar`.
            suggestions.push((
                generics.tail_span_for_predicate_suggestion(),
                post,
                format!("{where_prefix} {param_name}: {constraint}"),
                SuggestChangingConstraintsMessage::RestrictTypeFurther { ty: param_name },
            ));
            continue;
        }

        // If user has provided a colon, don't suggest adding another:
        //
        //   fn foo<T:>(t: T) { ... }
        //            - insert: consider restricting this type parameter with `T: Foo`
        if let Some(colon_span) = param.colon_span {
            suggestions.push((
                colon_span.shrink_to_hi(),
                post,
                format!(" {constraint}"),
                SuggestChangingConstraintsMessage::RestrictType { ty: param_name },
            ));
            continue;
        }

        // If user hasn't provided any bounds, suggest adding a new one:
        //
        //   fn foo<T>(t: T) { ... }
        //          - help: consider restricting this type parameter with `T: Foo`
        suggestions.push((
            param.span.shrink_to_hi(),
            post,
            format!(": {constraint}"),
            SuggestChangingConstraintsMessage::RestrictType { ty: param_name },
        ));
    }

    // FIXME: remove the suggestions that are from derive, as the span is not correct
    suggestions = suggestions
        .into_iter()
        .filter(|(span, _, _, _)| !span.in_derive_expansion())
        .collect::<Vec<_>>();
    let suggested = !suggestions.is_empty();
    if suggestions.len() == 1 {
        let (span, post, suggestion, msg) = suggestions.pop().unwrap();
        let msg = match msg {
            SuggestChangingConstraintsMessage::RestrictBoundFurther => {
                format!("consider further restricting this bound")
            }
            SuggestChangingConstraintsMessage::RestrictTypeFurther { ty }
            | SuggestChangingConstraintsMessage::RestrictType { ty }
                if ty.starts_with("impl ") =>
            {
                format!("consider restricting opaque type `{ty}` with {post}")
            }
            SuggestChangingConstraintsMessage::RestrictType { ty } => {
                format!("consider restricting type parameter `{ty}` with {post}")
            }
            SuggestChangingConstraintsMessage::RestrictTypeFurther { ty } => {
                format!("consider further restricting type parameter `{ty}` with {post}")
            }
            SuggestChangingConstraintsMessage::RemoveMaybeUnsized => {
                format!("consider removing the `?Sized` bound to make the type parameter `Sized`")
            }
            SuggestChangingConstraintsMessage::ReplaceMaybeUnsizedWithSized => {
                format!("consider replacing `?Sized` with `Sized`")
            }
        };

        err.span_suggestion_verbose(span, msg, suggestion, applicability);
    } else if suggestions.len() > 1 {
        let post = if unstable_suggestion { " (some of them are unstable traits)" } else { "" };
        err.multipart_suggestion_verbose(
            format!("consider restricting type parameters{post}"),
            suggestions.into_iter().map(|(span, _, suggestion, _)| (span, suggestion)).collect(),
            applicability,
        );
    }

    suggested
}

/// Collect al types that have an implicit `'static` obligation that we could suggest `'_` for.
pub(crate) struct TraitObjectVisitor<'tcx>(pub(crate) Vec<&'tcx hir::Ty<'tcx>>);

impl<'v> hir::intravisit::Visitor<'v> for TraitObjectVisitor<'v> {
    fn visit_ty(&mut self, ty: &'v hir::Ty<'v, AmbigArg>) {
        match ty.kind {
            hir::TyKind::TraitObject(_, tagged_ptr)
                if let hir::Lifetime {
                    kind:
                        hir::LifetimeKind::ImplicitObjectLifetimeDefault | hir::LifetimeKind::Static,
                    ..
                } = tagged_ptr.pointer() =>
            {
                self.0.push(ty.as_unambig_ty())
            }
            hir::TyKind::OpaqueDef(..) => self.0.push(ty.as_unambig_ty()),
            _ => {}
        }
        hir::intravisit::walk_ty(self, ty);
    }
}

pub struct IsSuggestableVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    infer_suggestable: bool,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for IsSuggestableVisitor<'tcx> {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        match *t.kind() {
            Infer(InferTy::TyVar(_)) if self.infer_suggestable => {}

            FnDef(..)
            | Closure(..)
            | Infer(..)
            | Coroutine(..)
            | CoroutineWitness(..)
            | Bound(_, _)
            | Placeholder(_)
            | Error(_) => {
                return ControlFlow::Break(());
            }

            Alias(Opaque, AliasTy { def_id, .. }) => {
                let parent = self.tcx.parent(def_id);
                let parent_ty = self.tcx.type_of(parent).instantiate_identity();
                if let DefKind::TyAlias | DefKind::AssocTy = self.tcx.def_kind(parent)
                    && let Alias(Opaque, AliasTy { def_id: parent_opaque_def_id, .. }) =
                        *parent_ty.kind()
                    && parent_opaque_def_id == def_id
                {
                    // Okay
                } else {
                    return ControlFlow::Break(());
                }
            }

            Alias(Projection, AliasTy { def_id, .. })
                if self.tcx.def_kind(def_id) != DefKind::AssocTy =>
            {
                return ControlFlow::Break(());
            }

            // FIXME: It would be nice to make this not use string manipulation,
            // but it's pretty hard to do this, since `ty::ParamTy` is missing
            // sufficient info to determine if it is synthetic, and we don't
            // always have a convenient way of getting `ty::Generics` at the call
            // sites we invoke `IsSuggestable::is_suggestable`.
            Param(param) if param.name.as_str().starts_with("impl ") => {
                return ControlFlow::Break(());
            }

            _ => {}
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: Const<'tcx>) -> Self::Result {
        match c.kind() {
            ConstKind::Infer(InferConst::Var(_)) if self.infer_suggestable => {}

            ConstKind::Infer(..)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(..)
            | ConstKind::Error(..) => {
                return ControlFlow::Break(());
            }
            _ => {}
        }

        c.super_visit_with(self)
    }
}

pub struct MakeSuggestableFolder<'tcx> {
    tcx: TyCtxt<'tcx>,
    infer_suggestable: bool,
    placeholder: Option<Ty<'tcx>>,
}

impl<'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for MakeSuggestableFolder<'tcx> {
    type Error = ();

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        let t = match *t.kind() {
            Infer(InferTy::TyVar(_)) if self.infer_suggestable => t,

            FnDef(def_id, args) if self.placeholder.is_none() => {
                Ty::new_fn_ptr(self.tcx, self.tcx.fn_sig(def_id).instantiate(self.tcx, args))
            }

            Closure(..)
            | FnDef(..)
            | Infer(..)
            | Coroutine(..)
            | CoroutineWitness(..)
            | Bound(_, _)
            | Placeholder(_)
            | Error(_) => {
                if let Some(placeholder) = self.placeholder {
                    // We replace these with infer (which is passed in from an infcx).
                    placeholder
                } else {
                    return Err(());
                }
            }

            Alias(Opaque, AliasTy { def_id, .. }) => {
                let parent = self.tcx.parent(def_id);
                let parent_ty = self.tcx.type_of(parent).instantiate_identity();
                if let hir::def::DefKind::TyAlias | hir::def::DefKind::AssocTy =
                    self.tcx.def_kind(parent)
                    && let Alias(Opaque, AliasTy { def_id: parent_opaque_def_id, .. }) =
                        *parent_ty.kind()
                    && parent_opaque_def_id == def_id
                {
                    t
                } else {
                    return Err(());
                }
            }

            // FIXME: It would be nice to make this not use string manipulation,
            // but it's pretty hard to do this, since `ty::ParamTy` is missing
            // sufficient info to determine if it is synthetic, and we don't
            // always have a convenient way of getting `ty::Generics` at the call
            // sites we invoke `IsSuggestable::is_suggestable`.
            Param(param) if param.name.as_str().starts_with("impl ") => {
                return Err(());
            }

            _ => t,
        };

        t.try_super_fold_with(self)
    }

    fn try_fold_const(&mut self, c: Const<'tcx>) -> Result<Const<'tcx>, ()> {
        let c = match c.kind() {
            ConstKind::Infer(InferConst::Var(_)) if self.infer_suggestable => c,

            ConstKind::Infer(..)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(..)
            | ConstKind::Error(..) => {
                return Err(());
            }

            _ => c,
        };

        c.try_super_fold_with(self)
    }
}
