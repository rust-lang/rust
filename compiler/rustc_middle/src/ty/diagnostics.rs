//! Diagnostics related methods for `Ty`.

use std::ops::ControlFlow;

use crate::ty::{
    AliasTy, Const, ConstKind, FallibleTypeFolder, InferConst, InferTy, Opaque, PolyTraitPredicate,
    Projection, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable,
    TypeVisitor,
};

use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diagnostic, DiagnosticArgValue, IntoDiagnosticArg};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::WherePredicate;
use rustc_span::Span;
use rustc_type_ir::sty::TyKind::*;

impl<'tcx> IntoDiagnosticArg for Ty<'tcx> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.to_string().into_diagnostic_arg()
    }
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
            Adt(_, substs) => substs.non_erasable_generics().next().is_none(),
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

    fn make_suggestable(self, tcx: TyCtxt<'tcx>, infer_suggestable: bool) -> Option<Self>;
}

impl<'tcx, T> IsSuggestable<'tcx> for T
where
    T: TypeVisitable<TyCtxt<'tcx>> + TypeFoldable<TyCtxt<'tcx>>,
{
    fn is_suggestable(self, tcx: TyCtxt<'tcx>, infer_suggestable: bool) -> bool {
        self.visit_with(&mut IsSuggestableVisitor { tcx, infer_suggestable }).is_continue()
    }

    fn make_suggestable(self, tcx: TyCtxt<'tcx>, infer_suggestable: bool) -> Option<T> {
        self.try_fold_with(&mut MakeSuggestableFolder { tcx, infer_suggestable }).ok()
    }
}

pub fn suggest_arbitrary_trait_bound<'tcx>(
    tcx: TyCtxt<'tcx>,
    generics: &hir::Generics<'_>,
    err: &mut Diagnostic,
    trait_pred: PolyTraitPredicate<'tcx>,
    associated_ty: Option<(&'static str, Ty<'tcx>)>,
) -> bool {
    if !trait_pred.is_suggestable(tcx, false) {
        return false;
    }

    let param_name = trait_pred.skip_binder().self_ty().to_string();
    let mut constraint = trait_pred.print_modifiers_and_trait_path().to_string();

    if let Some((name, term)) = associated_ty {
        // FIXME: this case overlaps with code in TyCtxt::note_and_explain_type_err.
        // That should be extracted into a helper function.
        if constraint.ends_with('>') {
            constraint = format!("{}, {} = {}>", &constraint[..constraint.len() - 1], name, term);
        } else {
            constraint.push_str(&format!("<{} = {}>", name, term));
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
        &format!(
            "consider {} `where` clause, but there might be an alternative better way to express \
             this requirement",
            if generics.where_clause_span.is_empty() { "introducing a" } else { "extending the" },
        ),
        format!("{} {}: {}", generics.add_where_or_trailing_comma(), param_name, constraint),
        Applicability::MaybeIncorrect,
    );
    true
}

#[derive(Debug)]
enum SuggestChangingConstraintsMessage<'a> {
    RestrictBoundFurther,
    RestrictType { ty: &'a str },
    RestrictTypeFurther { ty: &'a str },
    RemovingQSized,
}

fn suggest_removing_unsized_bound(
    generics: &hir::Generics<'_>,
    suggestions: &mut Vec<(Span, String, SuggestChangingConstraintsMessage<'_>)>,
    param: &hir::GenericParam<'_>,
    def_id: Option<DefId>,
) {
    // See if there's a `?Sized` bound that can be removed to suggest that.
    // First look at the `where` clause because we can have `where T: ?Sized`,
    // then look at params.
    for (where_pos, predicate) in generics.predicates.iter().enumerate() {
        let WherePredicate::BoundPredicate(predicate) = predicate else {
            continue;
        };
        if !predicate.is_param_bound(param.def_id.to_def_id()) {
            continue;
        };

        for (pos, bound) in predicate.bounds.iter().enumerate() {
            let hir::GenericBound::Trait(poly, hir::TraitBoundModifier::Maybe) = bound else {
                continue;
            };
            if poly.trait_ref.trait_def_id() != def_id {
                continue;
            }
            let sp = generics.span_for_bound_removal(where_pos, pos);
            suggestions.push((
                sp,
                String::new(),
                SuggestChangingConstraintsMessage::RemovingQSized,
            ));
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
    err: &mut Diagnostic,
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
    err: &mut Diagnostic,
    param_names_and_constraints: impl Iterator<Item = (&'a str, &'a str, Option<DefId>)>,
    span_to_replace: Option<Span>,
) -> bool {
    let mut grouped = FxHashMap::default();
    param_names_and_constraints.for_each(|(param_name, constraint, def_id)| {
        grouped.entry(param_name).or_insert(Vec::new()).push((constraint, def_id))
    });

    let mut applicability = Applicability::MachineApplicable;
    let mut suggestions = Vec::new();

    for (param_name, mut constraints) in grouped {
        let param = generics.params.iter().find(|p| p.name.ident().as_str() == param_name);
        let Some(param) = param else { return false };

        {
            let mut sized_constraints =
                constraints.drain_filter(|(_, def_id)| *def_id == tcx.lang_items().sized_trait());
            if let Some((constraint, def_id)) = sized_constraints.next() {
                applicability = Applicability::MaybeIncorrect;

                err.span_label(
                    param.span,
                    &format!("this type parameter needs to be `{}`", constraint),
                );
                suggest_removing_unsized_bound(generics, &mut suggestions, param, def_id);
            }
        }

        if constraints.is_empty() {
            continue;
        }

        let mut constraint = constraints.iter().map(|&(c, _)| c).collect::<Vec<_>>();
        constraint.sort();
        constraint.dedup();
        let constraint = constraint.join(" + ");
        let mut suggest_restrict = |span, bound_list_non_empty| {
            suggestions.push((
                span,
                if span_to_replace.is_some() {
                    constraint.clone()
                } else if bound_list_non_empty {
                    format!(" + {}", constraint)
                } else {
                    format!(" {}", constraint)
                },
                SuggestChangingConstraintsMessage::RestrictBoundFurther,
            ))
        };

        if let Some(span) = span_to_replace {
            suggest_restrict(span, true);
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
        if let Some(span) = generics.bounds_span_for_suggestions(param.def_id) {
            suggest_restrict(span, true);
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
                constraints
                    .iter()
                    .map(|&(constraint, _)| format!(", {}: {}", param_name, constraint))
                    .collect::<String>(),
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
            // Suggest a bound, but there is no existing `where` clause *and* the type param has a
            // default (`<T=Foo>`), so we suggest adding `where T: Bar`.
            suggestions.push((
                generics.tail_span_for_predicate_suggestion(),
                format!(" where {}: {}", param_name, constraint),
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
                format!(" {}", constraint),
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
            format!(": {}", constraint),
            SuggestChangingConstraintsMessage::RestrictType { ty: param_name },
        ));
    }

    // FIXME: remove the suggestions that are from derive, as the span is not correct
    suggestions = suggestions
        .into_iter()
        .filter(|(span, _, _)| !span.in_derive_expansion())
        .collect::<Vec<_>>();

    if suggestions.len() == 1 {
        let (span, suggestion, msg) = suggestions.pop().unwrap();

        let s;
        let msg = match msg {
            SuggestChangingConstraintsMessage::RestrictBoundFurther => {
                "consider further restricting this bound"
            }
            SuggestChangingConstraintsMessage::RestrictType { ty } => {
                s = format!("consider restricting type parameter `{}`", ty);
                &s
            }
            SuggestChangingConstraintsMessage::RestrictTypeFurther { ty } => {
                s = format!("consider further restricting type parameter `{}`", ty);
                &s
            }
            SuggestChangingConstraintsMessage::RemovingQSized => {
                "consider removing the `?Sized` bound to make the type parameter `Sized`"
            }
        };

        err.span_suggestion_verbose(span, msg, suggestion, applicability);
    } else if suggestions.len() > 1 {
        err.multipart_suggestion_verbose(
            "consider restricting type parameters",
            suggestions.into_iter().map(|(span, suggestion, _)| (span, suggestion)).collect(),
            applicability,
        );
    }

    true
}

/// Collect al types that have an implicit `'static` obligation that we could suggest `'_` for.
pub struct TraitObjectVisitor<'tcx>(pub Vec<&'tcx hir::Ty<'tcx>>, pub crate::hir::map::Map<'tcx>);

impl<'v> hir::intravisit::Visitor<'v> for TraitObjectVisitor<'v> {
    fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
        match ty.kind {
            hir::TyKind::TraitObject(
                _,
                hir::Lifetime {
                    res:
                        hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Static,
                    ..
                },
                _,
            ) => {
                self.0.push(ty);
            }
            hir::TyKind::OpaqueDef(item_id, _, _) => {
                self.0.push(ty);
                let item = self.1.item(item_id);
                hir::intravisit::walk_item(self, item);
            }
            _ => {}
        }
        hir::intravisit::walk_ty(self, ty);
    }
}

/// Collect al types that have an implicit `'static` obligation that we could suggest `'_` for.
pub struct StaticLifetimeVisitor<'tcx>(pub Vec<Span>, pub crate::hir::map::Map<'tcx>);

impl<'v> hir::intravisit::Visitor<'v> for StaticLifetimeVisitor<'v> {
    fn visit_lifetime(&mut self, lt: &'v hir::Lifetime) {
        if let hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Static = lt.res
        {
            self.0.push(lt.ident.span);
        }
    }
}

pub struct IsSuggestableVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    infer_suggestable: bool,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for IsSuggestableVisitor<'tcx> {
    type BreakTy = ();

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *t.kind() {
            Infer(InferTy::TyVar(_)) if self.infer_suggestable => {}

            FnDef(..)
            | Closure(..)
            | Infer(..)
            | Generator(..)
            | GeneratorWitness(..)
            | Bound(_, _)
            | Placeholder(_)
            | Error(_) => {
                return ControlFlow::Break(());
            }

            Alias(Opaque, AliasTy { def_id, .. }) => {
                let parent = self.tcx.parent(def_id);
                let parent_ty = self.tcx.type_of(parent).subst_identity();
                if let DefKind::TyAlias | DefKind::AssocTy = self.tcx.def_kind(parent)
                    && let Alias(Opaque, AliasTy { def_id: parent_opaque_def_id, .. }) = *parent_ty.kind()
                    && parent_opaque_def_id == def_id
                {
                    // Okay
                } else {
                    return ControlFlow::Break(());
                }
            }

            Alias(Projection, AliasTy { def_id, .. }) => {
                if self.tcx.def_kind(def_id) != DefKind::AssocTy {
                    return ControlFlow::Break(());
                }
            }

            Param(param) => {
                // FIXME: It would be nice to make this not use string manipulation,
                // but it's pretty hard to do this, since `ty::ParamTy` is missing
                // sufficient info to determine if it is synthetic, and we don't
                // always have a convenient way of getting `ty::Generics` at the call
                // sites we invoke `IsSuggestable::is_suggestable`.
                if param.name.as_str().starts_with("impl ") {
                    return ControlFlow::Break(());
                }
            }

            _ => {}
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: Const<'tcx>) -> ControlFlow<Self::BreakTy> {
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
}

impl<'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for MakeSuggestableFolder<'tcx> {
    type Error = ();

    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        let t = match *t.kind() {
            Infer(InferTy::TyVar(_)) if self.infer_suggestable => t,

            FnDef(def_id, substs) => {
                self.tcx.mk_fn_ptr(self.tcx.fn_sig(def_id).subst(self.tcx, substs))
            }

            // FIXME(compiler-errors): We could replace these with infer, I guess.
            Closure(..)
            | Infer(..)
            | Generator(..)
            | GeneratorWitness(..)
            | Bound(_, _)
            | Placeholder(_)
            | Error(_) => {
                return Err(());
            }

            Alias(Opaque, AliasTy { def_id, .. }) => {
                let parent = self.tcx.parent(def_id);
                let parent_ty = self.tcx.type_of(parent).subst_identity();
                if let hir::def::DefKind::TyAlias | hir::def::DefKind::AssocTy = self.tcx.def_kind(parent)
                    && let Alias(Opaque, AliasTy { def_id: parent_opaque_def_id, .. }) = *parent_ty.kind()
                    && parent_opaque_def_id == def_id
                {
                    t
                } else {
                    return Err(());
                }
            }

            Param(param) => {
                // FIXME: It would be nice to make this not use string manipulation,
                // but it's pretty hard to do this, since `ty::ParamTy` is missing
                // sufficient info to determine if it is synthetic, and we don't
                // always have a convenient way of getting `ty::Generics` at the call
                // sites we invoke `IsSuggestable::is_suggestable`.
                if param.name.as_str().starts_with("impl ") {
                    return Err(());
                }

                t
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

#[derive(Diagnostic)]
#[diag(middle_const_not_used_in_type_alias)]
pub(super) struct ConstNotUsedTraitAlias {
    pub ct: String,
    #[primary_span]
    pub span: Span,
}
