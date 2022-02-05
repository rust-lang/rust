//! Diagnostics related methods for `Ty`.

use crate::ty::subst::{GenericArg, GenericArgKind};
use crate::ty::TyKind::*;
use crate::ty::{
    ConstKind, DefIdTree, ExistentialPredicate, ExistentialProjection, ExistentialTraitRef,
    InferTy, ProjectionTy, Term, Ty, TyCtxt, TypeAndMut,
};

use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diagnostic, DiagnosticArgValue, IntoDiagnosticArg};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{QPath, TyKind, WhereBoundPredicate, WherePredicate};
use rustc_span::Span;

impl<'tcx> IntoDiagnosticArg for Ty<'tcx> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        format!("{}", self).into_diagnostic_arg()
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

    /// Whether the type can be safely suggested during error recovery.
    pub fn is_suggestable(self, tcx: TyCtxt<'tcx>) -> bool {
        fn generic_arg_is_suggestible<'tcx>(arg: GenericArg<'tcx>, tcx: TyCtxt<'tcx>) -> bool {
            match arg.unpack() {
                GenericArgKind::Type(ty) => ty.is_suggestable(tcx),
                GenericArgKind::Const(c) => const_is_suggestable(c.val()),
                _ => true,
            }
        }

        fn const_is_suggestable(kind: ConstKind<'_>) -> bool {
            match kind {
                ConstKind::Infer(..)
                | ConstKind::Bound(..)
                | ConstKind::Placeholder(..)
                | ConstKind::Error(..) => false,
                _ => true,
            }
        }

        // FIXME(compiler-errors): Some types are still not good to suggest,
        // specifically references with lifetimes within the function. Not
        //sure we have enough information to resolve whether a region is
        // temporary, so I'll leave this as a fixme.

        match self.kind() {
            FnDef(..)
            | Closure(..)
            | Infer(..)
            | Generator(..)
            | GeneratorWitness(..)
            | Bound(_, _)
            | Placeholder(_)
            | Error(_) => false,
            Opaque(did, substs) => {
                let parent = tcx.parent(*did).expect("opaque types always have a parent");
                if let hir::def::DefKind::TyAlias | hir::def::DefKind::AssocTy = tcx.def_kind(parent)
                    && let Opaque(parent_did, _) = tcx.type_of(parent).kind()
                    && parent_did == did
                {
                    substs.iter().all(|a| generic_arg_is_suggestible(a, tcx))
                } else {
                    false
                }
            }
            Dynamic(dty, _) => dty.iter().all(|pred| match pred.skip_binder() {
                ExistentialPredicate::Trait(ExistentialTraitRef { substs, .. }) => {
                    substs.iter().all(|a| generic_arg_is_suggestible(a, tcx))
                }
                ExistentialPredicate::Projection(ExistentialProjection {
                    substs, term, ..
                }) => {
                    let term_is_suggestable = match term {
                        Term::Ty(ty) => ty.is_suggestable(tcx),
                        Term::Const(c) => const_is_suggestable(c.val()),
                    };
                    term_is_suggestable && substs.iter().all(|a| generic_arg_is_suggestible(a, tcx))
                }
                _ => true,
            }),
            Projection(ProjectionTy { substs: args, .. }) | Adt(_, args) => {
                args.iter().all(|a| generic_arg_is_suggestible(a, tcx))
            }
            Tuple(args) => args.iter().all(|ty| ty.is_suggestable(tcx)),
            Slice(ty) | RawPtr(TypeAndMut { ty, .. }) | Ref(_, ty, _) => ty.is_suggestable(tcx),
            Array(ty, c) => ty.is_suggestable(tcx) && const_is_suggestable(c.val()),
            _ => true,
        }
    }
}

pub fn suggest_arbitrary_trait_bound(
    generics: &hir::Generics<'_>,
    err: &mut Diagnostic,
    param_name: &str,
    constraint: &str,
) -> bool {
    let param = generics.params.iter().find(|p| p.name.ident().as_str() == param_name);
    match (param, param_name) {
        (Some(_), "Self") => return false,
        _ => {}
    }
    // Suggest a where clause bound for a non-type parameter.
    let (action, prefix) = if generics.predicates.is_empty() {
        ("introducing a", " where ")
    } else {
        ("extending the", ", ")
    };
    err.span_suggestion_verbose(
        generics.tail_span_for_predicate_suggestion(),
        &format!(
            "consider {} `where` bound, but there might be an alternative better way to express \
             this requirement",
            action,
        ),
        format!("{}{}: {}", prefix, param_name, constraint),
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
    param_name: &str,
    param: &hir::GenericParam<'_>,
    def_id: Option<DefId>,
) {
    // See if there's a `?Sized` bound that can be removed to suggest that.
    // First look at the `where` clause because we can have `where T: ?Sized`,
    // then look at params.
    for (where_pos, predicate) in generics.predicates.iter().enumerate() {
        match predicate {
            WherePredicate::BoundPredicate(WhereBoundPredicate {
                bounded_ty:
                    hir::Ty {
                        kind:
                            hir::TyKind::Path(hir::QPath::Resolved(
                                None,
                                hir::Path {
                                    segments: [segment],
                                    res: hir::def::Res::Def(hir::def::DefKind::TyParam, _),
                                    ..
                                },
                            )),
                        ..
                    },
                bounds,
                span,
                ..
            }) if segment.ident.as_str() == param_name => {
                for (pos, bound) in bounds.iter().enumerate() {
                    match bound {
                        hir::GenericBound::Trait(poly, hir::TraitBoundModifier::Maybe)
                            if poly.trait_ref.trait_def_id() == def_id => {}
                        _ => continue,
                    }
                    let sp = match (bounds.len(), pos, generics.predicates.len(), where_pos) {
                        // where T: ?Sized
                        // ^^^^^^^^^^^^^^^
                        (1, _, 1, _) => generics.where_clause_span,
                        // where Foo: Bar, T: ?Sized,
                        //               ^^^^^^^^^^^
                        (1, _, len, pos) if pos == len - 1 => {
                            generics.predicates[pos - 1].span().shrink_to_hi().to(*span)
                        }
                        // where T: ?Sized, Foo: Bar,
                        //       ^^^^^^^^^^^
                        (1, _, _, pos) => span.until(generics.predicates[pos + 1].span()),
                        // where T: ?Sized + Bar, Foo: Bar,
                        //          ^^^^^^^^^
                        (_, 0, _, _) => bound.span().to(bounds[1].span().shrink_to_lo()),
                        // where T: Bar + ?Sized, Foo: Bar,
                        //             ^^^^^^^^^
                        (_, pos, _, _) => bounds[pos - 1].span().shrink_to_hi().to(bound.span()),
                    };

                    suggestions.push((
                        sp,
                        String::new(),
                        SuggestChangingConstraintsMessage::RemovingQSized,
                    ));
                }
            }
            _ => {}
        }
    }
    for (pos, bound) in param.bounds.iter().enumerate() {
        match bound {
            hir::GenericBound::Trait(poly, hir::TraitBoundModifier::Maybe)
                if poly.trait_ref.trait_def_id() == def_id =>
            {
                let sp = match (param.bounds.len(), pos) {
                    // T: ?Sized,
                    //  ^^^^^^^^
                    (1, _) => param.span.shrink_to_hi().to(bound.span()),
                    // T: ?Sized + Bar,
                    //    ^^^^^^^^^
                    (_, 0) => bound.span().to(param.bounds[1].span().shrink_to_lo()),
                    // T: Bar + ?Sized,
                    //       ^^^^^^^^^
                    (_, pos) => param.bounds[pos - 1].span().shrink_to_hi().to(bound.span()),
                };

                suggestions.push((
                    sp,
                    String::new(),
                    SuggestChangingConstraintsMessage::RemovingQSized,
                ));
            }
            _ => {}
        }
    }
}

/// Suggest restricting a type param with a new bound.
pub fn suggest_constraining_type_param(
    tcx: TyCtxt<'_>,
    generics: &hir::Generics<'_>,
    err: &mut Diagnostic,
    param_name: &str,
    constraint: &str,
    def_id: Option<DefId>,
) -> bool {
    suggest_constraining_type_params(
        tcx,
        generics,
        err,
        [(param_name, constraint, def_id)].into_iter(),
    )
}

/// Suggest restricting a type param with a new bound.
pub fn suggest_constraining_type_params<'a>(
    tcx: TyCtxt<'_>,
    generics: &hir::Generics<'_>,
    err: &mut Diagnostic,
    param_names_and_constraints: impl Iterator<Item = (&'a str, &'a str, Option<DefId>)>,
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
                suggest_removing_unsized_bound(
                    generics,
                    &mut suggestions,
                    param_name,
                    param,
                    def_id,
                );
            }
        }

        if constraints.is_empty() {
            continue;
        }

        let constraint = constraints.iter().map(|&(c, _)| c).collect::<Vec<_>>().join(" + ");
        let mut suggest_restrict = |span, bound_list_non_empty| {
            suggestions.push((
                span,
                if bound_list_non_empty {
                    format!(" + {}", constraint)
                } else {
                    format!(" {}", constraint)
                },
                SuggestChangingConstraintsMessage::RestrictBoundFurther,
            ))
        };

        if param_name.starts_with("impl ") {
            // If there's an `impl Trait` used in argument position, suggest
            // restricting it:
            //
            //   fn foo(t: impl Foo) { ... }
            //             --------
            //             |
            //             help: consider further restricting this bound with `+ Bar`
            //
            // Suggestion for tools in this case is:
            //
            //   fn foo(t: impl Foo) { ... }
            //             --------
            //             |
            //             replace with: `impl Foo + Bar`

            // `impl Trait` must have at least one trait in the list
            let bound_list_non_empty = true;

            suggest_restrict(param.span.shrink_to_hi(), bound_list_non_empty);
            continue;
        }

        if generics.predicates.is_empty()
        // Given `trait Base<T = String>: Super<T>` where `T: Copy`, suggest restricting in the
        // `where` clause instead of `trait Base<T: Copy = String>: Super<T>`.
        && !matches!(param.kind, hir::GenericParamKind::Type { default: Some(_), .. })
        {
            if let Some(span) = param.bounds_span_for_suggestions() {
                // If user has provided some bounds, suggest restricting them:
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

                // `bounds_span_for_suggestions` returns `None` if the list is empty
                let bound_list_non_empty = true;

                suggest_restrict(span, bound_list_non_empty);
            } else {
                let (colon, span) = match param.colon_span_for_suggestions(tcx.sess.source_map()) {
                    // If there is already a colon after generic, do not suggest adding it again
                    Some(sp) => ("", sp.shrink_to_hi()),
                    None => (":", param.span.shrink_to_hi()),
                };

                // If user hasn't provided any bounds, suggest adding a new one:
                //
                //   fn foo<T>(t: T) { ... }
                //          - help: consider restricting this type parameter with `T: Foo`
                suggestions.push((
                    span,
                    format!("{colon} {constraint}"),
                    SuggestChangingConstraintsMessage::RestrictType { ty: param_name },
                ));
            }
        } else {
            // This part is a bit tricky, because using the `where` clause user can
            // provide zero, one or many bounds for the same type parameter, so we
            // have following cases to consider:
            //
            // 1) When the type parameter has been provided zero bounds
            //
            //    Message:
            //      fn foo<X, Y>(x: X, y: Y) where Y: Foo { ... }
            //             - help: consider restricting this type parameter with `where X: Bar`
            //
            //    Suggestion:
            //      fn foo<X, Y>(x: X, y: Y) where Y: Foo { ... }
            //                                           - insert: `, X: Bar`
            //
            //
            // 2) When the type parameter has been provided one bound
            //
            //    Message:
            //      fn foo<T>(t: T) where T: Foo { ... }
            //                            ^^^^^^
            //                            |
            //                            help: consider further restricting this bound with `+ Bar`
            //
            //    Suggestion:
            //      fn foo<T>(t: T) where T: Foo { ... }
            //                            ^^
            //                            |
            //                            replace with: `T: Bar +`
            //
            //
            // 3) When the type parameter has been provided many bounds
            //
            //    Message:
            //      fn foo<T>(t: T) where T: Foo, T: Bar {... }
            //             - help: consider further restricting this type parameter with `where T: Zar`
            //
            //    Suggestion:
            //      fn foo<T>(t: T) where T: Foo, T: Bar {... }
            //                                          - insert: `, T: Zar`
            //
            // Additionally, there may be no `where` clause whatsoever in the case that this was
            // reached because the generic parameter has a default:
            //
            //    Message:
            //      trait Foo<T=()> {... }
            //             - help: consider further restricting this type parameter with `where T: Zar`
            //
            //    Suggestion:
            //      trait Foo<T=()> where T: Zar {... }
            //                     - insert: `where T: Zar`

            if matches!(param.kind, hir::GenericParamKind::Type { default: Some(_), .. })
                && generics.predicates.len() == 0
            {
                // Suggest a bound, but there is no existing `where` clause *and* the type param has a
                // default (`<T=Foo>`), so we suggest adding `where T: Bar`.
                suggestions.push((
                    generics.tail_span_for_predicate_suggestion(),
                    format!(" where {}: {}", param_name, constraint),
                    SuggestChangingConstraintsMessage::RestrictTypeFurther { ty: param_name },
                ));
            } else {
                let mut param_spans = Vec::new();
                let mut non_empty = false;

                for predicate in generics.predicates {
                    if let WherePredicate::BoundPredicate(WhereBoundPredicate {
                        span,
                        bounded_ty,
                        bounds,
                        ..
                    }) = predicate
                    {
                        if let TyKind::Path(QPath::Resolved(_, path)) = &bounded_ty.kind {
                            if let Some(segment) = path.segments.first() {
                                if segment.ident.to_string() == param_name {
                                    non_empty = !bounds.is_empty();

                                    param_spans.push(span);
                                }
                            }
                        }
                    }
                }

                match param_spans[..] {
                    [&param_span] => suggest_restrict(param_span.shrink_to_hi(), non_empty),
                    _ => {
                        suggestions.push((
                            generics.tail_span_for_predicate_suggestion(),
                            constraints
                                .iter()
                                .map(|&(constraint, _)| format!(", {}: {}", param_name, constraint))
                                .collect::<String>(),
                            SuggestChangingConstraintsMessage::RestrictTypeFurther {
                                ty: param_name,
                            },
                        ));
                    }
                }
            }
        }
    }

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
                    name:
                        hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Static,
                    ..
                },
                _,
            ) => {
                self.0.push(ty);
            }
            hir::TyKind::OpaqueDef(item_id, _) => {
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
        if let hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Static =
            lt.name
        {
            self.0.push(lt.span);
        }
    }
}
