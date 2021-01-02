//! Diagnostics related methods for `TyS`.

use crate::ty::sty::InferTy;
use crate::ty::TyKind::*;
use crate::ty::{TyCtxt, TyS};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{QPath, TyKind, WhereBoundPredicate, WherePredicate};

impl<'tcx> TyS<'tcx> {
    /// Similar to `TyS::is_primitive`, but also considers inferred numeric values to be primitive.
    pub fn is_primitive_ty(&self) -> bool {
        matches!(
            self.kind(),
            Bool | Char | Str | Int(_) | Uint(_) | Float(_)
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
    pub fn is_simple_ty(&self) -> bool {
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
    pub fn is_simple_text(&self) -> bool {
        match self.kind() {
            Adt(_, substs) => substs.types().next().is_none(),
            Ref(_, ty, _) => ty.is_simple_text(),
            _ => self.is_simple_ty(),
        }
    }

    /// Whether the type can be safely suggested during error recovery.
    pub fn is_suggestable(&self) -> bool {
        !matches!(
            self.kind(),
            Opaque(..)
                | FnDef(..)
                | FnPtr(..)
                | Dynamic(..)
                | Closure(..)
                | Infer(..)
                | Projection(..)
        )
    }
}

/// Suggest restricting a type param with a new bound.
pub fn suggest_constraining_type_param(
    tcx: TyCtxt<'_>,
    generics: &hir::Generics<'_>,
    err: &mut DiagnosticBuilder<'_>,
    param_name: &str,
    constraint: &str,
    def_id: Option<DefId>,
) -> bool {
    let param = generics.params.iter().find(|p| p.name.ident().as_str() == param_name);

    let param = if let Some(param) = param {
        param
    } else {
        return false;
    };

    const MSG_RESTRICT_BOUND_FURTHER: &str = "consider further restricting this bound";
    let msg_restrict_type = format!("consider restricting type parameter `{}`", param_name);
    let msg_restrict_type_further =
        format!("consider further restricting type parameter `{}`", param_name);

    if def_id == tcx.lang_items().sized_trait() {
        // Type parameters are already `Sized` by default.
        err.span_label(param.span, &format!("this type parameter needs to be `{}`", constraint));
        return true;
    }
    let mut suggest_restrict = |span| {
        err.span_suggestion_verbose(
            span,
            MSG_RESTRICT_BOUND_FURTHER,
            format!(" + {}", constraint),
            Applicability::MachineApplicable,
        );
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

        suggest_restrict(param.span.shrink_to_hi());
        return true;
    }

    if generics.where_clause.predicates.is_empty()
        // Given `trait Base<T = String>: Super<T>` where `T: Copy`, suggest restricting in the
        // `where` clause instead of `trait Base<T: Copy = String>: Super<T>`.
        && !matches!(param.kind, hir::GenericParamKind::Type { default: Some(_), .. })
    {
        if let Some(bounds_span) = param.bounds_span() {
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
            suggest_restrict(bounds_span.shrink_to_hi());
        } else {
            // If user hasn't provided any bounds, suggest adding a new one:
            //
            //   fn foo<T>(t: T) { ... }
            //          - help: consider restricting this type parameter with `T: Foo`
            err.span_suggestion_verbose(
                param.span.shrink_to_hi(),
                &msg_restrict_type,
                format!(": {}", constraint),
                Applicability::MachineApplicable,
            );
        }

        true
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
            && generics.where_clause.predicates.len() == 0
        {
            // Suggest a bound, but there is no existing `where` clause *and* the type param has a
            // default (`<T=Foo>`), so we suggest adding `where T: Bar`.
            err.span_suggestion_verbose(
                generics.where_clause.tail_span_for_suggestion(),
                &msg_restrict_type_further,
                format!(" where {}: {}", param_name, constraint),
                Applicability::MachineApplicable,
            );
        } else {
            let mut param_spans = Vec::new();

            for predicate in generics.where_clause.predicates {
                if let WherePredicate::BoundPredicate(WhereBoundPredicate {
                    span,
                    bounded_ty,
                    ..
                }) = predicate
                {
                    if let TyKind::Path(QPath::Resolved(_, path)) = &bounded_ty.kind {
                        if let Some(segment) = path.segments.first() {
                            if segment.ident.to_string() == param_name {
                                param_spans.push(span);
                            }
                        }
                    }
                }
            }

            match param_spans[..] {
                [&param_span] => suggest_restrict(param_span.shrink_to_hi()),
                _ => {
                    err.span_suggestion_verbose(
                        generics.where_clause.tail_span_for_suggestion(),
                        &msg_restrict_type_further,
                        format!(", {}: {}", param_name, constraint),
                        Applicability::MachineApplicable,
                    );
                }
            }
        }

        true
    }
}

/// Collect al types that have an implicit `'static` obligation that we could suggest `'_` for.
pub struct TraitObjectVisitor<'tcx>(pub Vec<&'tcx hir::Ty<'tcx>>, pub crate::hir::map::Map<'tcx>);

impl<'v> hir::intravisit::Visitor<'v> for TraitObjectVisitor<'v> {
    type Map = rustc_hir::intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
        match ty.kind {
            hir::TyKind::TraitObject(
                _,
                hir::Lifetime {
                    name:
                        hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Static,
                    ..
                },
            ) => {
                self.0.push(ty);
            }
            hir::TyKind::OpaqueDef(item_id, _) => {
                self.0.push(ty);
                let item = self.1.expect_item(item_id.id);
                hir::intravisit::walk_item(self, item);
            }
            _ => {}
        }
        hir::intravisit::walk_ty(self, ty);
    }
}
