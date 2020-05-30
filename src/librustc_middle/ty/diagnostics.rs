//! Diagnostics related methods for `TyS`.

use crate::ty::sty::InferTy;
use crate::ty::TyKind::*;
use crate::ty::{TyCtxt, TyS};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{QPath, TyKind, WhereBoundPredicate, WherePredicate};
use rustc_span::{BytePos, Span};

impl<'tcx> TyS<'tcx> {
    /// Similar to `TyS::is_primitive`, but also considers inferred numeric values to be primitive.
    pub fn is_primitive_ty(&self) -> bool {
        match self.kind {
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
            _ => false,
        }
    }

    /// Whether the type is succinctly representable as a type instead of just referred to with a
    /// description in error messages. This is used in the main error message.
    pub fn is_simple_ty(&self) -> bool {
        match self.kind {
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
        match self.kind {
            Adt(_, substs) => substs.types().next().is_none(),
            Ref(_, ty, _) => ty.is_simple_text(),
            _ => self.is_simple_ty(),
        }
    }

    /// Whether the type can be safely suggested during error recovery.
    pub fn is_suggestable(&self) -> bool {
        match self.kind {
            Opaque(..) | FnDef(..) | FnPtr(..) | Dynamic(..) | Closure(..) | Infer(..)
            | Projection(..) => false,
            _ => true,
        }
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

        let mut param_spans = Vec::new();

        for predicate in generics.where_clause.predicates {
            if let WherePredicate::BoundPredicate(WhereBoundPredicate {
                span, bounded_ty, ..
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

        let where_clause_span = generics.where_clause.span_for_predicates_or_empty_place();
        // Account for `fn foo<T>(t: T) where T: Foo,` so we don't suggest two trailing commas.
        let mut trailing_comma = false;
        if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(where_clause_span) {
            trailing_comma = snippet.ends_with(',');
        }
        let where_clause_span = if trailing_comma {
            let hi = where_clause_span.hi();
            Span::new(hi - BytePos(1), hi, where_clause_span.ctxt())
        } else {
            where_clause_span.shrink_to_hi()
        };

        match &param_spans[..] {
            &[&param_span] => suggest_restrict(param_span.shrink_to_hi()),
            _ => {
                err.span_suggestion_verbose(
                    where_clause_span,
                    &msg_restrict_type_further,
                    format!(", {}: {}", param_name, constraint),
                    Applicability::MachineApplicable,
                );
            }
        }

        true
    }
}

pub struct TraitObjectVisitor(pub Vec<rustc_span::Span>);
impl<'v> hir::intravisit::Visitor<'v> for TraitObjectVisitor {
    type Map = rustc_hir::intravisit::ErasedMap<'v>;

    fn nested_visit_map(&mut self) -> hir::intravisit::NestedVisitorMap<Self::Map> {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_ty(&mut self, ty: &hir::Ty<'_>) {
        if let hir::TyKind::TraitObject(
            _,
            hir::Lifetime { name: hir::LifetimeName::ImplicitObjectLifetimeDefault, .. },
        ) = ty.kind
        {
            self.0.push(ty.span);
        }
    }
}
