use super::{parse_quote, traversable_derive, visit, Error, Foldable, ToTokens};
use syn::visit_mut::VisitMut;

/// A folder that normalizes syn types for comparison in tests.
struct Normalizer;

/// Generates a folding method for [`Normalizer`] that ensures certain collections
/// are sorted consistently, thus eliminating any non-deterministic output.
macro_rules! normalizing_sort {
        ($($method:ident($field:ident in $ty:ty);)*) => {$(
            fn $method(&mut self, i: &mut $ty) {
                syn::visit_mut::$method(self, i);
                let mut vec = std::mem::take(&mut i.$field).into_iter().collect::<Vec<_>>();
                vec.sort_unstable_by_key(|x| x.to_token_stream().to_string());
                i.$field = vec.into_iter().collect();
            }
        )*};
    }

impl VisitMut for Normalizer {
    // Each of the following fields in the following types can be reordered without
    // affecting semantics, and therefore need to be normalized.
    normalizing_sort! {
        visit_where_clause_mut(predicates in syn::WhereClause);
        visit_predicate_lifetime_mut(bounds in syn::PredicateLifetime);
        visit_bound_lifetimes_mut(lifetimes in syn::BoundLifetimes);
        visit_lifetime_param_mut(bounds in syn::LifetimeParam);
        visit_type_param_mut(bounds in syn::TypeParam);
        visit_type_impl_trait_mut(bounds in syn::TypeImplTrait);
        visit_type_trait_object_mut(bounds in syn::TypeTraitObject);
        visit_generics_mut(params in syn::Generics);
    }

    fn visit_macro_mut(&mut self, i: &mut syn::Macro) {
        syn::visit_mut::visit_macro_mut(self, i);
        if i.path.is_ident("noop_if_trivially_traversable") {
            let mut expr = i
                .parse_body()
                .expect("body of `noop_if_trivially_traversable` macro should be an expression");
            self.visit_expr_mut(&mut expr);
            i.tokens = expr.into_token_stream();
        }
    }

    // For convenience, we also simplify paths by removing absolute crate/module
    // references.
    fn visit_path_mut(&mut self, i: &mut syn::Path) {
        syn::visit_mut::visit_path_mut(self, i);

        let n = if i.leading_colon.is_some() && i.segments.len() >= 2 {
            let segment = &i.segments[0];
            if *segment == parse_quote! { rustc_type_ir } {
                let segment = &i.segments[1];
                if i.segments.len() > 2 && *segment == parse_quote! { fold }
                    || *segment == parse_quote! { visit }
                {
                    2
                } else if *segment == parse_quote! { Interner }
                    || segment.ident == "TriviallyTraverses"
                    || *segment == parse_quote! { noop_if_trivially_traversable }
                {
                    1
                } else {
                    return;
                }
            } else if *segment == parse_quote! { rustc_middle } {
                let segment = &i.segments[1];
                if i.segments.len() > 2 && *segment == parse_quote! { ty } {
                    2
                } else {
                    return;
                }
            } else if i.segments.len() > 2 && *segment == parse_quote! { core } {
                let segment = &i.segments[1];
                if *segment == parse_quote! { ops } {
                    2
                } else if *segment == parse_quote! { result } {
                    i.segments.len() - 1
                } else {
                    return;
                }
            } else {
                return;
            }
        } else {
            return;
        };

        *i = syn::Path {
            leading_colon: None,
            segments: std::mem::take(&mut i.segments).into_iter().skip(n).collect(),
        };
    }
}

#[derive(Default, Debug)]
struct ErrorFinder(Vec<String>);

impl ErrorFinder {
    fn contains(&self, message: &str) -> bool {
        self.0.iter().any(|error| error.starts_with(message))
    }
}

impl visit::Visit<'_> for ErrorFinder {
    fn visit_macro(&mut self, i: &syn::Macro) {
        if i.path == parse_quote! { ::core::compile_error } {
            self.0.push(
                i.parse_body::<syn::LitStr>()
                    .expect("expected compile_error macro to be invoked with a string literal")
                    .value(),
            );
        } else {
            syn::visit::visit_macro(self, i)
        }
    }
}

fn result(input: syn::DeriveInput) -> Result<syn::ItemConst, Error> {
    traversable_derive::<Foldable>(synstructure::Structure::new(&input)).and_then(syn::parse2)
}

fn expect_success(input: syn::DeriveInput, expected: syn::ItemImpl) {
    let result = result(input).expect("expected compiled code to parse");
    let syn::Expr::Block(syn::ExprBlock { block: syn::Block { stmts, .. }, .. }) = *result.expr
    else {
        panic!("expected const expr to be a block")
    };
    assert_eq!(stmts.len(), 1, "expected const expr to contain a single statement");
    let syn::Stmt::Item(syn::Item::Impl(mut actual)) = stmts.into_iter().next().unwrap() else {
        panic!("expected statement in const expr to be an impl")
    };
    Normalizer.visit_item_impl_mut(&mut actual);

    assert!(
        actual == expected,
        "EXPECTED: {}\nACTUAL:   {}",
        expected.to_token_stream(),
        actual.to_token_stream()
    );
}

fn expect_failure(input: syn::DeriveInput, expected: &str) {
    let mut actual = ErrorFinder::default();
    match result(input) {
        Ok(result) => visit::Visit::visit_item_const(&mut actual, &result),
        Err(err) => actual.0.push(err.to_string()),
    }

    assert!(actual.contains(expected), "EXPECTED: {expected}...\nACTUAL:   {actual:?}");
}

macro_rules! expect {
    ({$($input:tt)*} => {$($output:tt)*} $($rest:tt)*) => {
        expect_success(parse_quote! { $($input)* }, parse_quote! { $($output)* });
        expect! { $($rest)* }
    };
    ({$($input:tt)*} => $msg:literal $($rest:tt)*) => {
        expect_failure(parse_quote! { $($input)* }, $msg);
        expect! { $($rest)* }
    };
    () => {};
}

#[test]
fn interesting_fields_are_constrained() {
    expect! {
        {
            struct SomethingInteresting<'a, 'tcx, T>(
                T,
                T::Assoc,
                Const<'tcx>,
                Complex<'tcx, T>,
                Generic<T>,
                Trivial,
                TrivialGeneric<'a, Foo>,
            );
        } => {
            impl<'a, 'tcx, T> TypeFoldable<TyCtxt<'tcx>> for SomethingInteresting<'a, 'tcx, T>
            where
                Complex<'tcx, T>: TypeFoldable<TyCtxt<'tcx>>,
                Generic<T>: TypeFoldable<TyCtxt<'tcx>>,
                Self: TypeVisitable<TyCtxt<'tcx>>,
                T: TypeFoldable<TyCtxt<'tcx>>,
                T::Assoc: TypeFoldable<TyCtxt<'tcx>>
            {
                fn try_fold_with<_T: FallibleTypeFolder<TyCtxt<'tcx>>>(self, folder: &mut _T) -> Result<Self, _T::Error> {
                    Ok(match self {
                        SomethingInteresting (__binding_0, __binding_1, __binding_2, __binding_3, __binding_4, __binding_5, __binding_6,) => { SomethingInteresting(
                            noop_if_trivially_traversable!(__binding_0.try_fold_with::< TyCtxt<'tcx> >(folder))?,
                            noop_if_trivially_traversable!(__binding_1.try_fold_with::< TyCtxt<'tcx> >(folder))?,
                            noop_if_trivially_traversable!(__binding_2.try_fold_with::< TyCtxt<'tcx> >(folder))?,
                            noop_if_trivially_traversable!(__binding_3.try_fold_with::< TyCtxt<'tcx> >(folder))?,
                            noop_if_trivially_traversable!(__binding_4.try_fold_with::< TyCtxt<'tcx> >(folder))?,
                            __binding_5,
                            __binding_6,
                        )}
                    })
                }
            }
        }
    }
}

#[test]
fn skipping_trivial_type_is_superfluous() {
    expect! {
        {
            #[skip_traversal()]
            struct NothingInteresting<'a>;
        } => "trivially traversable types are always skipped, so this attribute is superfluous"
    }
}

#[test]
fn skipping_interesting_type_requires_justification() {
    expect! {
        {
            #[skip_traversal()]
            struct SomethingInteresting<'tcx>;
        } => "Justification must be provided for skipping this potentially interesting type"

        {
            #[skip_traversal(because_trivial)]
            struct SomethingInteresting<'tcx>;
        } => "unsupported skip reason"

        {
            #[skip_traversal(despite_potential_miscompilation_because = ".")]
            struct SomethingInteresting<'tcx>;
        } => {
            impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for SomethingInteresting<'tcx> {
                fn try_fold_with<T: FallibleTypeFolder<TyCtxt<'tcx>>>(self, folder: &mut T) -> Result<Self, T::Error> {
                    Ok(self) // no attempt to fold fields
                }
            }
        }
    }
}

#[test]
fn skipping_interesting_field_requires_justification() {
    expect! {
        {
            struct SomethingInteresting<'tcx>(
                #[skip_traversal()]
                Const<'tcx>,
            );
        } => "Justification must be provided for skipping potentially interesting fields"

        {
            struct SomethingInteresting<'tcx>(
                #[skip_traversal(because_trivial)]
                Const<'tcx>,
            );
        } => "unsupported skip reason"

        {
            struct SomethingInteresting<'tcx>(
                #[skip_traversal(despite_potential_miscompilation_because = ".")]
                Const<'tcx>,
            );
        } => {
            impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for SomethingInteresting<'tcx> {
                fn try_fold_with<T: FallibleTypeFolder<TyCtxt<'tcx>>>(self, folder: &mut T) -> Result<Self, T::Error> {
                    Ok(match self {
                        SomethingInteresting(__binding_0,) => { SomethingInteresting(__binding_0,) } // not folded
                    })
                }
            }
        }
    }
}

#[test]
fn skipping_generic_type_requires_justification() {
    expect! {
        {
            #[skip_traversal()]
            struct SomethingInteresting<T>;
        } => "Justification must be provided for skipping this potentially interesting type"

        {
            #[skip_traversal(despite_potential_miscompilation_because = ".")]
            struct SomethingInteresting<T>;
        } => {
            impl<I: Interner, T> TypeFoldable<I> for SomethingInteresting<T>
            where
                Self: TypeVisitable<I>
            {
                fn try_fold_with<_T: FallibleTypeFolder<I>>(self, folder: &mut _T) -> Result<Self, _T::Error> {
                    Ok(self) // no attempt to fold fields
                }
            }
        }
    }
}

#[test]
fn skipping_generic_field_requires_justification() {
    expect! {
        {
            struct SomethingInteresting<T>(
                #[skip_traversal()]
                T,
            );
        } => "Justification must be provided for skipping potentially interesting fields"

        {
            struct SomethingInteresting<T>(
                #[skip_traversal(because_trivial)]
                T,
            );
        } => {
            impl<I: Interner, T> TypeFoldable<I> for SomethingInteresting<T>
            where
                I: TriviallyTraverses<T>, // `because_trivial`
                Self: TypeVisitable<I>
            {
                fn try_fold_with<_T: FallibleTypeFolder<I>>(self, folder: &mut _T) -> Result<Self, _T::Error> {
                    Ok(match self {
                        SomethingInteresting(__binding_0,) => { SomethingInteresting(__binding_0,) } // not folded
                    })
                }
            }
        }

        {
            struct SomethingInteresting<T>(
                #[skip_traversal(despite_potential_miscompilation_because = ".")]
                T,
            );
        } => {
            impl<I: Interner, T> TypeFoldable<I> for SomethingInteresting<T>
            where
                Self: TypeVisitable<I> // no constraint on T
            {
                fn try_fold_with<_T: FallibleTypeFolder<I>>(self, folder: &mut _T) -> Result<Self, _T::Error> {
                    Ok(match self {
                        SomethingInteresting(__binding_0,) => { SomethingInteresting(__binding_0,) } // not folded
                    })
                }
            }
        }

        {
            struct SomethingInteresting<T>(
                #[skip_traversal(because_trivial)]
                T,
                T,
            );
        } => "This annotation only makes sense if all fields of type `T` are annotated identically"

        {
            struct SomethingInteresting<T>(
                #[skip_traversal(despite_potential_miscompilation_because = ".")]
                T,
                #[skip_traversal(because_trivial)]
                T,
            );
        } => "This annotation only makes sense if all fields of type `T` are annotated identically"
    }
}
