use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use std::collections::HashSet;
use syn::{parse_quote, visit, Field, Generics, Lifetime};

#[cfg(test)]
mod tests;

/// Generate a type parameter with the given `suffix` that does not conflict with
/// any of the `existing` generics.
fn gen_param(suffix: impl ToString, existing: &Generics) -> Ident {
    let mut suffix = suffix.to_string();
    while existing.type_params().any(|t| t.ident == suffix) {
        suffix.insert(0, '_');
    }
    Ident::new(&suffix, Span::call_site())
}

#[derive(Clone, Copy, PartialEq)]
enum Type {
    /// Describes a type that is not parameterised by the interner, and therefore cannot
    /// be of any interest to traversers.
    Trivial,

    /// Describes a type that is parameterised by the interner lifetime `'tcx` but that is
    /// otherwise not generic.
    NotGeneric,

    /// Describes a type that is generic.
    Generic,
}
use Type::*;

pub struct Interner<'a>(Option<&'a Lifetime>);

impl<'a> Interner<'a> {
    /// Return the `TyCtxt` interner for the given `structure`.
    ///
    /// If the input represented by `structure` has a `'tcx` lifetime parameter, then that will be used
    /// used as the lifetime of the `TyCtxt`. Otherwise a `'tcx` lifetime parameter that is unrelated
    /// to the input will be used.
    fn resolve(generics: &'a Generics) -> Self {
        Self(
            generics
                .lifetimes()
                .find_map(|def| (def.lifetime.ident == "tcx").then_some(&def.lifetime)),
        )
    }
}

impl ToTokens for Interner<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let default = parse_quote! { 'tcx };
        let lt = self.0.unwrap_or(&default);
        tokens.extend(quote! { ::rustc_middle::ty::TyCtxt<#lt> });
    }
}

pub struct Foldable;
pub struct Visitable;

/// An abstraction over traversable traits.
pub trait Traversable {
    /// The trait that this `Traversable` represents, parameterised by `interner`.
    fn traversable(interner: &Interner<'_>) -> TokenStream;

    /// Any supertraits that this trait is required to implement.
    fn supertraits(interner: &Interner<'_>) -> TokenStream;

    /// A (`noop`) traversal of this trait upon the `bind` expression.
    fn traverse(bind: TokenStream, noop: bool) -> TokenStream;

    /// A `match` arm for `variant`, where `f` generates the tokens for each binding.
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        f: impl FnMut(&synstructure::BindingInfo<'_>) -> TokenStream,
    ) -> TokenStream;

    /// The body of an implementation given the `interner`, `traverser` and match expression `body`.
    fn impl_body(
        interner: Interner<'_>,
        traverser: impl ToTokens,
        body: impl ToTokens,
    ) -> TokenStream;
}

impl Traversable for Foldable {
    fn traversable(interner: &Interner<'_>) -> TokenStream {
        quote! { ::rustc_middle::ty::fold::TypeFoldable<#interner> }
    }
    fn supertraits(interner: &Interner<'_>) -> TokenStream {
        Visitable::traversable(interner)
    }
    fn traverse(bind: TokenStream, noop: bool) -> TokenStream {
        if noop {
            bind
        } else {
            quote! { ::rustc_middle::ty::fold::TypeFoldable::try_fold_with(#bind, folder)? }
        }
    }
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        mut f: impl FnMut(&synstructure::BindingInfo<'_>) -> TokenStream,
    ) -> TokenStream {
        let bindings = variant.bindings();
        variant.construct(|_, index| f(&bindings[index]))
    }
    fn impl_body(
        interner: Interner<'_>,
        traverser: impl ToTokens,
        body: impl ToTokens,
    ) -> TokenStream {
        quote! {
            fn try_fold_with<#traverser: ::rustc_middle::ty::fold::FallibleTypeFolder<#interner>>(
                self,
                folder: &mut #traverser
            ) -> ::core::result::Result<Self, #traverser::Error> {
                ::core::result::Result::Ok(#body)
            }
        }
    }
}

impl Traversable for Visitable {
    fn traversable(interner: &Interner<'_>) -> TokenStream {
        quote! { ::rustc_middle::ty::visit::TypeVisitable<#interner> }
    }
    fn supertraits(_: &Interner<'_>) -> TokenStream {
        quote! { ::core::clone::Clone + ::core::fmt::Debug }
    }
    fn traverse(bind: TokenStream, noop: bool) -> TokenStream {
        if noop {
            quote! {}
        } else {
            quote! { ::rustc_middle::ty::visit::TypeVisitable::visit_with(#bind, visitor)?; }
        }
    }
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        f: impl FnMut(&synstructure::BindingInfo<'_>) -> TokenStream,
    ) -> TokenStream {
        variant.bindings().iter().map(f).collect()
    }
    fn impl_body(
        interner: Interner<'_>,
        traverser: impl ToTokens,
        body: impl ToTokens,
    ) -> TokenStream {
        quote! {
            fn visit_with<#traverser: ::rustc_middle::ty::visit::TypeVisitor<#interner>>(
                &self,
                visitor: &mut #traverser
            ) -> ::core::ops::ControlFlow<#traverser::BreakTy> {
                #body
                ::core::ops::ControlFlow::Continue(())
            }
        }
    }
}

impl Interner<'_> {
    /// We consider a type to be internable if it references either a generic type parameter or,
    /// if the interner is `TyCtxt<'tcx>`, the `'tcx` lifetime.
    fn type_of<'a>(
        &self,
        referenced_ty_params: &[&Ident],
        fields: impl IntoIterator<Item = &'a Field>,
    ) -> Type {
        use visit::Visit;

        struct Info<'a> {
            interner: &'a Lifetime,
            contains_interner: bool,
        }

        impl Visit<'_> for Info<'_> {
            fn visit_lifetime(&mut self, i: &Lifetime) {
                if i == self.interner {
                    self.contains_interner = true;
                } else {
                    visit::visit_lifetime(self, i)
                }
            }
        }

        if !referenced_ty_params.is_empty() {
            Generic
        } else if let Some(interner) = &self.0 && fields.into_iter().any(|field| {
            let mut info = Info { interner, contains_interner: false };
            info.visit_type(&field.ty);
            info.contains_interner
        }) {
            NotGeneric
        } else {
            Trivial
        }
    }
}

pub fn traversable_derive<T: Traversable>(
    mut structure: synstructure::Structure<'_>,
) -> TokenStream {
    let ast = structure.ast();

    let interner = Interner::resolve(&ast.generics);
    let traverser = gen_param("T", &ast.generics);
    let traversable = T::traversable(&interner);

    structure.underscore_const(true);
    structure.add_bounds(synstructure::AddBounds::None);
    structure.bind_with(|_| synstructure::BindStyle::Move);

    if interner.0.is_none() {
        structure.add_impl_generic(parse_quote! { 'tcx });
    }

    // If our derived implementation will be generic over the traversable type, then we must
    // constrain it to only those generic combinations that satisfy the traversable trait's
    // supertraits.
    if ast.generics.type_params().next().is_some() {
        let supertraits = T::supertraits(&interner);
        structure.add_where_predicate(parse_quote! { Self: #supertraits });
    }

    // We add predicates to each generic field type, rather than to our generic type parameters.
    // This results in a "perfect derive", but it can result in trait solver cycles if any type
    // parameters are involved in recursive type definitions; fortunately that is not the case (yet).
    let mut predicates = HashSet::new();
    let arms = structure.each_variant(|variant| {
        T::arm(variant, |bind| {
            let ast = bind.ast();
            let field_ty = interner.type_of(&bind.referenced_ty_params(), [ast]);
            if field_ty == Generic {
                predicates.insert(ast.ty.clone());
            }
            T::traverse(bind.into_token_stream(), field_ty == Trivial)
        })
    });
    // the order in which `where` predicates appear in rust source is irrelevant
    #[allow(rustc::potential_query_instability)]
    for ty in predicates {
        structure.add_where_predicate(parse_quote! { #ty: #traversable });
    }
    let body = quote! { match self { #arms } };

    structure.bound_impl(traversable, T::impl_body(interner, traverser, body))
}
