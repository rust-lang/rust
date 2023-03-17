use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned, ToTokens};
use std::collections::HashMap;
use syn::{
    parse::{Error, ParseStream},
    parse_quote,
    spanned::Spanned,
    Attribute, Generics, Ident, LitStr, Token,
};

mod kw {
    syn::custom_keyword!(but_impl_because);
}

fn parse_skip_reason(input: ParseStream<'_>) -> Result<(), Error> {
    input.parse::<kw::but_impl_because>()?;
    input.parse::<Token![=]>()?;
    let reason = input.parse::<LitStr>()?;
    if reason.value().trim().is_empty() {
        Err(Error::new_spanned(
            reason,
            "the value of `but_impl_because` must be a non-empty string",
        ))
    } else {
        Ok(())
    }
}

/// Generate a type parameter with the given `suffix` that does not conflict with
/// any of the `existing` generics.
fn gen_param(suffix: impl ToString, existing: &Generics) -> Ident {
    let mut suffix = suffix.to_string();
    while existing.type_params().any(|t| t.ident == suffix) {
        suffix.insert(0, '_');
    }
    Ident::new(&suffix, Span::call_site())
}

/// Return the interner for the given `structure`.
///
/// If the input represented by `structure` has a `'tcx` lifetime parameter, then we `TyCtxt<'tcx>`
/// will be returned; otherwise our derived implementation will be generic over the interner.
fn gen_interner(structure: &mut synstructure::Structure<'_>) -> TokenStream {
    structure
        .ast()
        .generics
        .lifetimes()
        .find_map(|def| (def.lifetime.ident == "tcx").then_some(&def.lifetime))
        .map(|lt| quote! { ::rustc_middle::ty::TyCtxt<#lt> })
        .unwrap_or_else(|| {
            let ident = gen_param("I", &structure.ast().generics);
            structure.add_impl_generic(parse_quote! { #ident: ::rustc_type_ir::Interner });
            ident.into_token_stream()
        })
}

/// Returns the first `#[skip_traversal]` attribute in `attrs`.
fn find_skip_traversal_attribute(attrs: &[Attribute]) -> Option<&Attribute> {
    attrs.iter().find(|&attr| attr.path.is_ident("skip_traversal"))
}

pub struct Foldable;
pub struct Visitable;

/// An abstraction over traversable traits.
pub trait Traversable {
    /// The trait that this `Traversable` represents, parameterised by `interner`.
    fn traversable(interner: &impl ToTokens) -> TokenStream;

    /// Any supertraits that this trait is required to implement.
    fn supertraits(interner: &impl ToTokens) -> TokenStream;

    /// A (`noop`) traversal of this trait upon the `bind` expression.
    fn traverse(bind: TokenStream, noop: bool) -> TokenStream;

    /// A `match` arm for `variant`, where `f` generates the tokens for each binding.
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        f: impl FnMut(&synstructure::BindingInfo<'_>) -> TokenStream,
    ) -> TokenStream;

    /// The body of an implementation given the `interner`, `traverser` and match expression `body`.
    fn impl_body(
        interner: impl ToTokens,
        traverser: impl ToTokens,
        body: impl ToTokens,
    ) -> TokenStream;
}

impl Traversable for Foldable {
    fn traversable(interner: &impl ToTokens) -> TokenStream {
        quote! { ::rustc_type_ir::fold::TypeFoldable<#interner> }
    }
    fn supertraits(interner: &impl ToTokens) -> TokenStream {
        Visitable::traversable(interner)
    }
    fn traverse(bind: TokenStream, noop: bool) -> TokenStream {
        if noop {
            bind
        } else {
            quote! { ::rustc_type_ir::prefer_noop_traversal_if_applicable!(#bind.try_fold_with(folder))? }
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
        interner: impl ToTokens,
        traverser: impl ToTokens,
        body: impl ToTokens,
    ) -> TokenStream {
        quote! {
            fn try_fold_with<#traverser: ::rustc_type_ir::fold::FallibleTypeFolder<#interner>>(
                self,
                folder: &mut #traverser
            ) -> ::core::result::Result<Self, #traverser::Error> {
                ::core::result::Result::Ok(#body)
            }
        }
    }
}

impl Traversable for Visitable {
    fn traversable(interner: &impl ToTokens) -> TokenStream {
        quote! { ::rustc_type_ir::visit::TypeVisitable<#interner> }
    }
    fn supertraits(_: &impl ToTokens) -> TokenStream {
        quote! { ::core::clone::Clone + ::core::fmt::Debug }
    }
    fn traverse(bind: TokenStream, noop: bool) -> TokenStream {
        if noop {
            quote! {}
        } else {
            quote! { ::rustc_type_ir::prefer_noop_traversal_if_applicable!(#bind.visit_with(visitor))?; }
        }
    }
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        f: impl FnMut(&synstructure::BindingInfo<'_>) -> TokenStream,
    ) -> TokenStream {
        variant.bindings().iter().map(f).collect()
    }
    fn impl_body(
        interner: impl ToTokens,
        traverser: impl ToTokens,
        body: impl ToTokens,
    ) -> TokenStream {
        quote! {
            fn visit_with<#traverser: ::rustc_type_ir::visit::TypeVisitor<#interner>>(
                &self,
                visitor: &mut #traverser
            ) -> ::core::ops::ControlFlow<#traverser::BreakTy> {
                #body
                ::core::ops::ControlFlow::Continue(())
            }
        }
    }
}

pub fn traversable_derive<T: Traversable>(
    mut structure: synstructure::Structure<'_>,
) -> TokenStream {
    let skip_traversal = quote! { ::rustc_type_ir::SkipTraversalAutoImplOnly };

    let interner = gen_interner(&mut structure);
    let traverser = gen_param("T", &structure.ast().generics);
    let traversable = T::traversable(&interner);
    let ast = structure.ast();

    structure.add_bounds(synstructure::AddBounds::None);
    structure.bind_with(|_| synstructure::BindStyle::Move);

    // If our derived implementation will be generic over the traversable type, then we must
    // constrain it to only those generic combinations that satisfy the traversable trait's
    // supertraits.
    let is_generic = !ast.generics.params.is_empty();
    if is_generic {
        let supertraits = T::supertraits(&interner);
        structure.add_where_predicate(parse_quote! { Self: #supertraits });
    }

    let body = if let Some(attr) = find_skip_traversal_attribute(&ast.attrs) {
        if let Err(err) = attr.parse_args_with(parse_skip_reason) {
            return err.into_compile_error();
        }
        // If `is_generic`, it's possible that this no-op impl may not be applicable; but that's fine as it
        // will cause a compilation error forcing removal of the inappropriate `#[skip_traversal]` attribute.
        structure.add_where_predicate(parse_quote! { Self: #skip_traversal });
        T::traverse(quote! { self }, true)
    } else if !is_generic {
        quote_spanned!(ast.ident.span() => {
            ::core::compile_error!("\
                traversal of non-generic types are no-ops by default, so explicitly deriving the traversable traits for them is rarely necessary\n\
                if the need has arisen to due the appearance of this type in an anonymous tuple, consider replacing that tuple with a named struct\n\
                otherwise add `#[skip_traversal(but_impl_because = \"<reason for implementation>\")]` to this type\
            ")
        })
    } else {
        // We add predicates to each generic field type, rather than to our generic type parameters.
        // This results in a "perfect derive" that avoids having to propagate `#[skip_traversal]` annotations
        // into wrapping types, but it can result in trait solver cycles if any type parameters are involved
        // in recursive type definitions; fortunately that is not the case (yet).
        let mut predicates = HashMap::new();
        let arms = structure.each_variant(|variant| {
            let skipped_variant_span = find_skip_traversal_attribute(&variant.ast().attrs).map(Spanned::span);
            if variant.referenced_ty_params().is_empty() {
                if let Some(span) = skipped_variant_span {
                    return quote_spanned!(span => {
                        ::core::compile_error!("non-generic variants are automatically skipped where possible");
                    });
                }
            }
            T::arm(variant, |bind| {
                let ast = bind.ast();
                let skipped_span = skipped_variant_span.or_else(|| find_skip_traversal_attribute(&ast.attrs).map(Spanned::span));
                if bind.referenced_ty_params().is_empty() {
                    if skipped_variant_span.is_none() && let Some(span) = skipped_span {
                        return quote_spanned!(span => {
                            ::core::compile_error!("non-generic fields are automatically skipped where possible");
                        });
                    }
                } else if let Some(prev) = predicates.insert(ast.ty.clone(), skipped_span) && let Some(span) = prev.xor(skipped_span) {
                    // It makes no sense to allow this. Skipping the field requires that its type impls the `SkipTraversalAutoImplOnly`
                    // auto-trait, which means that it does not contain anything interesting to traversers; not also skipping all other
                    // fields of identical type is indicative of a likely erroneous assumption on the author's part--especially since
                    // such other (unannotated) fields will be skipped anyway, as the noop traversal takes precedence.
                    return quote_spanned!(span => {
                        ::core::compile_error!("generic field skipped here, but another field of the same type is not skipped");
                    });
                }
                T::traverse(bind.into_token_stream(), skipped_span.is_some())
            })
        });
        // the order in which `where` predicates appear in rust source is irrelevant
        #[allow(rustc::potential_query_instability)]
        for (ty, skip) in predicates {
            let bound = if skip.is_some() { &skip_traversal } else { &traversable };
            structure.add_where_predicate(parse_quote! { #ty: #bound });
        }
        quote! { match self { #arms } }
    };

    structure.bound_impl(traversable, T::impl_body(interner, traverser, body))
}
