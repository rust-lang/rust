use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use std::collections::{hash_map::Entry, HashMap};
use syn::{
    meta::ParseNestedMeta, parse::Error, parse_quote, visit, Attribute, Field, Generics, Lifetime,
    LitStr, Token,
};

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

#[derive(Clone, Copy)]
enum WhenToSkip {
    /// No skip_traversal annotation requires the annotated item to be skipped
    Never,

    /// A skip_traversal annotation requires the annotated item to be skipped, with its type
    /// constrained to TriviallyTraversable
    Always(Span),

    /// A `despite_potential_miscompilation_because` annotation is present, thus requiring the
    /// annotated item to be forcibly skipped without its type being constrained to
    /// TriviallyTraversable
    Forced,
}

impl Default for WhenToSkip {
    fn default() -> Self {
        Self::Never
    }
}

impl WhenToSkip {
    fn is_skipped(&self) -> bool {
        !matches!(self, WhenToSkip::Never)
    }
}

impl std::ops::BitOrAssign for WhenToSkip {
    fn bitor_assign(&mut self, rhs: Self) {
        match self {
            Self::Forced => (),
            Self::Always(_) => {
                if matches!(rhs, Self::Forced) {
                    *self = Self::Forced;
                }
            }
            Self::Never => *self = rhs,
        }
    }
}

impl WhenToSkip {
    fn find<const IS_TYPE: bool>(&mut self, attrs: &[Attribute], ty: Type) -> Result<(), Error> {
        fn parse_reason(meta: &ParseNestedMeta<'_>) -> Result<(), Error> {
            if !meta.value()?.parse::<LitStr>()?.value().trim().is_empty() {
                Ok(())
            } else {
                Err(meta.error("skip reason must be a non-empty string"))
            }
        }

        let mut found = None;
        for attr in attrs {
            if attr.path().is_ident("skip_traversal") {
                found = Some(attr);
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("despite_potential_miscompilation_because") {
                        parse_reason(&meta)?;
                        *self |= Self::Forced;
                        return Ok(());
                    }

                    if !IS_TYPE && ty == Generic && meta.path.is_ident("because_trivial") {
                        *self |= Self::Always(meta.error("").span());
                        return Ok(());
                    }

                    Err(meta.error("unsupported skip reason"))
                })?;
            }
        }

        if let (Self::Never, Some(attr)) = (self, found) {
            Err(Error::new_spanned(
                attr,
                if IS_TYPE {
                    match ty {
                        Trivial => {
                            "trivially traversable types are always skipped, so this attribute is superfluous"
                        }
                        _ => {
                            "\
                            Justification must be provided for skipping this potentially interesting type, by specifying\n\
                            `despite_potential_miscompilation_because = \"<reason>\"`\
                        "
                        }
                    }
                } else {
                    match ty {
                        Trivial => {
                            "trivially traversable fields are always skipped, so this attribute is superfluous"
                        }
                        _ => {
                            "\
                            Justification must be provided for skipping potentially interesting fields, by specifying EITHER:\n\
                            `because_trivial` if concrete instances do not actually contain anything of interest (enforced by the compiler); OR\n\
                            `despite_potential_miscompilation_because = \"<reason>\"` in the rare case that a field should always be skipped regardless\
                        "
                        }
                    }
                },
            ))
        } else {
            Ok(())
        }
    }
}

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
    fn traverse(bind: TokenStream, noop: bool, interner: &Interner<'_>) -> TokenStream;

    /// A `match` arm for `variant`, where `f` generates the tokens for each binding.
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        f: impl FnMut(&synstructure::BindingInfo<'_>) -> Result<TokenStream, Error>,
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
    fn traverse(bind: TokenStream, noop: bool, interner: &Interner<'_>) -> TokenStream {
        if noop {
            bind
        } else {
            quote! { ::rustc_middle::ty::noop_if_trivially_traversable!(#bind.try_fold_with::<#interner>(folder))? }
        }
    }
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        mut f: impl FnMut(&synstructure::BindingInfo<'_>) -> Result<TokenStream, Error>,
    ) -> TokenStream {
        let bindings = variant.bindings();
        variant.construct(|_, index| f(&bindings[index]).unwrap_or_else(Error::into_compile_error))
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
    fn traverse(bind: TokenStream, noop: bool, interner: &Interner<'_>) -> TokenStream {
        if noop {
            quote! {}
        } else {
            quote! { ::rustc_middle::ty::noop_if_trivially_traversable!(#bind.visit_with::<#interner>(visitor))?; }
        }
    }
    fn arm(
        variant: &synstructure::VariantInfo<'_>,
        f: impl FnMut(&synstructure::BindingInfo<'_>) -> Result<TokenStream, Error>,
    ) -> TokenStream {
        variant
            .bindings()
            .iter()
            .map(f)
            .collect::<Result<_, _>>()
            .unwrap_or_else(Error::into_compile_error)
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
) -> Result<TokenStream, Error> {
    use WhenToSkip::*;

    let ast = structure.ast();

    let interner = Interner::resolve(&ast.generics);
    let traverser = gen_param("T", &ast.generics);
    let traversable = T::traversable(&interner);

    let skip_traversal =
        |t: &dyn ToTokens| parse_quote! { #interner: ::rustc_middle::ty::TriviallyTraverses<#t> };

    structure.underscore_const(true);
    structure.add_bounds(synstructure::AddBounds::None);
    structure.bind_with(|_| synstructure::BindStyle::Move);

    let not_generic = if interner.0.is_none() {
        structure.add_impl_generic(parse_quote! { 'tcx });
        Trivial
    } else {
        NotGeneric
    };

    // If our derived implementation will be generic over the traversable type, then we must
    // constrain it to only those generic combinations that satisfy the traversable trait's
    // supertraits.
    let ty = if ast.generics.type_params().next().is_some() {
        let supertraits = T::supertraits(&interner);
        structure.add_where_predicate(parse_quote! { Self: #supertraits });
        Generic
    } else {
        not_generic
    };

    let mut when_to_skip = WhenToSkip::default();
    when_to_skip.find::<true>(&ast.attrs, ty)?;
    let body = if when_to_skip.is_skipped() {
        if let Always(_) = when_to_skip {
            structure.add_where_predicate(skip_traversal(&<Token![Self]>::default()));
        }
        T::traverse(quote! { self }, true, &interner)
    } else {
        // We add predicates to each generic field type, rather than to our generic type parameters.
        // This results in a "perfect derive" that avoids having to propagate `#[skip_traversal]` annotations
        // into wrapping types, but it can result in trait solver cycles if any type parameters are involved
        // in recursive type definitions; fortunately that is not the case (yet).
        let mut predicates = HashMap::<_, (_, _)>::new();

        let arms = structure.each_variant(|variant| {
            let variant_ty = interner.type_of(&variant.referenced_ty_params(), variant.ast().fields);
            let mut skipped_variant = WhenToSkip::default();
            if let Err(err) = skipped_variant.find::<false>(variant.ast().attrs, variant_ty) {
                return err.into_compile_error();
            }
            T::arm(variant, |bind| {
                let ast = bind.ast();
                let is_skipped = variant_ty == Trivial || {
                    let field_ty = interner.type_of(&bind.referenced_ty_params(), [ast]);
                    field_ty == Trivial || {
                        let mut skipped_field = skipped_variant;
                        skipped_field.find::<false>(&ast.attrs, field_ty)?;

                        match predicates.entry(ast.ty.clone()) {
                            Entry::Occupied(existing) => match (&mut existing.into_mut().0, skipped_field) {
                                (Never, Never) | (Never, Forced) | (Forced, Forced) | (Always(_), Always(_)) => (),
                                (existing @ Forced, Never) => *existing = Never,
                                (&mut Always(span), _) | (_, Always(span)) => return Err(Error::new(span, format!("\
                                    This annotation only makes sense if all fields of type `{0}` are annotated identically.\n\
                                    In particular, the derived impl will only be applicable when `{0}: TriviallyTraversable` and therefore all traversals of `{0}` will be no-ops;\n\
                                    accordingly it makes no sense for other fields of type `{0}` to omit `#[skip_traversal]` or to include `despite_potential_miscompilation_because`.\
                                ", ast.ty.to_token_stream()))),
                            },
                            Entry::Vacant(entry) => { entry.insert((skipped_field, field_ty)); }
                        }

                        skipped_field.is_skipped()
                    }
                };

                Ok(T::traverse(bind.into_token_stream(), is_skipped, &interner))
            })
        });

        // the order in which `where` predicates appear in rust source is irrelevant
        #[allow(rustc::potential_query_instability)]
        for (ty, (when_to_skip, field_ty)) in predicates {
            structure.add_where_predicate(match when_to_skip {
                Always(_) => skip_traversal(&ty),
                // we only need to add traversable predicate for generic types
                Never if field_ty == Generic => parse_quote! { #ty: #traversable },
                _ => continue,
            });
        }
        quote! { match self { #arms } }
    };

    Ok(structure.bound_impl(traversable, T::impl_body(interner, traverser, body)))
}
