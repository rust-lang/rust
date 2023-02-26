use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_quote, Attribute, Meta, NestedMeta};

pub struct Foldable;
pub struct Visitable;

/// An abstraction over traversable traits.
pub trait Traversable {
    /// The trait that this `Traversable` represents.
    fn traversable() -> TokenStream;

    /// The `match` arms for a traversal of this type.
    fn arms(structure: &mut synstructure::Structure<'_>) -> TokenStream;

    /// The body of an implementation given the match `arms`.
    fn impl_body(arms: impl ToTokens) -> TokenStream;
}

impl Traversable for Foldable {
    fn traversable() -> TokenStream {
        quote! { ::rustc_middle::ty::fold::TypeFoldable<::rustc_middle::ty::TyCtxt<'tcx>> }
    }
    fn arms(structure: &mut synstructure::Structure<'_>) -> TokenStream {
        structure.each_variant(|vi| {
            let bindings = vi.bindings();
            vi.construct(|_, index| {
                let bind = &bindings[index];

                // retain value of fields with #[type_foldable(identity)]
                let fixed = bind
                    .ast()
                    .attrs
                    .iter()
                    .map(Attribute::parse_meta)
                    .filter_map(Result::ok)
                    .flat_map(|attr| match attr {
                        Meta::List(list) if list.path.is_ident("type_foldable") => list.nested,
                        _ => Default::default(),
                    })
                    .any(|nested| match nested {
                        NestedMeta::Meta(Meta::Path(path)) => path.is_ident("identity"),
                        _ => false,
                    });

                if fixed {
                    bind.to_token_stream()
                } else {
                    quote! {
                        ::rustc_middle::ty::fold::TypeFoldable::try_fold_with(#bind, __folder)?
                    }
                }
            })
        })
    }
    fn impl_body(arms: impl ToTokens) -> TokenStream {
        quote! {
            fn try_fold_with<__F: ::rustc_middle::ty::fold::FallibleTypeFolder<::rustc_middle::ty::TyCtxt<'tcx>>>(
                self,
                __folder: &mut __F
            ) -> ::core::result::Result<Self, __F::Error> {
                ::core::result::Result::Ok(match self { #arms })
            }
        }
    }
}

impl Traversable for Visitable {
    fn traversable() -> TokenStream {
        quote! { ::rustc_middle::ty::visit::TypeVisitable<::rustc_middle::ty::TyCtxt<'tcx>> }
    }
    fn arms(structure: &mut synstructure::Structure<'_>) -> TokenStream {
        // ignore fields with #[type_visitable(ignore)]
        structure.filter(|bi| {
            !bi.ast()
                .attrs
                .iter()
                .map(Attribute::parse_meta)
                .filter_map(Result::ok)
                .flat_map(|attr| match attr {
                    Meta::List(list) if list.path.is_ident("type_visitable") => list.nested,
                    _ => Default::default(),
                })
                .any(|nested| match nested {
                    NestedMeta::Meta(Meta::Path(path)) => path.is_ident("ignore"),
                    _ => false,
                })
        });

        structure.each(|bind| {
            quote! {
                ::rustc_middle::ty::visit::TypeVisitable::visit_with(#bind, __visitor)?;
            }
        })
    }
    fn impl_body(arms: impl ToTokens) -> TokenStream {
        quote! {
            fn visit_with<__V: ::rustc_middle::ty::visit::TypeVisitor<::rustc_middle::ty::TyCtxt<'tcx>>>(
                &self,
                __visitor: &mut __V
            ) -> ::std::ops::ControlFlow<__V::BreakTy> {
                match self { #arms }
                ::std::ops::ControlFlow::Continue(())
            }
        }
    }
}

pub fn traversable_derive<T: Traversable>(
    mut structure: synstructure::Structure<'_>,
) -> TokenStream {
    if let syn::Data::Union(_) = structure.ast().data {
        panic!("cannot derive on union")
    }

    structure.add_bounds(synstructure::AddBounds::Generics);
    structure.bind_with(|_| synstructure::BindStyle::Move);

    if !structure.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
        structure.add_impl_generic(parse_quote! { 'tcx });
    }

    let arms = T::arms(&mut structure);
    structure.bound_impl(T::traversable(), T::impl_body(arms))
}
