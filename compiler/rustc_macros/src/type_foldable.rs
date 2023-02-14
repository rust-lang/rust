use quote::{quote, ToTokens};
use syn::{parse_quote, Attribute, Meta, NestedMeta};

pub fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
        s.add_impl_generic(parse_quote! { 'tcx });
    }

    s.add_bounds(synstructure::AddBounds::Generics);
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_fold = s.each_variant(|vi| {
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
                    ::rustc_middle::ty::fold::ir::TypeFoldable::try_fold_with(#bind, __folder)?
                }
            }
        })
    });

    s.bound_impl(
        quote!(::rustc_middle::ty::fold::ir::TypeFoldable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            fn try_fold_with<__F: ::rustc_middle::ty::fold::FallibleTypeFolder<'tcx>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #body_fold })
            }
        },
    )
}
