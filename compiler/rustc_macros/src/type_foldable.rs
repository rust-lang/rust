use quote::{ToTokens, quote};
use syn::parse_quote;

pub(super) fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    s.underscore_const(true);

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
        s.add_impl_generic(parse_quote! { 'tcx });
    }

    s.add_bounds(synstructure::AddBounds::Generics);
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];

            let mut fixed = false;

            // retain value of fields with #[type_foldable(identity)]
            bind.ast().attrs.iter().for_each(|x| {
                if !x.path().is_ident("type_foldable") {
                    return;
                }
                let _ = x.parse_nested_meta(|nested| {
                    if nested.path.is_ident("identity") {
                        fixed = true;
                    }
                    Ok(())
                });
            });

            if fixed {
                bind.to_token_stream()
            } else {
                quote! {
                    ::rustc_middle::ty::TypeFoldable::try_fold_with(#bind, __folder)?
                }
            }
        })
    });

    s.bound_impl(
        quote!(::rustc_middle::ty::TypeFoldable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            fn try_fold_with<__F: ::rustc_middle::ty::FallibleTypeFolder<::rustc_middle::ty::TyCtxt<'tcx>>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #body_fold })
            }
        },
    )
}
