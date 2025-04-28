use quote::{ToTokens, quote};
use syn::parse_quote;

pub(super) fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
        s.add_impl_generic(parse_quote! { 'tcx });
    }

    s.add_bounds(synstructure::AddBounds::Generics);
    s.bind_with(|_| synstructure::BindStyle::Move);
    let try_body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];

            // retain value of fields with #[type_foldable(identity)]
            if has_ignore_attr(&bind.ast().attrs, "type_foldable", "identity") {
                bind.to_token_stream()
            } else {
                quote! {
                    ::rustc_middle::ty::TypeFoldable::try_fold_with(#bind, __folder)?
                }
            }
        })
    });

    let body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];

            // retain value of fields with #[type_foldable(identity)]
            if has_ignore_attr(&bind.ast().attrs, "type_foldable", "identity") {
                bind.to_token_stream()
            } else {
                quote! {
                    ::rustc_middle::ty::TypeFoldable::fold_with(#bind, __folder)
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
                Ok(match self { #try_body_fold })
            }

            fn fold_with<__F: ::rustc_middle::ty::TypeFolder<::rustc_middle::ty::TyCtxt<'tcx>>>(
                self,
                __folder: &mut __F
            ) -> Self {
                match self { #body_fold }
            }
        },
    )
}

fn has_ignore_attr(attrs: &[syn::Attribute], name: &'static str, meta: &'static str) -> bool {
    let mut ignored = false;
    attrs.iter().for_each(|attr| {
        if !attr.path().is_ident(name) {
            return;
        }
        let _ = attr.parse_nested_meta(|nested| {
            if nested.path.is_ident(meta) {
                ignored = true;
            }
            Ok(())
        });
    });

    ignored
}
