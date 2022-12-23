use quote::quote;
use syn::{parse_quote, Meta, NestedMeta};

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
            quote! {
                ::rustc_middle::ty::fold::TypeFoldable::try_fold_with(#bind, __folder)?
            }
        })
    });

    let is_inline_always = s.ast().attrs.iter().any(|attr| {
        let Ok(Meta::List(list)) = attr.parse_meta() else { return false; };

        if list.path.get_ident().map_or(true, |ident| ident.to_string() != "type_foldable") {
            return false;
        }

        let is_inline_always = list.nested.iter().any(|nested| match nested {
            NestedMeta::Meta(Meta::Path(path)) => {
                path.get_ident().map_or(false, |ident| ident.to_string() == "inline_always")
            }
            _ => false,
        });
        is_inline_always
    });

    let inline_attr = if is_inline_always {
        quote! { #[inline(always)] }
    } else {
        quote! {}
    };

    s.bound_impl(
        quote!(::rustc_middle::ty::fold::TypeFoldable<'tcx>),
        quote! {
            #inline_attr
            fn try_fold_with<__F: ::rustc_middle::ty::fold::FallibleTypeFolder<'tcx>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #body_fold })
            }
        },
    )
}
