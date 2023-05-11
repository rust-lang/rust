use quote::quote;
use syn::parse_quote;

pub fn type_visitable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    // ignore fields with #[type_visitable(ignore)]
    s.filter(|bi| {
        let mut ignored = false;

        bi.ast().attrs.iter().for_each(|attr| {
            if !attr.path().is_ident("type_visitable") {
                return;
            }
            let _ = attr.parse_nested_meta(|nested| {
                if nested.path.is_ident("ignore") {
                    ignored = true;
                }
                Ok(())
            });
        });

        !ignored
    });

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
        s.add_impl_generic(parse_quote! { 'tcx });
    }

    s.add_bounds(synstructure::AddBounds::Generics);
    let body_visit = s.each(|bind| {
        quote! {
            ::rustc_middle::ty::visit::TypeVisitable::visit_with(#bind, __visitor)?;
        }
    });
    s.bind_with(|_| synstructure::BindStyle::Move);

    s.bound_impl(
        quote!(::rustc_middle::ty::visit::TypeVisitable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            fn visit_with<__V: ::rustc_middle::ty::visit::TypeVisitor<::rustc_middle::ty::TyCtxt<'tcx>>>(
                &self,
                __visitor: &mut __V
            ) -> ::std::ops::ControlFlow<__V::BreakTy> {
                match *self { #body_visit }
                ::std::ops::ControlFlow::Continue(())
            }
        },
    )
}
