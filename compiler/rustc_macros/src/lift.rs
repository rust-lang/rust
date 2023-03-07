use quote::quote;
use syn::{self, parse_quote};

pub fn lift_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    s.add_bounds(synstructure::AddBounds::Generics);
    s.bind_with(|_| synstructure::BindStyle::Move);

    let tcx: syn::Lifetime = parse_quote!('tcx);
    let newtcx: syn::GenericParam = parse_quote!('__lifted);

    let lifted = {
        let ast = s.ast();
        let ident = &ast.ident;

        // Replace `'tcx` lifetime by the `'__lifted` lifetime
        let (_, generics, _) = ast.generics.split_for_impl();
        let mut generics: syn::AngleBracketedGenericArguments = syn::parse_quote! { #generics };
        for arg in generics.args.iter_mut() {
            match arg {
                syn::GenericArgument::Lifetime(l) if *l == tcx => {
                    *arg = parse_quote!('__lifted);
                }
                syn::GenericArgument::Type(t) => {
                    *arg = syn::parse_quote! { #t::Lifted };
                }
                _ => {}
            }
        }

        quote! { #ident #generics }
    };

    let body = s.each_variant(|vi| {
        let bindings = &vi.bindings();
        vi.construct(|_, index| {
            let bi = &bindings[index];
            quote! { __tcx.lift(#bi)?  }
        })
    });

    s.add_impl_generic(newtcx);
    s.bound_impl(
        quote!(::rustc_middle::ty::Lift<'__lifted>),
        quote! {
            type Lifted = #lifted;

            fn lift_to_tcx(self, __tcx: ::rustc_middle::ty::TyCtxt<'__lifted>) -> Option<#lifted> {
                Some(match self { #body })
            }
        },
    )
}
