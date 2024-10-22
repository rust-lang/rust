use quote::quote;
use syn::parse_quote;

pub(super) fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    s.underscore_const(true);

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "tcx") {
        s.add_impl_generic(parse_quote! { 'tcx });
    }

    s.add_bounds(synstructure::AddBounds::None);
    let mut where_clauses = None;
    s.add_trait_bounds(
        &parse_quote!(::rustc_type_ir::traverse::OptTryFoldWith<::rustc_middle::ty::TyCtxt<'tcx>>),
        &mut where_clauses,
        synstructure::AddBounds::Fields,
    );
    s.add_where_predicate(parse_quote! { Self: std::fmt::Debug + Clone });
    for pred in where_clauses.into_iter().flat_map(|c| c.predicates) {
        s.add_where_predicate(pred);
    }

    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];
            quote! {
                ::rustc_middle::ty::traverse::OptTryFoldWith::mk_try_fold_with()(#bind, __folder)?
            }
        })
    });

    s.bound_impl(
        quote!(::rustc_middle::ty::fold::TypeFoldable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            fn try_fold_with<__F: ::rustc_middle::ty::fold::FallibleTypeFolder<::rustc_middle::ty::TyCtxt<'tcx>>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #body_fold })
            }
        },
    )
}
