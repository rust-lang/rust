use quote::quote;
use syn::parse_quote;

pub(super) fn noop_type_traversable_derive(
    mut s: synstructure::Structure<'_>,
) -> proc_macro2::TokenStream {
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
        &parse_quote!(
            ::rustc_middle::ty::traverse::TypeTraversable<
                ::rustc_middle::ty::TyCtxt<'tcx>,
                Kind = ::rustc_middle::ty::traverse::NoopTypeTraversal,
            >
        ),
        &mut where_clauses,
        synstructure::AddBounds::Fields,
    );
    for pred in where_clauses.into_iter().flat_map(|c| c.predicates) {
        s.add_where_predicate(pred);
    }

    s.bound_impl(
        quote!(::rustc_middle::ty::traverse::TypeTraversable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            type Kind = ::rustc_middle::ty::traverse::NoopTypeTraversal;
        },
    )
}
