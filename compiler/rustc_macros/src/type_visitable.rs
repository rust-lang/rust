use quote::quote;
use syn::parse_quote;

pub(super) fn type_visitable_derive(
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
            ::rustc_middle::ty::traverse::OptVisitWith::<::rustc_middle::ty::TyCtxt<'tcx>>
        ),
        &mut where_clauses,
        synstructure::AddBounds::Fields,
    );
    s.add_where_predicate(parse_quote! { Self: std::fmt::Debug + Clone });
    for pred in where_clauses.into_iter().flat_map(|c| c.predicates) {
        s.add_where_predicate(pred);
    }

    let impl_traversable_s = s.clone();

    let body_visit = s.each(|bind| {
        quote! {
            match ::rustc_ast_ir::visit::VisitorResult::branch(
                ::rustc_middle::ty::traverse::OptVisitWith::mk_visit_with()(#bind, __visitor)
            ) {
                ::core::ops::ControlFlow::Continue(()) => {},
                ::core::ops::ControlFlow::Break(r) => {
                    return ::rustc_ast_ir::visit::VisitorResult::from_residual(r);
                },
            }
        }
    });
    s.bind_with(|_| synstructure::BindStyle::Move);

    let visitable_impl = s.bound_impl(
        quote!(::rustc_middle::ty::visit::TypeVisitable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            fn visit_with<__V: ::rustc_middle::ty::visit::TypeVisitor<::rustc_middle::ty::TyCtxt<'tcx>>>(
                &self,
                __visitor: &mut __V
            ) -> __V::Result {
                match *self { #body_visit }
                <__V::Result as ::rustc_ast_ir::visit::VisitorResult>::output()
            }
        },
    );

    let traversable_impl = impl_traversable_s.bound_impl(
        quote!(::rustc_middle::ty::traverse::TypeTraversable<::rustc_middle::ty::TyCtxt<'tcx>>),
        quote! {
            type Kind = ::rustc_middle::ty::traverse::ImportantTypeTraversal;
        },
    );

    quote! {
        #visitable_impl
        #traversable_impl
    }
}
