use quote::quote;

pub fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    s.add_bounds(synstructure::AddBounds::Generics);
    let body_visit = s.each(|bind| {
        quote! {
            ::rustc_middle::ty::fold::TypeFoldable::visit_with(#bind, __folder)?;
        }
    });
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];
            quote! {
                ::rustc_middle::ty::fold::TypeFoldable::fold_with(#bind, __folder)
            }
        })
    });

    s.bound_impl(
        quote!(::rustc_middle::ty::fold::TypeFoldable<'tcx>),
        quote! {
            fn super_fold_with<__F: ::rustc_middle::ty::fold::TypeFolder<'tcx>>(
                self,
                __folder: &mut __F
            ) -> Self {
                match self { #body_fold }
            }

            fn super_visit_with<__F: ::rustc_middle::ty::fold::TypeVisitor<'tcx>>(
                &self,
                __folder: &mut __F
            ) -> ::std::ops::ControlFlow<__F::BreakTy> {
                match *self { #body_visit }
                ::std::ops::ControlFlow::CONTINUE
            }
        },
    )
}
