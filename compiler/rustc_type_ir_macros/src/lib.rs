use quote::{ToTokens, quote};
use syn::visit_mut::VisitMut;
use syn::{Attribute, parse_quote};
use synstructure::decl_derive;

decl_derive!(
    [TypeVisitable_Generic, attributes(type_visitable)] => type_visitable_derive
);
decl_derive!(
    [TypeFoldable_Generic, attributes(type_foldable)] => type_foldable_derive
);
decl_derive!(
    [Lift_Generic] => lift_derive
);

fn has_ignore_attr(attrs: &[Attribute], name: &'static str, meta: &'static str) -> bool {
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

fn type_visitable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    if !s.ast().generics.type_params().any(|ty| ty.ident == "I") {
        s.add_impl_generic(parse_quote! { I });
    }

    s.filter(|bi| !has_ignore_attr(&bi.ast().attrs, "type_visitable", "ignore"));

    s.add_where_predicate(parse_quote! { I: Interner });
    s.add_bounds(synstructure::AddBounds::Fields);
    let body_visit = s.each(|bind| {
        quote! {
            match ::rustc_type_ir::VisitorResult::branch(
                ::rustc_type_ir::TypeVisitable::visit_with(#bind, __visitor)
            ) {
                ::core::ops::ControlFlow::Continue(()) => {},
                ::core::ops::ControlFlow::Break(r) => {
                    return ::rustc_type_ir::VisitorResult::from_residual(r);
                },
            }
        }
    });
    s.bind_with(|_| synstructure::BindStyle::Move);

    s.bound_impl(
        quote!(::rustc_type_ir::TypeVisitable<I>),
        quote! {
            fn visit_with<__V: ::rustc_type_ir::TypeVisitor<I>>(
                &self,
                __visitor: &mut __V
            ) -> __V::Result {
                match *self { #body_visit }
                <__V::Result as ::rustc_type_ir::VisitorResult>::output()
            }
        },
    )
}

fn type_foldable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    if !s.ast().generics.type_params().any(|ty| ty.ident == "I") {
        s.add_impl_generic(parse_quote! { I });
    }

    s.add_where_predicate(parse_quote! { I: Interner });
    s.add_bounds(synstructure::AddBounds::Fields);
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_try_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];

            // retain value of fields with #[type_foldable(identity)]
            if has_ignore_attr(&bind.ast().attrs, "type_foldable", "identity") {
                bind.to_token_stream()
            } else {
                quote! {
                    ::rustc_type_ir::TypeFoldable::try_fold_with(#bind, __folder)?
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
                    ::rustc_type_ir::TypeFoldable::fold_with(#bind, __folder)
                }
            }
        })
    });

    // We filter fields which get ignored and don't require them to implement
    // `TypeFoldable`. We do so after generating `body_fold` as we still need
    // to generate code for them.
    s.filter(|bi| !has_ignore_attr(&bi.ast().attrs, "type_foldable", "identity"));
    s.add_bounds(synstructure::AddBounds::Fields);
    s.bound_impl(
        quote!(::rustc_type_ir::TypeFoldable<I>),
        quote! {
            fn try_fold_with<__F: ::rustc_type_ir::FallibleTypeFolder<I>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #body_try_fold })
            }

            fn fold_with<__F: ::rustc_type_ir::TypeFolder<I>>(
                self,
                __folder: &mut __F
            ) -> Self {
                match self { #body_fold }
            }
        },
    )
}

fn lift_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    if !s.ast().generics.type_params().any(|ty| ty.ident == "I") {
        s.add_impl_generic(parse_quote! { I });
    }

    s.add_bounds(synstructure::AddBounds::None);
    s.add_where_predicate(parse_quote! { I: Interner });
    s.add_impl_generic(parse_quote! { J });
    s.add_where_predicate(parse_quote! { J: Interner });

    let mut wc = vec![];
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|field, index| {
            let ty = field.ty.clone();
            let lifted_ty = lift(ty.clone());
            wc.push(parse_quote! { #ty: ::rustc_type_ir::lift::Lift<J, Lifted = #lifted_ty> });
            let bind = &bindings[index];
            quote! {
                #bind.lift_to_interner(interner)?
            }
        })
    });
    for wc in wc {
        s.add_where_predicate(wc);
    }

    let (_, ty_generics, _) = s.ast().generics.split_for_impl();
    let name = s.ast().ident.clone();
    let self_ty: syn::Type = parse_quote! { #name #ty_generics };
    let lifted_ty = lift(self_ty);

    s.bound_impl(
        quote!(::rustc_type_ir::lift::Lift<J>),
        quote! {
            type Lifted = #lifted_ty;

            fn lift_to_interner(
                self,
                interner: J,
            ) -> Option<Self::Lifted> {
                Some(match self { #body_fold })
            }
        },
    )
}

fn lift(mut ty: syn::Type) -> syn::Type {
    struct ItoJ;
    impl VisitMut for ItoJ {
        fn visit_type_path_mut(&mut self, i: &mut syn::TypePath) {
            if i.qself.is_none() {
                if let Some(first) = i.path.segments.first_mut() {
                    if first.ident == "I" {
                        *first = parse_quote! { J };
                    }
                }
            }
            syn::visit_mut::visit_type_path_mut(self, i);
        }
    }

    ItoJ.visit_type_mut(&mut ty);

    ty
}
