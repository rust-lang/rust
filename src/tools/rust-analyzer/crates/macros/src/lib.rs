//! Proc macros for rust-analyzer.

use quote::{ToTokens, quote};
use syn::parse_quote;
use synstructure::decl_derive;

decl_derive!(
    [TypeFoldable, attributes(type_foldable)] =>
    /// Derives `TypeFoldable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// The fold will produce a value of the same struct or enum variant as the input, with
    /// each field respectively folded using the `TypeFoldable` implementation for its type.
    /// However, if a field of a struct or an enum variant is annotated with
    /// `#[type_foldable(identity)]` then that field will retain its incumbent value (and its
    /// type is not required to implement `TypeFoldable`).
    type_foldable_derive
);
decl_derive!(
    [TypeVisitable, attributes(type_visitable)] =>
    /// Derives `TypeVisitable` for the annotated `struct` or `enum` (`union` is not supported).
    ///
    /// Each field of the struct or enum variant will be visited in definition order, using the
    /// `TypeVisitable` implementation for its type. However, if a field of a struct or an enum
    /// variant is annotated with `#[type_visitable(ignore)]` then that field will not be
    /// visited (and its type is not required to implement `TypeVisitable`).
    type_visitable_derive
);

fn type_visitable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
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

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "db") {
        s.add_impl_generic(parse_quote! { 'db });
    }

    s.add_bounds(synstructure::AddBounds::Generics);
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
        quote!(::rustc_type_ir::TypeVisitable<::hir_ty::next_solver::DbInterner<'db>>),
        quote! {
            fn visit_with<__V: ::rustc_type_ir::TypeVisitor<::hir_ty::next_solver::DbInterner<'db>>>(
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

    if !s.ast().generics.lifetimes().any(|lt| lt.lifetime.ident == "db") {
        s.add_impl_generic(parse_quote! { 'db });
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

    s.bound_impl(
        quote!(::rustc_type_ir::TypeFoldable<::hir_ty::next_solver::DbInterner<'db>>),
        quote! {
            fn try_fold_with<__F: ::rustc_type_ir::FallibleTypeFolder<::hir_ty::next_solver::DbInterner<'db>>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #try_body_fold })
            }

            fn fold_with<__F: ::rustc_type_ir::TypeFolder<::hir_ty::next_solver::DbInterner<'db>>>(
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

decl_derive!(
    [UpmapFromRaFixture] => upmap_from_ra_fixture
);

fn upmap_from_ra_fixture(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    s.add_bounds(synstructure::AddBounds::Generics);
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];

            quote! {
                ::ide_db::ra_fixture::UpmapFromRaFixture::upmap_from_ra_fixture(
                    #bind, __analysis, __virtual_file_id, __real_file_id,
                )?
            }
        })
    });

    s.bound_impl(
        quote!(::ide_db::ra_fixture::UpmapFromRaFixture),
        quote! {
            fn upmap_from_ra_fixture(
                self,
                __analysis: &::ide_db::ra_fixture::RaFixtureAnalysis,
                __virtual_file_id: ::ide_db::ra_fixture::FileId,
                __real_file_id: ::ide_db::ra_fixture::FileId,
            ) -> Result<Self, ()> {
                Ok(match self { #body })
            }
        },
    )
}
