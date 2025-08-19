use quote::quote;
use synstructure::BindingInfo;

pub(super) fn visitable_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    let has_attr = |bind: &BindingInfo<'_>, name| {
        let mut found = false;
        bind.ast().attrs.iter().for_each(|attr| {
            if !attr.path().is_ident("visitable") {
                return;
            }
            let _ = attr.parse_nested_meta(|nested| {
                if nested.path.is_ident(name) {
                    found = true;
                }
                Ok(())
            });
        });
        found
    };

    let get_attr = |bind: &BindingInfo<'_>, name: &str| {
        let mut content = None;
        bind.ast().attrs.iter().for_each(|attr| {
            if !attr.path().is_ident("visitable") {
                return;
            }
            let _ = attr.parse_nested_meta(|nested| {
                if nested.path.is_ident(name) {
                    let value = nested.value()?;
                    let value = value.parse()?;
                    content = Some(value);
                }
                Ok(())
            });
        });
        content
    };

    s.add_bounds(synstructure::AddBounds::Generics);
    s.bind_with(|_| synstructure::BindStyle::Ref);
    let ref_visit = s.each(|bind| {
        let extra = get_attr(bind, "extra").unwrap_or(quote! {});
        if has_attr(bind, "ignore") {
            quote! {}
        } else {
            quote! { rustc_ast_ir::try_visit!(crate::visit::Visitable::visit(#bind, __visitor, (#extra))) }
        }
    });

    s.bind_with(|_| synstructure::BindStyle::RefMut);
    let mut_visit = s.each(|bind| {
        let extra = get_attr(bind, "extra").unwrap_or(quote! {});
        if has_attr(bind, "ignore") {
            quote! {}
        } else {
            quote! { crate::mut_visit::MutVisitable::visit_mut(#bind, __visitor, (#extra)) }
        }
    });

    s.gen_impl(quote! {
        gen impl<'__ast, __V> crate::visit::Walkable<'__ast, __V> for @Self
            where __V: crate::visit::Visitor<'__ast>,
        {
            fn walk_ref(&'__ast self, __visitor: &mut __V) -> __V::Result {
                match *self { #ref_visit }
                <__V::Result as rustc_ast_ir::visit::VisitorResult>::output()
            }
        }

        gen impl<__V> crate::mut_visit::MutWalkable<__V> for @Self
            where __V: crate::mut_visit::MutVisitor,
        {
            fn walk_mut(&mut self, __visitor: &mut __V) {
                match *self { #mut_visit }
            }
        }
    })
}
