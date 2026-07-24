use indexmap::IndexSet;
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
    [Lift_Generic, attributes(lift)] => lift_derive
);
#[cfg(not(feature = "nightly"))]
decl_derive!(
    [GenericTypeVisitable] => customizable_type_visitable_derive
);

struct TransformedTy {
    ty: syn::Type,
    generic_parameter_bounds: IndexSet<syn::Ident>,
}

enum TypeParameterPath {
    Interner,
    GenericParameter(syn::Ident),
}

enum TypeParameterTransform {
    Continue,
    Stop,
}

type TypeParameterVisitor =
    fn(TypeParameterPath, &mut syn::TypePath, &mut IndexSet<syn::Ident>) -> TypeParameterTransform;

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
    let generic_parameters =
        s.ast().generics.type_params().map(|ty| ty.ident.clone()).collect::<Vec<_>>();
    let mut generic_parameter_bounds = IndexSet::new();
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_try_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|_, index| {
            let bind = &bindings[index];

            // retain value of fields with #[type_foldable(identity)]
            if has_ignore_attr(&bind.ast().attrs, "type_foldable", "identity") {
                bind.to_token_stream()
            } else {
                for param in
                    type_foldable_generic_parameters(bind.ast().ty.clone(), &generic_parameters)
                {
                    generic_parameter_bounds.insert(param);
                }

                quote! {
                    ::rustc_type_ir::TypeFoldable::try_fold_with(#bind, __folder)?
                }
            }
        })
    });

    // We filter fields which get ignored and don't require them to implement
    // `TypeFoldable`. We do so after generating `body_fold` as we still need
    // to generate code for them.
    s.filter(|bi| !has_ignore_attr(&bi.ast().attrs, "type_foldable", "identity"));
    s.add_bounds(synstructure::AddBounds::Fields);
    for param in generic_parameter_bounds {
        s.add_where_predicate(parse_quote! { #param: ::rustc_type_ir::TypeFoldable<I> });
    }
    s.bound_impl(
        quote!(::rustc_type_ir::TypeFoldable<I>),
        quote! {
            fn try_fold_with<__F: ::rustc_type_ir::FallibleTypeFolder<I>>(
                self,
                __folder: &mut __F
            ) -> Result<Self, __F::Error> {
                Ok(match self { #body_try_fold })
            }
        },
    )
}

fn type_foldable_generic_parameters(
    ty: syn::Type,
    generic_parameters: &[syn::Ident],
) -> IndexSet<syn::Ident> {
    transform_type_parameters(ty, generic_parameters, |path, _, generic_parameter_bounds| {
        if let TypeParameterPath::GenericParameter(param) = path {
            generic_parameter_bounds.insert(param);
        }
        TypeParameterTransform::Continue
    })
    .generic_parameter_bounds
}

/// `Lift_Generic` is specialised for structs/enums parameterised by an interner
/// `I: Interner`. It derives `Lift<J>` by rewriting interner associated types
/// from `I::Assoc` to `J::Assoc`. The required associated type lift bounds are
/// supplied by `I: LiftInto<J>`.
///
/// Ordinary generic parameters still get explicit `Lift<J>` bounds. Interner
/// independent fields must either implement `Lift` manually or use
/// `#[lift(identity)]`.
///
/// `PhantomData` is a special case that occurs enough in the code base to be
/// handled here directly. We collect any generic bounds from the type then
/// produce another `PhantomData`.
fn lift_derive(mut s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    if !s.ast().generics.type_params().any(|ty| ty.ident == "I") {
        s.add_impl_generic(parse_quote! { I });
    }

    s.add_bounds(synstructure::AddBounds::None);
    s.add_impl_generic(parse_quote! { J });
    s.add_where_predicate(parse_quote! { J: Interner });
    s.add_where_predicate(parse_quote! { I: ::rustc_type_ir::LiftInto<J> });

    let generic_parameters =
        s.ast().generics.type_params().map(|ty| ty.ident.clone()).collect::<Vec<_>>();

    let mut wc = vec![];
    s.bind_with(|_| synstructure::BindStyle::Move);
    let body_fold = s.each_variant(|vi| {
        let bindings = vi.bindings();
        vi.construct(|field, index| {
            let ty = field.ty.clone();
            let bind = &bindings[index];
            // Allow field to be ignored from lift
            if has_ignore_attr(&field.attrs, "lift", "identity") {
                return bind.to_token_stream();
            }

            let lifted = lift(ty.clone(), &generic_parameters);

            // Field types involving ordinary generic parameters still need
            // explicit bounds for those parameters, e.g. `Binder<I, T>` needs
            // `T: Lift<J>` so its own derived `Lift` impl applies. Interner
            // associated types are covered by `I: LiftInto<J>`.
            for param in lifted.generic_parameter_bounds {
                wc.push(parse_quote! { #param: ::rustc_type_ir::lift::Lift<J> });
            }

            if is_type_phantom(&ty) {
                return quote! {
                    PhantomData
                };
            }

            quote! {
                #bind.lift_to_interner(interner)
            }
        })
    });
    for wc in wc {
        s.add_where_predicate(wc);
    }

    let (_, ty_generics, _) = s.ast().generics.split_for_impl();
    let name = s.ast().ident.clone();
    let self_ty: syn::Type = parse_quote! { #name #ty_generics };
    let lifted = lift(self_ty, &generic_parameters);
    let lifted_ty = lifted.ty;

    s.bound_impl(
        quote!(::rustc_type_ir::lift::Lift<J>),
        quote! {
            type Lifted = #lifted_ty;

            fn lift_to_interner(
                self,
                interner: J,
            ) -> Self::Lifted {
                match self { #body_fold }
            }
        },
    )
}

fn get_first_path_segment(ty: &syn::Type) -> Option<&syn::PathSegment> {
    if let syn::Type::Path(ty) = ty
        && ty.path.segments.len() == 1
    {
        ty.path.segments.first()
    } else {
        None
    }
}

/// Return if the type is `PhantomData`
fn is_type_phantom(ty: &syn::Type) -> bool {
    get_first_path_segment(ty).is_some_and(|segment| segment.ident == "PhantomData")
}

fn lift(ty: syn::Type, generic_parameters: &[syn::Ident]) -> TransformedTy {
    transform_type_parameters(ty, generic_parameters, |path, ty, generic_parameter_bounds| {
        match path {
            TypeParameterPath::Interner => {
                *ty.path.segments.first_mut().unwrap() = parse_quote! { J };
                TypeParameterTransform::Continue
            }
            TypeParameterPath::GenericParameter(param) => {
                generic_parameter_bounds.insert(param.clone());
                *ty = parse_quote! { <#param as ::rustc_type_ir::lift::Lift<J>>::Lifted };
                TypeParameterTransform::Stop
            }
        }
    })
}

fn transform_type_parameters(
    mut ty: syn::Type,
    generic_parameters: &[syn::Ident],
    visit: TypeParameterVisitor,
) -> TransformedTy {
    struct TypeParameterTransformer<'a> {
        generic_parameters: &'a [syn::Ident],
        generic_parameter_bounds: IndexSet<syn::Ident>,
        visit: TypeParameterVisitor,
    }

    impl VisitMut for TypeParameterTransformer<'_> {
        fn visit_type_path_mut(&mut self, i: &mut syn::TypePath) {
            let path = if i.qself.is_none() {
                let segments_len = i.path.segments.len();
                i.path.segments.first().and_then(|first| {
                    if first.ident == "I" {
                        Some(TypeParameterPath::Interner)
                    } else if segments_len == 1
                        && matches!(first.arguments, syn::PathArguments::None)
                        && self.generic_parameters.contains(&first.ident)
                    {
                        Some(TypeParameterPath::GenericParameter(first.ident.clone()))
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            if let Some(path) = path {
                if let TypeParameterTransform::Stop =
                    (self.visit)(path, i, &mut self.generic_parameter_bounds)
                {
                    return;
                }
            }

            syn::visit_mut::visit_type_path_mut(self, i);
        }
    }

    let mut visitor = TypeParameterTransformer {
        generic_parameters,
        generic_parameter_bounds: IndexSet::new(),
        visit,
    };
    visitor.visit_type_mut(&mut ty);
    TransformedTy { ty, generic_parameter_bounds: visitor.generic_parameter_bounds }
}

#[cfg(not(feature = "nightly"))]
fn customizable_type_visitable_derive(
    mut s: synstructure::Structure<'_>,
) -> proc_macro2::TokenStream {
    if let syn::Data::Union(_) = s.ast().data {
        panic!("cannot derive on union")
    }

    s.add_impl_generic(parse_quote!(__V));
    s.add_bounds(synstructure::AddBounds::Fields);
    let body_visit = s.each(|bind| {
        quote! {
            ::rustc_type_ir::GenericTypeVisitable::<__V>::generic_visit_with(#bind, __visitor);
        }
    });
    s.bind_with(|_| synstructure::BindStyle::Move);

    s.bound_impl(
        quote!(::rustc_type_ir::GenericTypeVisitable<__V>),
        quote! {
            fn generic_visit_with(
                &self,
                __visitor: &mut __V
            ) {
                match *self { #body_visit }
            }
        },
    )
}

#[cfg(feature = "nightly")]
#[proc_macro_derive(GenericTypeVisitable)]
pub fn customizable_type_visitable_derive(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}
