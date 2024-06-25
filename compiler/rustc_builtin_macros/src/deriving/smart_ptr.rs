use std::mem::swap;

use ast::HasAttrs;
use rustc_ast::{
    self as ast, GenericArg, GenericBound, GenericParamKind, ItemKind, MetaItem,
    TraitBoundModifiers,
};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use smallvec::{smallvec, SmallVec};
use thin_vec::{thin_vec, ThinVec};

macro_rules! path {
    ($span:expr, $($part:ident)::*) => { vec![$(Ident::new(sym::$part, $span),)*] }
}

pub fn expand_deriving_smart_ptr(
    cx: &ExtCtxt<'_>,
    span: Span,
    _mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    _is_const: bool,
) {
    let (name_ident, generics) = if let Annotatable::Item(aitem) = item
        && let ItemKind::Struct(_, g) = &aitem.kind
    {
        (aitem.ident, g)
    } else {
        cx.dcx().struct_span_err(span, "`SmartPointer` can only be derived on `struct`s").emit();
        return;
    };

    // Convert generic parameters (from the struct) into generic args.
    let mut pointee_param = None;
    let mut multiple_pointee_diag: SmallVec<[_; 2]> = smallvec![];
    let self_params = generics
        .params
        .iter()
        .enumerate()
        .map(|(idx, p)| match p.kind {
            GenericParamKind::Lifetime => GenericArg::Lifetime(cx.lifetime(p.span(), p.ident)),
            GenericParamKind::Type { .. } => {
                if p.attrs().iter().any(|attr| attr.has_name(sym::pointee)) {
                    if pointee_param.is_some() {
                        multiple_pointee_diag.push(cx.dcx().struct_span_err(
                            p.span(),
                            "`SmartPointer` can only admit one type as pointee",
                        ));
                    } else {
                        pointee_param = Some(idx);
                    }
                }
                GenericArg::Type(cx.ty_ident(p.span(), p.ident))
            }
            GenericParamKind::Const { .. } => GenericArg::Const(cx.const_ident(p.span(), p.ident)),
        })
        .collect::<Vec<_>>();
    let Some(pointee_param_idx) = pointee_param else {
        cx.dcx().struct_span_err(
            span,
            "At least one generic type should be designated as `#[pointee]` in order to derive `SmartPointer` traits",
        ).emit();
        return;
    };
    if !multiple_pointee_diag.is_empty() {
        for diag in multiple_pointee_diag {
            diag.emit();
        }
        return;
    }

    // Create the type of `self`.
    let path = cx.path_all(span, false, vec![name_ident], self_params.clone());
    let self_type = cx.ty_path(path);

    // Declare helper function that adds implementation blocks.
    // FIXME(dingxiangfei2009): Investigate the set of attributes on target struct to be propagated to impls
    let attrs = thin_vec![cx.attr_word(sym::automatically_derived, span),];
    let mut add_impl_block = |generics, trait_symbol, trait_args| {
        let mut parts = path!(span, core::ops);
        parts.push(Ident::new(trait_symbol, span));
        let trait_path = cx.path_all(span, true, parts, trait_args);
        let trait_ref = cx.trait_ref(trait_path);
        let item = cx.item(
            span,
            Ident::empty(),
            attrs.clone(),
            ast::ItemKind::Impl(Box::new(ast::Impl {
                safety: ast::Safety::Default,
                polarity: ast::ImplPolarity::Positive,
                defaultness: ast::Defaultness::Final,
                constness: ast::Const::No,
                generics,
                of_trait: Some(trait_ref),
                self_ty: self_type.clone(),
                items: ThinVec::new(),
            })),
        );
        push(Annotatable::Item(item));
    };

    // Create unsized `self`, that is, one where the `#[pointee]` type arg is replaced with `__S`. For
    // example, instead of `MyType<'a, T>`, it will be `MyType<'a, __S>`.
    let s_ty = cx.ty_ident(span, Ident::new(sym::__S, span));
    let mut alt_self_params = self_params;
    alt_self_params[pointee_param_idx] = GenericArg::Type(s_ty.clone());
    let alt_self_type = cx.ty_path(cx.path_all(span, false, vec![name_ident], alt_self_params));

    // Find the `#[pointee]` parameter and add an `Unsize<__S>` bound to it.
    let mut impl_generics = generics.clone();
    {
        let p = &mut impl_generics.params[pointee_param_idx];
        let arg = GenericArg::Type(s_ty.clone());
        let unsize = cx.path_all(span, true, path!(span, core::marker::Unsize), vec![arg]);
        p.bounds.push(cx.trait_bound(unsize, false));
        let mut attrs = thin_vec![];
        swap(&mut p.attrs, &mut attrs);
        p.attrs = attrs.into_iter().filter(|attr| !attr.has_name(sym::pointee)).collect();
    }

    // Add the `__S: ?Sized` extra parameter to the impl block.
    let sized = cx.path_global(span, path!(span, core::marker::Sized));
    let bound = GenericBound::Trait(
        cx.poly_trait_ref(span, sized),
        TraitBoundModifiers {
            polarity: ast::BoundPolarity::Maybe(span),
            constness: ast::BoundConstness::Never,
            asyncness: ast::BoundAsyncness::Normal,
        },
    );
    let extra_param = cx.typaram(span, Ident::new(sym::__S, span), vec![bound], None);
    impl_generics.params.push(extra_param);

    // Add the impl blocks for `DispatchFromDyn` and `CoerceUnsized`.
    let gen_args = vec![GenericArg::Type(alt_self_type.clone())];
    add_impl_block(impl_generics.clone(), sym::DispatchFromDyn, gen_args.clone());
    add_impl_block(impl_generics.clone(), sym::CoerceUnsized, gen_args.clone());
}
