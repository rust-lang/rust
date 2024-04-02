use rustc_ast::{
    self as ast, GenericArg, GenericBound, GenericParamKind, ItemKind, MetaItem,
    TraitBoundModifiers,
};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
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
    let (name_ident, generics) = match item {
        Annotatable::Item(aitem) => match &aitem.kind {
            ItemKind::Struct(_, g) => (aitem.ident, g),
            // FIXME: Improve error reporting.
            _ => cx.dcx().span_bug(span, "`#[derive(SmartPointer)]` on wrong kind"),
        },
        _ => cx.dcx().span_bug(span, "`#[derive(SmartPointer)]` on wrong item"),
    };

    // Convert generic parameters (from the struct) into generic args.
    let self_params = generics
        .params
        .iter()
        .map(|p| match p.kind {
            GenericParamKind::Lifetime => GenericArg::Lifetime(cx.lifetime(span, p.ident)),
            GenericParamKind::Type { .. } => GenericArg::Type(cx.ty_ident(span, p.ident)),
            GenericParamKind::Const { .. } => GenericArg::Const(cx.const_ident(span, p.ident)),
        })
        .collect::<Vec<_>>();

    // Create the type of `self`.
    let path = cx.path_all(span, false, vec![name_ident], self_params.clone());
    let self_type = cx.ty_path(path);

    // Declare helper function that adds implementation blocks.
    // FIXME: Copy attrs from struct?
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
                unsafety: ast::Unsafe::No,
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

    // Create unsized `self`, that is, one where the first type arg is replace with `__S`. For
    // example, instead of `MyType<'a, T>`, it will be `MyType<'a, __S>`.
    let s_ty = cx.ty_ident(span, Ident::new(sym::__S, span));
    let mut alt_self_params = self_params;
    for a in &mut alt_self_params {
        if matches!(*a, GenericArg::Type(_)) {
            *a = GenericArg::Type(s_ty.clone());
            break;
        }
    }
    let alt_self_type = cx.ty_path(cx.path_all(span, false, vec![name_ident], alt_self_params));

    // Find the first type parameter and add an `Unsize<__S>` bound to it.
    let mut impl_generics = generics.clone();
    for p in &mut impl_generics.params {
        if matches!(p.kind, ast::GenericParamKind::Type { .. }) {
            let arg = GenericArg::Type(s_ty.clone());
            let unsize = cx.path_all(span, true, path!(span, core::marker::Unsize), vec![arg]);
            p.bounds.push(cx.trait_bound(unsize, false));
            break;
        }
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

    // Add the impl blocks for `DispatchFromDyn`, `CoerceUnsized`, and `Receiver`.
    let gen_args = vec![GenericArg::Type(alt_self_type.clone())];
    add_impl_block(impl_generics.clone(), sym::DispatchFromDyn, gen_args.clone());
    add_impl_block(impl_generics.clone(), sym::CoerceUnsized, gen_args.clone());
    add_impl_block(generics.clone(), sym::Receiver, Vec::new());
}
