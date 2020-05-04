use ast::{AttrStyle, Ident, MacArgs};
use rustc_ast::{ast, ptr::P, tokenstream::TokenStream};
use rustc_attr::{mk_attr, HasAttrs};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_feature::BUILTIN_ATTRIBUTE_MAP;
use rustc_parse::validate_attr;
use rustc_span::{sym, symbol::kw, Span};
use std::iter;

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    // Validate input against the `#[rustc_must_use]` template.
    let (_, _, template, _) = &BUILTIN_ATTRIBUTE_MAP[&sym::rustc_must_use];
    let attr = ecx.attribute(meta.clone());
    validate_attr::check_builtin_attribute(ecx.parse_sess, &attr, sym::must_use, template.clone());

    let reason = meta.name_value_literal();
    let mac_args = match reason {
        None => MacArgs::Empty,
        Some(lit) => MacArgs::Eq(span, TokenStream::new(vec![lit.token_tree().into()])),
    };

    // def-site context makes rustc accept the unstable `rustc_must_use` and `MustUse` trait.
    let def_span = ecx.with_def_site_ctxt(item.span());

    // Put a `#[rustc_must_use]` on the item in any case. This allows rustdoc to pick it up and
    // render it.
    item.visit_attrs(|attrs| {
        attrs.push(mk_attr(
            AttrStyle::Outer,
            ast::Path::from_ident(Ident::with_dummy_span(sym::rustc_must_use)),
            mac_args,
            def_span,
        ));
    });

    // This macro desugars `#[must_use]` on types to `impl`s of the `MustUse` trait.
    // Any other uses are forwarded to the `#[rustc_must_use]` macro.
    if let Annotatable::Item(ast_item) = &item {
        match &ast_item.kind {
            ast::ItemKind::Enum(_, generics)
            | ast::ItemKind::Struct(_, generics)
            | ast::ItemKind::Union(_, generics) => {
                // Generate a derive-style impl for a concrete type.
                let impl_items = if let Some(reason) = reason {
                    let item = ast::AssocItem {
                        attrs: vec![],
                        id: ast::DUMMY_NODE_ID,
                        span: def_span,
                        vis: ast::Visibility {
                            node: ast::VisibilityKind::Inherited,
                            span: def_span,
                        },
                        ident: Ident::new(sym::REASON, def_span),
                        kind: ast::AssocItemKind::Const(
                            ast::Defaultness::Final,
                            ecx.ty(
                                def_span,
                                ast::TyKind::Rptr(
                                    Some(ecx.lifetime(
                                        def_span,
                                        Ident::new(kw::StaticLifetime, def_span),
                                    )),
                                    ecx.ty_mt(
                                        ecx.ty_ident(def_span, Ident::new(sym::str, def_span)),
                                        ast::Mutability::Not,
                                    ),
                                ),
                            ),
                            Some(ecx.expr_lit(def_span, reason.kind.clone())),
                        ),
                        tokens: None,
                    };

                    vec![P(item)]
                } else {
                    vec![]
                };

                let mut impl_generics = generics.clone();
                for param in impl_generics.params.iter_mut() {
                    match &mut param.kind {
                        ast::GenericParamKind::Type { default } => {
                            // Delete defaults as they're not usable in impls.
                            *default = None;
                        }

                        ast::GenericParamKind::Lifetime
                        | ast::GenericParamKind::Const { ty: _ } => {}
                    }
                }
                let new_impl = ast::ItemKind::Impl {
                    unsafety: ast::Unsafe::No,
                    polarity: ast::ImplPolarity::Positive,
                    defaultness: ast::Defaultness::Final,
                    constness: ast::Const::No,
                    generics: impl_generics,
                    of_trait: Some(ecx.trait_ref(ecx.path(
                        def_span,
                        vec![
                            Ident::new(kw::DollarCrate, def_span),
                            Ident::new(sym::marker, def_span),
                            Ident::new(sym::MustUse, def_span),
                        ],
                    ))),
                    self_ty: ecx.ty_path(
                        ecx.path_all(
                            def_span,
                            false,
                            vec![ast_item.ident],
                            generics
                                .params
                                .iter()
                                .map(|param| match &param.kind {
                                    ast::GenericParamKind::Lifetime => ast::GenericArg::Lifetime(
                                        ecx.lifetime(def_span, param.ident),
                                    ),
                                    ast::GenericParamKind::Type { default: _ } => {
                                        ast::GenericArg::Type(ecx.ty_ident(def_span, param.ident))
                                    }
                                    ast::GenericParamKind::Const { ty: _ } => {
                                        ast::GenericArg::Const(
                                            ecx.const_ident(def_span, param.ident),
                                        )
                                    }
                                })
                                .collect(),
                        ),
                    ),
                    items: impl_items,
                };

                // Copy some important attributes from the original item, just like the built-in
                // derives.
                let attrs = ast_item
                    .attrs
                    .iter()
                    .filter(|a| {
                        [sym::allow, sym::warn, sym::deny, sym::forbid, sym::stable, sym::unstable]
                            .contains(&a.name_or_empty())
                    })
                    .cloned()
                    .chain(iter::once(
                        ecx.attribute(ecx.meta_word(def_span, sym::automatically_derived)),
                    ))
                    .collect();

                let new_impl =
                    Annotatable::Item(ecx.item(def_span, Ident::invalid(), attrs, new_impl));

                return vec![item, new_impl];
            }

            ast::ItemKind::Trait(_, _, _, _, _) => {
                // Generate a blanket `impl<T: Trait> MustUse for T` impl.
                // FIXME(jschievink): This currently doesn't work due to overlapping impls. Even
                // with specialization, you couldn't implement 2 `#[must_use]` traits for the same
                // type.
                // Technically, we could permit overlapping impls here, since it cannot lead to
                // unsoundness. At worst, the lint will only print one of the applicable messages
                // (though it could attempt to collect messages from all impls that are known to
                // apply).
            }

            ast::ItemKind::TraitAlias(_, _)
            | ast::ItemKind::Impl { .. }
            | ast::ItemKind::MacCall(..)
            | ast::ItemKind::MacroDef(..)
            | ast::ItemKind::ExternCrate(..)
            | ast::ItemKind::Use(..)
            | ast::ItemKind::Static(..)
            | ast::ItemKind::Const(..)
            | ast::ItemKind::Fn(..)
            | ast::ItemKind::Mod(..)
            | ast::ItemKind::ForeignMod(..)
            | ast::ItemKind::GlobalAsm(..)
            | ast::ItemKind::TyAlias(..) => {}
        }
    }

    vec![item]
}
