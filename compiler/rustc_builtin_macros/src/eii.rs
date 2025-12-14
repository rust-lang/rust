use rustc_ast::token::{Delimiter, TokenKind};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{
    DUMMY_NODE_ID, EiiExternTarget, EiiImpl, ItemKind, Stmt, StmtKind, ast, token, tokenstream,
};
use rustc_ast_pretty::pprust::path_to_string;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::errors::{
    EiiExternTargetExpectedList, EiiExternTargetExpectedMacro, EiiExternTargetExpectedUnsafe,
    EiiMacroExpectedMaxOneArgument, EiiOnlyOnce, EiiSharedMacroExpectedFunction,
};

/// ```rust
/// #[eii]
/// fn panic_handler();
///
/// // or:
///
/// #[eii(panic_handler)]
/// fn panic_handler();
///
/// // expansion:
///
/// extern "Rust" {
///     fn panic_handler();
/// }
///
/// #[rustc_builtin_macro(eii_shared_macro)]
/// #[eii_extern_target(panic_handler)]
/// macro panic_handler() {}
/// ```
pub(crate) fn eii(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    eii_(ecx, span, meta_item, item, false)
}

pub(crate) fn unsafe_eii(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    eii_(ecx, span, meta_item, item, true)
}

fn eii_(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
    impl_unsafe: bool,
) -> Vec<Annotatable> {
    let span = ecx.with_def_site_ctxt(span);

    let (item, stmt) = if let Annotatable::Item(item) = item {
        (item, false)
    } else if let Annotatable::Stmt(ref stmt) = item
        && let StmtKind::Item(ref item) = stmt.kind
    {
        (item.clone(), true)
    } else {
        ecx.dcx().emit_err(EiiSharedMacroExpectedFunction {
            span,
            name: path_to_string(&meta_item.path),
        });
        return vec![item];
    };

    let orig_item = item.clone();

    let item = *item;

    let ast::Item { attrs, id: _, span: item_span, vis, kind: ItemKind::Fn(mut func), tokens: _ } =
        item
    else {
        ecx.dcx().emit_err(EiiSharedMacroExpectedFunction {
            span,
            name: path_to_string(&meta_item.path),
        });
        return vec![Annotatable::Item(Box::new(item))];
    };

    // Detect when this is the *second* eii attribute on an item.
    let mut new_attrs = ThinVec::new();
    for i in attrs {
        if i.has_name(sym::eii) {
            ecx.dcx().emit_err(EiiOnlyOnce {
                span: i.span,
                first_span: span,
                name: path_to_string(&meta_item.path),
            });
        } else {
            new_attrs.push(i);
        }
    }
    let attrs = new_attrs;

    let macro_name = if meta_item.is_word() {
        func.ident
    } else if let Some([first]) = meta_item.meta_item_list()
        && let Some(m) = first.meta_item()
        && m.path.segments.len() == 1
    {
        m.path.segments[0].ident
    } else {
        ecx.dcx().emit_err(EiiMacroExpectedMaxOneArgument {
            span: meta_item.span,
            name: path_to_string(&meta_item.path),
        });
        return vec![Annotatable::Item(orig_item)];
    };

    let mut return_items = Vec::new();

    if func.body.is_some() {
        let mut default_func = func.clone();
        func.body = None;
        default_func.eii_impls.push(ast::EiiImpl {
            node_id: DUMMY_NODE_ID,
            eii_macro_path: ast::Path::from_ident(macro_name),
            impl_safety: if impl_unsafe { ast::Safety::Unsafe(span) } else { ast::Safety::Default },
            span,
            inner_span: macro_name.span,
            is_default: true, // important!
        });

        return_items.push(Box::new(ast::Item {
            attrs: ThinVec::new(),
            id: ast::DUMMY_NODE_ID,
            span,
            vis: ast::Visibility { span, kind: ast::VisibilityKind::Inherited, tokens: None },
            kind: ast::ItemKind::Const(Box::new(ast::ConstItem {
                ident: Ident { name: kw::Underscore, span },
                defaultness: ast::Defaultness::Final,
                generics: ast::Generics::default(),
                ty: Box::new(ast::Ty {
                    id: DUMMY_NODE_ID,
                    kind: ast::TyKind::Tup(ThinVec::new()),
                    span,
                    tokens: None,
                }),
                rhs: Some(ast::ConstItemRhs::Body(Box::new(ast::Expr {
                    id: DUMMY_NODE_ID,
                    kind: ast::ExprKind::Block(
                        Box::new(ast::Block {
                            stmts: thin_vec![ast::Stmt {
                                id: DUMMY_NODE_ID,
                                kind: ast::StmtKind::Item(Box::new(ast::Item {
                                    attrs: thin_vec![], // FIXME: re-add some original attrs
                                    id: DUMMY_NODE_ID,
                                    span: item_span,
                                    vis: ast::Visibility {
                                        span,
                                        kind: ast::VisibilityKind::Inherited,
                                        tokens: None
                                    },
                                    kind: ItemKind::Fn(default_func),
                                    tokens: None,
                                })),
                                span
                            }],
                            id: DUMMY_NODE_ID,
                            rules: ast::BlockCheckMode::Default,
                            span,
                            tokens: None,
                        }),
                        None,
                    ),
                    span,
                    attrs: ThinVec::new(),
                    tokens: None,
                }))),
                define_opaque: None,
            })),
            tokens: None,
        }))
    }

    let decl_span = span.to(func.sig.span);

    let abi = match func.sig.header.ext {
        // extern "X" fn  =>  extern "X" {}
        ast::Extern::Explicit(lit, _) => Some(lit),
        // extern fn  =>  extern {}
        ast::Extern::Implicit(_) => None,
        // fn  =>  extern "Rust" {}
        ast::Extern::None => Some(ast::StrLit {
            symbol: sym::Rust,
            suffix: None,
            symbol_unescaped: sym::Rust,
            style: ast::StrStyle::Cooked,
            span,
        }),
    };

    // ABI has been moved to the extern {} block, so we remove it from the fn item.
    func.sig.header.ext = ast::Extern::None;

    // And mark safe functions explicitly as `safe fn`.
    if func.sig.header.safety == ast::Safety::Default {
        func.sig.header.safety = ast::Safety::Safe(func.sig.span);
    }

    // extern "…" { safe fn item(); }
    let mut extern_item_attrs = attrs.clone();
    extern_item_attrs.push(ast::Attribute {
        kind: ast::AttrKind::Normal(Box::new(ast::NormalAttr {
            item: ast::AttrItem {
                unsafety: ast::Safety::Default,
                // Add the rustc_eii_extern_item on the foreign item. Usually, foreign items are mangled.
                // This attribute makes sure that we later know that this foreign item's symbol should not be.
                path: ast::Path::from_ident(Ident::new(sym::rustc_eii_extern_item, span)),
                args: ast::AttrArgs::Empty,
                tokens: None,
            },
            tokens: None,
        })),
        id: ecx.sess.psess.attr_id_generator.mk_attr_id(),
        style: ast::AttrStyle::Outer,
        span,
    });

    let extern_block = Box::new(ast::Item {
        attrs: ast::AttrVec::default(),
        id: ast::DUMMY_NODE_ID,
        span,
        vis: ast::Visibility { span, kind: ast::VisibilityKind::Inherited, tokens: None },
        kind: ast::ItemKind::ForeignMod(ast::ForeignMod {
            extern_span: span,
            safety: ast::Safety::Unsafe(span),
            abi,
            items: From::from([Box::new(ast::ForeignItem {
                attrs: extern_item_attrs,
                id: ast::DUMMY_NODE_ID,
                span: item_span,
                vis,
                kind: ast::ForeignItemKind::Fn(func.clone()),
                tokens: None,
            })]),
        }),
        tokens: None,
    });

    let mut macro_attrs = attrs.clone();
    macro_attrs.push(
        // #[builtin_macro(eii_shared_macro)]
        ast::Attribute {
            kind: ast::AttrKind::Normal(Box::new(ast::NormalAttr {
                item: ast::AttrItem {
                    unsafety: ast::Safety::Default,
                    path: ast::Path::from_ident(Ident::new(sym::rustc_builtin_macro, span)),
                    args: ast::AttrArgs::Delimited(ast::DelimArgs {
                        dspan: DelimSpan::from_single(span),
                        delim: Delimiter::Parenthesis,
                        tokens: TokenStream::new(vec![tokenstream::TokenTree::token_alone(
                            token::TokenKind::Ident(sym::eii_shared_macro, token::IdentIsRaw::No),
                            span,
                        )]),
                    }),
                    tokens: None,
                },
                tokens: None,
            })),
            id: ecx.sess.psess.attr_id_generator.mk_attr_id(),
            style: ast::AttrStyle::Outer,
            span,
        },
    );

    let macro_def = Box::new(ast::Item {
        attrs: macro_attrs,
        id: ast::DUMMY_NODE_ID,
        span,
        // pub
        vis: ast::Visibility { span, kind: ast::VisibilityKind::Public, tokens: None },
        kind: ast::ItemKind::MacroDef(
            // macro macro_name
            macro_name,
            ast::MacroDef {
                // { () => {} }
                body: Box::new(ast::DelimArgs {
                    dspan: DelimSpan::from_single(span),
                    delim: Delimiter::Brace,
                    tokens: TokenStream::from_iter([
                        TokenTree::Delimited(
                            DelimSpan::from_single(span),
                            DelimSpacing::new(Spacing::Alone, Spacing::Alone),
                            Delimiter::Parenthesis,
                            TokenStream::default(),
                        ),
                        TokenTree::token_alone(TokenKind::FatArrow, span),
                        TokenTree::Delimited(
                            DelimSpan::from_single(span),
                            DelimSpacing::new(Spacing::Alone, Spacing::Alone),
                            Delimiter::Brace,
                            TokenStream::default(),
                        ),
                    ]),
                }),
                macro_rules: false,
                // #[eii_extern_target(func.ident)]
                eii_extern_target: Some(ast::EiiExternTarget {
                    extern_item_path: ast::Path::from_ident(func.ident),
                    impl_unsafe,
                    span: decl_span,
                }),
            },
        ),
        tokens: None,
    });

    return_items.push(extern_block);
    return_items.push(macro_def);

    if stmt {
        return_items
            .into_iter()
            .map(|i| {
                Annotatable::Stmt(Box::new(Stmt {
                    id: DUMMY_NODE_ID,
                    kind: StmtKind::Item(i),
                    span,
                }))
            })
            .collect()
    } else {
        return_items.into_iter().map(|i| Annotatable::Item(i)).collect()
    }
}

pub(crate) fn eii_extern_target(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let i = if let Annotatable::Item(ref mut item) = item {
        item
    } else if let Annotatable::Stmt(ref mut stmt) = item
        && let StmtKind::Item(ref mut item) = stmt.kind
    {
        item
    } else {
        ecx.dcx().emit_err(EiiExternTargetExpectedMacro { span });
        return vec![item];
    };

    let ItemKind::MacroDef(_, d) = &mut i.kind else {
        ecx.dcx().emit_err(EiiExternTargetExpectedMacro { span });
        return vec![item];
    };

    let Some(list) = meta_item.meta_item_list() else {
        ecx.dcx().emit_err(EiiExternTargetExpectedList { span: meta_item.span });
        return vec![item];
    };

    if list.len() > 2 {
        ecx.dcx().emit_err(EiiExternTargetExpectedList { span: meta_item.span });
        return vec![item];
    }

    let Some(extern_item_path) = list.get(0).and_then(|i| i.meta_item()).map(|i| i.path.clone())
    else {
        ecx.dcx().emit_err(EiiExternTargetExpectedList { span: meta_item.span });
        return vec![item];
    };

    let impl_unsafe = if let Some(i) = list.get(1) {
        if i.lit().and_then(|i| i.kind.str()).is_some_and(|i| i == kw::Unsafe) {
            true
        } else {
            ecx.dcx().emit_err(EiiExternTargetExpectedUnsafe { span: i.span() });
            return vec![item];
        }
    } else {
        false
    };

    d.eii_extern_target = Some(EiiExternTarget { extern_item_path, impl_unsafe, span });

    // Return the original item and the new methods.
    vec![item]
}

/// all Eiis share this function as the implementation for their attribute.
pub(crate) fn eii_shared_macro(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let i = if let Annotatable::Item(ref mut item) = item {
        item
    } else if let Annotatable::Stmt(ref mut stmt) = item
        && let StmtKind::Item(ref mut item) = stmt.kind
    {
        item
    } else {
        ecx.dcx().emit_err(EiiSharedMacroExpectedFunction {
            span,
            name: path_to_string(&meta_item.path),
        });
        return vec![item];
    };

    let ItemKind::Fn(f) = &mut i.kind else {
        ecx.dcx().emit_err(EiiSharedMacroExpectedFunction {
            span,
            name: path_to_string(&meta_item.path),
        });
        return vec![item];
    };

    let is_default = if meta_item.is_word() {
        false
    } else if let Some([first]) = meta_item.meta_item_list()
        && let Some(m) = first.meta_item()
        && m.path.segments.len() == 1
    {
        m.path.segments[0].ident.name == kw::Default
    } else {
        ecx.dcx().emit_err(EiiMacroExpectedMaxOneArgument {
            span: meta_item.span,
            name: path_to_string(&meta_item.path),
        });
        return vec![item];
    };

    f.eii_impls.push(EiiImpl {
        node_id: DUMMY_NODE_ID,
        eii_macro_path: meta_item.path.clone(),
        impl_safety: meta_item.unsafety,
        span,
        inner_span: meta_item.path.span,
        is_default,
    });

    vec![item]
}
