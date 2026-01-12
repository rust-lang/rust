use rustc_ast::token::{Delimiter, TokenKind};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{
    Attribute, DUMMY_NODE_ID, EiiExternTarget, EiiImpl, ItemKind, MetaItem, Path, Stmt, StmtKind,
    Visibility, ast,
};
use rustc_ast_pretty::pprust::path_to_string;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{ErrorGuaranteed, Ident, Span, kw, sym};
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
    eii_attr_span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
    impl_unsafe: bool,
) -> Vec<Annotatable> {
    let eii_attr_span = ecx.with_def_site_ctxt(eii_attr_span);

    let (item, wrap_item): (_, &dyn Fn(_) -> _) = if let Annotatable::Item(item) = item {
        (item, &Annotatable::Item)
    } else if let Annotatable::Stmt(ref stmt) = item
        && let StmtKind::Item(ref item) = stmt.kind
    {
        (item.clone(), &|item| {
            Annotatable::Stmt(Box::new(Stmt {
                id: DUMMY_NODE_ID,
                kind: StmtKind::Item(item),
                span: eii_attr_span,
            }))
        })
    } else {
        ecx.dcx().emit_err(EiiSharedMacroExpectedFunction {
            span: eii_attr_span,
            name: path_to_string(&meta_item.path),
        });
        return vec![item];
    };

    let ast::Item { attrs, id: _, span: _, vis, kind: ItemKind::Fn(func), tokens: _ } =
        item.as_ref()
    else {
        ecx.dcx().emit_err(EiiSharedMacroExpectedFunction {
            span: eii_attr_span,
            name: path_to_string(&meta_item.path),
        });
        return vec![wrap_item(item)];
    };
    // only clone what we need
    let attrs = attrs.clone();
    let func = (**func).clone();
    let vis = vis.clone();

    let attrs_from_decl =
        filter_attrs_for_multiple_eii_attr(ecx, attrs, eii_attr_span, &meta_item.path);

    let Ok(macro_name) = name_for_impl_macro(ecx, &func, &meta_item) else {
        return vec![wrap_item(item)];
    };

    // span of the declaring item without attributes
    let item_span = func.sig.span;
    let foreign_item_name = func.ident;

    let mut return_items = Vec::new();

    if func.body.is_some() {
        return_items.push(Box::new(generate_default_impl(
            ecx,
            &func,
            impl_unsafe,
            macro_name,
            eii_attr_span,
            item_span,
            foreign_item_name,
        )))
    }

    return_items.push(Box::new(generate_foreign_item(
        ecx,
        eii_attr_span,
        item_span,
        func,
        vis,
        &attrs_from_decl,
    )));
    return_items.push(Box::new(generate_attribute_macro_to_implement(
        ecx,
        eii_attr_span,
        macro_name,
        foreign_item_name,
        impl_unsafe,
        &attrs_from_decl,
    )));

    return_items.into_iter().map(wrap_item).collect()
}

/// Decide on the name of the macro that can be used to implement the EII.
/// This is either an explicitly given name, or the name of the item in the
/// declaration of the EII.
fn name_for_impl_macro(
    ecx: &mut ExtCtxt<'_>,
    func: &ast::Fn,
    meta_item: &MetaItem,
) -> Result<Ident, ErrorGuaranteed> {
    if meta_item.is_word() {
        Ok(func.ident)
    } else if let Some([first]) = meta_item.meta_item_list()
        && let Some(m) = first.meta_item()
        && m.path.segments.len() == 1
    {
        Ok(m.path.segments[0].ident)
    } else {
        Err(ecx.dcx().emit_err(EiiMacroExpectedMaxOneArgument {
            span: meta_item.span,
            name: path_to_string(&meta_item.path),
        }))
    }
}

/// Ensure that in the list of attrs, there's only a single `eii` attribute.
fn filter_attrs_for_multiple_eii_attr(
    ecx: &mut ExtCtxt<'_>,
    attrs: ThinVec<Attribute>,
    eii_attr_span: Span,
    eii_attr_path: &Path,
) -> ThinVec<Attribute> {
    attrs
        .into_iter()
        .filter(|i| {
            if i.has_name(sym::eii) {
                ecx.dcx().emit_err(EiiOnlyOnce {
                    span: i.span,
                    first_span: eii_attr_span,
                    name: path_to_string(eii_attr_path),
                });
                false
            } else {
                true
            }
        })
        .collect()
}

fn generate_default_impl(
    ecx: &mut ExtCtxt<'_>,
    func: &ast::Fn,
    impl_unsafe: bool,
    macro_name: Ident,
    eii_attr_span: Span,
    item_span: Span,
    foreign_item_name: Ident,
) -> ast::Item {
    // FIXME: re-add some original attrs
    let attrs = ThinVec::new();

    let mut default_func = func.clone();
    default_func.eii_impls.push(EiiImpl {
        node_id: DUMMY_NODE_ID,
        inner_span: macro_name.span,
        eii_macro_path: ast::Path::from_ident(macro_name),
        impl_safety: if impl_unsafe {
            ast::Safety::Unsafe(eii_attr_span)
        } else {
            ast::Safety::Default
        },
        span: eii_attr_span,
        is_default: true,
        known_eii_macro_resolution: Some(ast::EiiExternTarget {
            extern_item_path: ast::Path {
                span: foreign_item_name.span,
                segments: thin_vec![
                    ast::PathSegment {
                        ident: Ident::from_str_and_span("super", foreign_item_name.span,),
                        id: DUMMY_NODE_ID,
                        args: None
                    },
                    ast::PathSegment { ident: foreign_item_name, id: DUMMY_NODE_ID, args: None },
                ],
                tokens: None,
            },
            impl_unsafe,
        }),
    });

    ast::Item {
        attrs: ThinVec::new(),
        id: ast::DUMMY_NODE_ID,
        span: eii_attr_span,
        vis: ast::Visibility {
            span: eii_attr_span,
            kind: ast::VisibilityKind::Inherited,
            tokens: None,
        },
        kind: ast::ItemKind::Const(Box::new(ast::ConstItem {
            ident: Ident { name: kw::Underscore, span: eii_attr_span },
            defaultness: ast::Defaultness::Final,
            generics: ast::Generics::default(),
            ty: Box::new(ast::Ty {
                id: DUMMY_NODE_ID,
                kind: ast::TyKind::Tup(ThinVec::new()),
                span: eii_attr_span,
                tokens: None,
            }),
            rhs: Some(ast::ConstItemRhs::Body(Box::new(ast::Expr {
                id: DUMMY_NODE_ID,
                kind: ast::ExprKind::Block(
                    Box::new(ast::Block {
                        stmts: thin_vec![ast::Stmt {
                            id: DUMMY_NODE_ID,
                            kind: ast::StmtKind::Item(Box::new(ast::Item {
                                attrs: ThinVec::new(),
                                id: DUMMY_NODE_ID,
                                span: item_span,
                                vis: ast::Visibility {
                                    span: item_span,
                                    kind: ast::VisibilityKind::Inherited,
                                    tokens: None
                                },
                                kind: ItemKind::Mod(
                                    ast::Safety::Default,
                                    Ident::from_str_and_span("dflt", item_span),
                                    ast::ModKind::Loaded(
                                        thin_vec![
                                            Box::new(ast::Item {
                                                attrs: thin_vec![ecx.attr_nested_word(
                                                    sym::allow,
                                                    sym::unused_imports,
                                                    item_span
                                                ),],
                                                id: DUMMY_NODE_ID,
                                                span: item_span,
                                                vis: ast::Visibility {
                                                    span: eii_attr_span,
                                                    kind: ast::VisibilityKind::Inherited,
                                                    tokens: None
                                                },
                                                kind: ItemKind::Use(ast::UseTree {
                                                    prefix: ast::Path::from_ident(
                                                        Ident::from_str_and_span(
                                                            "super", item_span,
                                                        )
                                                    ),
                                                    kind: ast::UseTreeKind::Glob,
                                                    span: item_span,
                                                }),
                                                tokens: None,
                                            }),
                                            Box::new(ast::Item {
                                                attrs,
                                                id: DUMMY_NODE_ID,
                                                span: item_span,
                                                vis: ast::Visibility {
                                                    span: eii_attr_span,
                                                    kind: ast::VisibilityKind::Inherited,
                                                    tokens: None
                                                },
                                                kind: ItemKind::Fn(Box::new(default_func)),
                                                tokens: None,
                                            }),
                                        ],
                                        ast::Inline::Yes,
                                        ast::ModSpans {
                                            inner_span: item_span,
                                            inject_use_span: item_span,
                                        }
                                    )
                                ),
                                tokens: None,
                            })),
                            span: eii_attr_span,
                        }],
                        id: DUMMY_NODE_ID,
                        rules: ast::BlockCheckMode::Default,
                        span: eii_attr_span,
                        tokens: None,
                    }),
                    None,
                ),
                span: eii_attr_span,
                attrs: ThinVec::new(),
                tokens: None,
            }))),
            define_opaque: None,
        })),
        tokens: None,
    }
}

/// Generates a foreign item, like
///
/// ```rust, ignore
/// extern "…" { safe fn item(); }
/// ```
fn generate_foreign_item(
    ecx: &mut ExtCtxt<'_>,
    eii_attr_span: Span,
    item_span: Span,
    mut func: ast::Fn,
    vis: Visibility,
    attrs_from_decl: &[Attribute],
) -> ast::Item {
    let mut foreign_item_attrs = ThinVec::new();
    foreign_item_attrs.extend_from_slice(attrs_from_decl);

    // Add the rustc_eii_extern_item on the foreign item. Usually, foreign items are mangled.
    // This attribute makes sure that we later know that this foreign item's symbol should not be.
    foreign_item_attrs.push(ecx.attr_word(sym::rustc_eii_extern_item, eii_attr_span));

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
            span: eii_attr_span,
        }),
    };

    // ABI has been moved to the extern {} block, so we remove it from the fn item.
    func.sig.header.ext = ast::Extern::None;
    func.body = None;

    // And mark safe functions explicitly as `safe fn`.
    if func.sig.header.safety == ast::Safety::Default {
        func.sig.header.safety = ast::Safety::Safe(func.sig.span);
    }

    ast::Item {
        attrs: ast::AttrVec::default(),
        id: ast::DUMMY_NODE_ID,
        span: eii_attr_span,
        vis: ast::Visibility {
            span: eii_attr_span,
            kind: ast::VisibilityKind::Inherited,
            tokens: None,
        },
        kind: ast::ItemKind::ForeignMod(ast::ForeignMod {
            extern_span: eii_attr_span,
            safety: ast::Safety::Unsafe(eii_attr_span),
            abi,
            items: From::from([Box::new(ast::ForeignItem {
                attrs: foreign_item_attrs,
                id: ast::DUMMY_NODE_ID,
                span: item_span,
                vis,
                kind: ast::ForeignItemKind::Fn(Box::new(func.clone())),
                tokens: None,
            })]),
        }),
        tokens: None,
    }
}

/// Generate a stub macro (a bit like in core) that will roughly look like:
///
/// ```rust, ignore, example
/// // Since this a stub macro, the actual code that expands it lives in the compiler.
/// // This attribute tells the compiler that
/// #[builtin_macro(eii_shared_macro)]
/// // the metadata to link this macro to the generated foreign item.
/// #[eii_extern_target(<related_reign_item>)]
/// macro macro_name { () => {} }
/// ```
fn generate_attribute_macro_to_implement(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    macro_name: Ident,
    foreign_item_name: Ident,
    impl_unsafe: bool,
    attrs_from_decl: &[Attribute],
) -> ast::Item {
    let mut macro_attrs = ThinVec::new();

    // To avoid e.g. `error: attribute macro has missing stability attribute`
    // errors for eii's in std.
    macro_attrs.extend_from_slice(attrs_from_decl);

    // #[builtin_macro(eii_shared_macro)]
    macro_attrs.push(ecx.attr_nested_word(sym::rustc_builtin_macro, sym::eii_shared_macro, span));

    ast::Item {
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
                // #[eii_extern_target(foreign_item_ident)]
                eii_extern_target: Some(ast::EiiExternTarget {
                    extern_item_path: ast::Path::from_ident(foreign_item_name),
                    impl_unsafe,
                }),
            },
        ),
        tokens: None,
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

    d.eii_extern_target = Some(EiiExternTarget { extern_item_path, impl_unsafe });

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
        inner_span: meta_item.path.span,
        eii_macro_path: meta_item.path.clone(),
        impl_safety: meta_item.unsafety,
        span,
        is_default,
        known_eii_macro_resolution: None,
    });

    vec![item]
}
