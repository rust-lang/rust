use rustc_ast::token::{Delimiter, TokenKind};
use rustc_ast::tokenstream::{DelimSpacing, DelimSpan, Spacing, TokenStream, TokenTree};
use rustc_ast::{
    Attribute, DUMMY_NODE_ID, EiiDecl, EiiImpl, ItemKind, MetaItem, Mutability, Path, StmtKind,
    Visibility, ast,
};
use rustc_ast_pretty::pprust::path_to_string;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{ErrorGuaranteed, Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::diagnostics::{
    EiiAttributeNotSupported, EiiBothDeclAndImpl, EiiExternTargetExpectedList,
    EiiExternTargetExpectedMacro, EiiExternTargetExpectedUnsafe, EiiMacroExpectedMaxOneArgument,
    EiiMultipleImplementations, EiiOnlyOnce, EiiSharedMacroInStatementPosition,
    EiiSharedMacroTarget, EiiStaticArgumentRequired, EiiStaticDefaultApple, EiiStaticMutable,
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
/// #[eii_declaration(panic_handler)]
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
    orig_item: Annotatable,
    impl_unsafe: bool,
) -> Vec<Annotatable> {
    let eii_attr_span = ecx.with_def_site_ctxt(eii_attr_span);

    let item = if let Annotatable::Item(item) = orig_item {
        item
    } else if let Annotatable::Stmt(ref stmt) = orig_item
        && let StmtKind::Item(ref item) = stmt.kind
        && let ItemKind::Fn(ref f) = item.kind
    {
        ecx.dcx().emit_err(EiiSharedMacroInStatementPosition {
            span: eii_attr_span.to(item.span),
            name: path_to_string(&meta_item.path),
            item_span: f.ident.span,
        });
        return vec![orig_item];
    } else {
        ecx.dcx().emit_err(EiiSharedMacroTarget {
            span: eii_attr_span,
            name: path_to_string(&meta_item.path),
        });
        return vec![orig_item];
    };

    let ast::Item { attrs, id: _, span: _, vis, kind, tokens: _ } = item.as_ref();
    let (item_span, foreign_item_name) = match kind {
        ItemKind::Fn(func) => (func.sig.span, func.ident),
        ItemKind::Static(stat) => {
            // See https://github.com/rust-lang/rust/issues/157649
            if let Some(expr) = &stat.expr
                && ecx.sess.target.is_like_darwin
            {
                ecx.dcx().emit_err(EiiStaticDefaultApple {
                    span: expr.span,
                    name: path_to_string(&meta_item.path),
                });
                return vec![];
            }

            // Statics must have an explicit name for the eii
            if meta_item.is_word() {
                ecx.dcx().emit_err(EiiStaticArgumentRequired {
                    span: eii_attr_span,
                    name: path_to_string(&meta_item.path),
                });
                return vec![];
            }

            // Mut statics are currently not supported
            if stat.mutability == Mutability::Mut {
                ecx.dcx().emit_err(EiiStaticMutable {
                    span: eii_attr_span,
                    name: path_to_string(&meta_item.path),
                });
            }

            (item.span, stat.ident)
        }
        _ => {
            ecx.dcx().emit_err(EiiSharedMacroTarget {
                span: eii_attr_span,
                name: path_to_string(&meta_item.path),
            });
            return vec![Annotatable::Item(item)];
        }
    };

    match kind {
        ItemKind::Fn(func) => {
            if func.eii_impl.is_some() {
                ecx.dcx().emit_err(EiiBothDeclAndImpl { span: eii_attr_span });
                return vec![Annotatable::Item(item)];
            }
        }
        ItemKind::Static(stat) => {
            if stat.eii_impl.is_some() {
                ecx.dcx().emit_err(EiiBothDeclAndImpl { span: eii_attr_span });
                return vec![Annotatable::Item(item)];
            }
        }
        _ => unreachable!("Target was checked earlier"),
    };

    // only clone what we need
    let attrs = attrs.clone();
    let vis = vis.clone();

    let attrs_from_decl =
        filter_attrs_for_multiple_eii_attr(ecx, attrs, eii_attr_span, &meta_item.path);
    let (macro_attrs, foreign_item_attrs, default_func_attrs) =
        split_attrs(ecx, item_span, attrs_from_decl);

    let Ok(macro_name) = name_for_impl_macro(ecx, foreign_item_name, &meta_item) else {
        // we don't need to wrap in Annotatable::Stmt conditionally since
        // EII can't be used on items in statement position
        return vec![Annotatable::Item(item)];
    };

    let mut module_items = Vec::new();

    if let Some(default_impl) = generate_default_impl(
        ecx,
        kind,
        impl_unsafe,
        macro_name,
        eii_attr_span,
        item_span,
        foreign_item_name,
        default_func_attrs,
    ) {
        module_items.push(default_impl);
    }

    module_items.push(generate_foreign_item(
        ecx,
        eii_attr_span,
        item_span,
        kind,
        vis,
        foreign_item_attrs,
    ));
    module_items.push(generate_attribute_macro_to_implement(
        ecx,
        eii_attr_span,
        macro_name,
        foreign_item_name,
        impl_unsafe,
        macro_attrs,
    ));

    // we don't need to wrap in Annotatable::Stmt conditionally since
    // EII can't be used on items in statement position
    module_items.into_iter().map(Annotatable::Item).collect()
}

fn split_attrs(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    attrs: ThinVec<Attribute>,
) -> (ThinVec<Attribute>, ThinVec<Attribute>, ThinVec<Attribute>) {
    let mut macro_attributes = ThinVec::new();
    let mut foreign_item_attributes = ThinVec::new();
    let mut default_attributes = ThinVec::new();

    for attr in attrs {
        match attr.name() {
            // Inline only matters for the default function being inlined into callsites
            Some(sym::inline) => default_attributes.push(attr),
            // If an eii is marked a lang item, that's because we want to call its declaration, so
            // mark the foreign item as the lang item
            Some(sym::lang) => foreign_item_attributes.push(attr),
            // Deprecating an eii means deprecating the macro and the foreign item
            Some(sym::deprecated) => {
                foreign_item_attributes.push(attr.clone());
                macro_attributes.push(attr);
            }
            // The stability of an EII affects the usage of the macro and calling the foreign item
            Some(sym::stable) | Some(sym::unstable) => {
                foreign_item_attributes.push(attr.clone());
                macro_attributes.push(attr);
            }
            // `#[track_caller]` goes on the foreign item only: it's the symbol callers link
            // against, so it must carry the flag for call sites to pass the caller location.
            // Implementations derive it during codegen (see `EiiImpls` in `codegen_attrs.rs`),
            // so it must not be routed onto the default impl here.
            Some(sym::track_caller) => {
                foreign_item_attributes.push(attr);
            }
            // Doc attributes should be forwarded to the macro and the foreign item, since those are
            // the two items you interact with as a user.
            // FIXME: idk yet how EIIs show up in docs, might want to customize
            _ if attr.is_doc_comment() => {
                foreign_item_attributes.push(attr.clone());
                macro_attributes.push(attr);
            }
            Some(sym::eii) => unreachable!("should already be filtered out"),
            _ => {
                ecx.dcx().emit_err(EiiAttributeNotSupported { span, attr_span: attr.span() });
            }
        }
    }

    (macro_attributes, foreign_item_attributes, default_attributes)
}

/// Decide on the name of the macro that can be used to implement the EII.
/// This is either an explicitly given name, or the name of the item in the
/// declaration of the EII.
fn name_for_impl_macro(
    ecx: &mut ExtCtxt<'_>,
    item_ident: Ident,
    meta_item: &MetaItem,
) -> Result<Ident, ErrorGuaranteed> {
    if meta_item.is_word() {
        Ok(item_ident)
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
    item_kind: &ItemKind,
    impl_unsafe: bool,
    macro_name: Ident,
    eii_attr_span: Span,
    item_span: Span,
    foreign_item_name: Ident,
    attrs: ThinVec<Attribute>,
) -> Option<Box<ast::Item>> {
    match item_kind {
        ItemKind::Fn(func) => {
            if func.body.is_none() {
                return None;
            }
        }
        ItemKind::Static(stat) => {
            if stat.expr.is_none() {
                return None;
            }
        }
        _ => unreachable!("Target was checked earlier"),
    };

    let eii_impl = Box::new(EiiImpl {
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
        known_eii_macro_resolution: Some(ecx.path(
            foreign_item_name.span,
            // prefix self to explicitly escape the const block generated below
            // NOTE: this is why EIIs can't be used on statements
            vec![Ident::from_str_and_span("self", foreign_item_name.span), foreign_item_name],
        )),
    });

    let mut item_kind = item_kind.clone();
    match &mut item_kind {
        ItemKind::Fn(func) => {
            assert!(func.eii_impl.is_none());
            func.eii_impl = Some(eii_impl);
        }
        ItemKind::Static(stat) => {
            assert!(stat.eii_impl.is_none());
            stat.eii_impl = Some(eii_impl);
        }
        _ => unreachable!("Target was checked earlier"),
    };

    let anon_mod = |span: Span, stmts: ThinVec<ast::Stmt>| {
        let unit = ecx.ty(item_span, ast::TyKind::Tup(ThinVec::new()));
        let underscore = Ident::new(kw::Underscore, item_span);
        ecx.item_const(
            span,
            underscore,
            unit,
            Some(ecx.expr_block(ecx.block(span, stmts))),
            ast::ConstItemKind::Body,
        )
    };

    // const _: () = {
    //     <orig item>
    // }
    Some(anon_mod(
        item_span,
        thin_vec![ecx.stmt_item(item_span, ecx.item(item_span, attrs, item_kind))],
    ))
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
    item_kind: &ItemKind,
    vis: Visibility,
    attrs_from_decl: ThinVec<Attribute>,
) -> Box<ast::Item> {
    let mut foreign_item_attrs = attrs_from_decl;

    // Add the rustc_eii_foreign_item on the foreign item. Usually, foreign items are mangled.
    // This attribute makes sure that we later know that this foreign item's symbol should not be.
    foreign_item_attrs.push(ecx.attr_word(sym::rustc_eii_foreign_item, eii_attr_span));

    // We set the abi to the default "rust" abi, which can be overridden by `generate_foreign_func`,
    // if a specific abi was specified on the EII function
    let mut abi = Some(ast::StrLit {
        symbol: sym::Rust,
        suffix: None,
        symbol_unescaped: sym::Rust,
        style: ast::StrStyle::Cooked,
        span: eii_attr_span,
    });
    let foreign_kind = match item_kind {
        ItemKind::Fn(func) => generate_foreign_func(func.clone(), &mut abi),
        ItemKind::Static(stat) => generate_foreign_static(stat.clone()),
        _ => unreachable!("Target was checked earlier"),
    };

    ecx.item(
        eii_attr_span,
        ThinVec::new(),
        ast::ItemKind::ForeignMod(ast::ForeignMod {
            extern_span: eii_attr_span,
            safety: ast::Safety::Unsafe(eii_attr_span),
            abi,
            items: From::from([Box::new(ast::ForeignItem {
                attrs: foreign_item_attrs,
                id: ast::DUMMY_NODE_ID,
                span: item_span,
                vis,
                kind: foreign_kind,
                tokens: None,
            })]),
        }),
    )
}

fn generate_foreign_func(
    mut func: Box<ast::Fn>,
    abi: &mut Option<ast::StrLit>,
) -> ast::ForeignItemKind {
    match func.sig.header.ext {
        // extern "X" fn  =>  extern "X" {}
        ast::Extern::Explicit(lit, _) => *abi = Some(lit),
        // extern fn  =>  extern {}
        ast::Extern::Implicit(_) => *abi = None,
        // no abi was specified, so we keep the default
        ast::Extern::None => {}
    };

    // ABI has been moved to the extern {} block, so we remove it from the fn item.
    func.sig.header.ext = ast::Extern::None;
    func.body = None;

    // And mark safe functions explicitly as `safe fn`.
    if func.sig.header.safety == ast::Safety::Default {
        func.sig.header.safety = ast::Safety::Safe(func.sig.span);
    }

    ast::ForeignItemKind::Fn(func)
}

fn generate_foreign_static(mut stat: Box<ast::StaticItem>) -> ast::ForeignItemKind {
    if stat.safety == ast::Safety::Default {
        stat.safety = ast::Safety::Safe(stat.ident.span);
    }

    stat.expr = None;

    ast::ForeignItemKind::Static(stat)
}

/// Generate a stub macro (a bit like in core) that will roughly look like:
///
/// ```rust, ignore, example
/// // Since this a stub macro, the actual code that expands it lives in the compiler.
/// // This attribute tells the compiler that
/// #[builtin_macro(eii_shared_macro)]
/// // the metadata to link this macro to the generated foreign item.
/// #[eii_declaration(<related_foreign_item>)]
/// macro macro_name { () => {} }
/// ```
fn generate_attribute_macro_to_implement(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    macro_name: Ident,
    foreign_item_name: Ident,
    impl_unsafe: bool,
    attrs_from_decl: ThinVec<Attribute>,
) -> Box<ast::Item> {
    let mut macro_attrs = attrs_from_decl;

    // Avoid "missing stability attribute" errors for eiis in std. See #146993.
    macro_attrs.push(ecx.attr_name_value_str(sym::rustc_macro_transparency, sym::semiopaque, span));

    // #[builtin_macro(eii_shared_macro)]
    macro_attrs.push(ecx.attr_nested_word(sym::rustc_builtin_macro, sym::eii_shared_macro, span));

    // cant use ecx methods here to construct item since we need it to be public
    Box::new(ast::Item {
        attrs: macro_attrs,
        id: ast::DUMMY_NODE_ID,
        span,
        // pub
        vis: ast::Visibility { span, kind: ast::VisibilityKind::Public },
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
                // #[eii_declaration(foreign_item_ident)]
                eii_declaration: Some(ast::EiiDecl {
                    foreign_item: ast::Path::from_ident(foreign_item_name),
                    impl_unsafe,
                }),
            },
        ),
        tokens: None,
    })
}

pub(crate) fn eii_declaration(
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

    d.eii_declaration = Some(EiiDecl { foreign_item: extern_item_path, impl_unsafe });

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
        ecx.dcx().emit_err(EiiSharedMacroTarget { span, name: path_to_string(&meta_item.path) });
        return vec![item];
    };

    let eii_impl = match &mut i.kind {
        ItemKind::Fn(func) => &mut func.eii_impl,
        ItemKind::Static(stat) => &mut stat.eii_impl,
        _ => {
            ecx.dcx()
                .emit_err(EiiSharedMacroTarget { span, name: path_to_string(&meta_item.path) });
            return vec![item];
        }
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

    if eii_impl.is_some() {
        ecx.dcx().emit_err(EiiMultipleImplementations { span });
    }
    *eii_impl = Some(Box::new(EiiImpl {
        node_id: DUMMY_NODE_ID,
        inner_span: meta_item.path.span,
        eii_macro_path: meta_item.path.clone(),
        impl_safety: meta_item.unsafety,
        span,
        is_default,
        known_eii_macro_resolution: None,
    }));

    vec![item]
}
