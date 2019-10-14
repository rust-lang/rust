//! Each macro must have a definition, so `#[plugin]` attributes
//! inject a dummy `macro_rules` item for each macro they define.

use syntax::ast::*;
use syntax::attr;
use syntax::edition::Edition;
use syntax::ext::base::{Resolver, NamedSyntaxExtension};
use syntax::parse::token;
use syntax::ptr::P;
use syntax::source_map::respan;
use syntax::symbol::sym;
use syntax::tokenstream::*;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::hygiene::{ExpnData, ExpnKind, AstPass};

use std::mem;

fn plugin_macro_def(name: Name, span: Span) -> P<Item> {
    let rustc_builtin_macro = attr::mk_attr_outer(
        attr::mk_word_item(Ident::new(sym::rustc_builtin_macro, span)));

    let parens: TreeAndJoint = TokenTree::Delimited(
        DelimSpan::from_single(span), token::Paren, TokenStream::default()
    ).into();
    let trees = vec![parens.clone(), TokenTree::token(token::FatArrow, span).into(), parens];

    P(Item {
        ident: Ident::new(name, span),
        attrs: vec![rustc_builtin_macro],
        id: DUMMY_NODE_ID,
        kind: ItemKind::MacroDef(MacroDef { tokens: TokenStream::new(trees), legacy: true }),
        vis: respan(span, VisibilityKind::Inherited),
        span: span,
        tokens: None,
    })
}

pub fn inject(
    krate: &mut Crate,
    resolver: &mut dyn Resolver,
    named_exts: Vec<NamedSyntaxExtension>,
    edition: Edition,
) {
    if !named_exts.is_empty() {
        let mut extra_items = Vec::new();
        let span = DUMMY_SP.fresh_expansion(ExpnData::allow_unstable(
            ExpnKind::AstPass(AstPass::PluginMacroDefs), DUMMY_SP, edition,
            [sym::rustc_attrs][..].into(),
        ));
        for (name, ext) in named_exts {
            resolver.register_builtin_macro(Ident::with_dummy_span(name), ext);
            extra_items.push(plugin_macro_def(name, span));
        }
        // The `macro_rules` items must be inserted before any other items.
        mem::swap(&mut extra_items, &mut krate.module.items);
        krate.module.items.append(&mut extra_items);
    }
}
