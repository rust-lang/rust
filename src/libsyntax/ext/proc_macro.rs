use crate::ast::{self, ItemKind, Attribute, Mac};
use crate::attr::{mark_used, mark_known, HasAttrs};
use crate::errors::{Applicability, FatalError};
use crate::ext::base::{self, *};
use crate::ext::build::AstBuilder;
use crate::ext::proc_macro_server;
use crate::parse::{self, token};
use crate::parse::parser::PathStyle;
use crate::symbol::{sym, Symbol};
use crate::tokenstream::{self, TokenStream};
use crate::visit::Visitor;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use syntax_pos::hygiene::{ExpnInfo, ExpnKind};
use syntax_pos::{Span, DUMMY_SP};

const EXEC_STRATEGY: proc_macro::bridge::server::SameThread =
    proc_macro::bridge::server::SameThread;

pub struct BangProcMacro {
    pub client: proc_macro::bridge::client::Client<
        fn(proc_macro::TokenStream) -> proc_macro::TokenStream,
    >,
}

impl base::ProcMacro for BangProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt<'_>,
                   span: Span,
                   input: TokenStream)
                   -> TokenStream {
        let server = proc_macro_server::Rustc::new(ecx);
        match self.client.run(&EXEC_STRATEGY, server, input) {
            Ok(stream) => stream,
            Err(e) => {
                let msg = "proc macro panicked";
                let mut err = ecx.struct_span_fatal(span, msg);
                if let Some(s) = e.as_str() {
                    err.help(&format!("message: {}", s));
                }

                err.emit();
                FatalError.raise();
            }
        }
    }
}

pub struct AttrProcMacro {
    pub client: proc_macro::bridge::client::Client<
        fn(proc_macro::TokenStream, proc_macro::TokenStream) -> proc_macro::TokenStream,
    >,
}

impl base::AttrProcMacro for AttrProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt<'_>,
                   span: Span,
                   annotation: TokenStream,
                   annotated: TokenStream)
                   -> TokenStream {
        let server = proc_macro_server::Rustc::new(ecx);
        match self.client.run(&EXEC_STRATEGY, server, annotation, annotated) {
            Ok(stream) => stream,
            Err(e) => {
                let msg = "custom attribute panicked";
                let mut err = ecx.struct_span_fatal(span, msg);
                if let Some(s) = e.as_str() {
                    err.help(&format!("message: {}", s));
                }

                err.emit();
                FatalError.raise();
            }
        }
    }
}

pub struct ProcMacroDerive {
    pub client: proc_macro::bridge::client::Client<
        fn(proc_macro::TokenStream) -> proc_macro::TokenStream,
    >,
    pub attrs: Vec<ast::Name>,
}

impl MultiItemModifier for ProcMacroDerive {
    fn expand(&self,
              ecx: &mut ExtCtxt<'_>,
              span: Span,
              _meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Vec<Annotatable> {
        let item = match item {
            Annotatable::Item(item) => item,
            Annotatable::ImplItem(_) |
            Annotatable::TraitItem(_) |
            Annotatable::ForeignItem(_) |
            Annotatable::Stmt(_) |
            Annotatable::Expr(_) => {
                ecx.span_err(span, "proc-macro derives may only be \
                                    applied to a struct, enum, or union");
                return Vec::new()
            }
        };
        match item.node {
            ItemKind::Struct(..) |
            ItemKind::Enum(..) |
            ItemKind::Union(..) => {},
            _ => {
                ecx.span_err(span, "proc-macro derives may only be \
                                    applied to a struct, enum, or union");
                return Vec::new()
            }
        }

        // Mark attributes as known, and used.
        MarkAttrs(&self.attrs).visit_item(&item);

        let token = token::Interpolated(Lrc::new(token::NtItem(item)));
        let input = tokenstream::TokenTree::token(token, DUMMY_SP).into();

        let server = proc_macro_server::Rustc::new(ecx);
        let stream = match self.client.run(&EXEC_STRATEGY, server, input) {
            Ok(stream) => stream,
            Err(e) => {
                let msg = "proc-macro derive panicked";
                let mut err = ecx.struct_span_fatal(span, msg);
                if let Some(s) = e.as_str() {
                    err.help(&format!("message: {}", s));
                }

                err.emit();
                FatalError.raise();
            }
        };

        let error_count_before = ecx.parse_sess.span_diagnostic.err_count();
        let msg = "proc-macro derive produced unparseable tokens";

        let mut parser = parse::stream_to_parser(ecx.parse_sess, stream, Some("proc-macro derive"));
        let mut items = vec![];

        loop {
            match parser.parse_item() {
                Ok(None) => break,
                Ok(Some(item)) => {
                    items.push(Annotatable::Item(item))
                }
                Err(mut err) => {
                    // FIXME: handle this better
                    err.cancel();
                    ecx.struct_span_fatal(span, msg).emit();
                    FatalError.raise();
                }
            }
        }


        // fail if there have been errors emitted
        if ecx.parse_sess.span_diagnostic.err_count() > error_count_before {
            ecx.struct_span_fatal(span, msg).emit();
            FatalError.raise();
        }

        items
    }
}

struct MarkAttrs<'a>(&'a [ast::Name]);

impl<'a> Visitor<'a> for MarkAttrs<'a> {
    fn visit_attribute(&mut self, attr: &Attribute) {
        if let Some(ident) = attr.ident() {
            if self.0.contains(&ident.name) {
                mark_used(attr);
                mark_known(attr);
            }
        }
    }

    fn visit_mac(&mut self, _mac: &Mac) {}
}

pub fn is_proc_macro_attr(attr: &Attribute) -> bool {
    [sym::proc_macro, sym::proc_macro_attribute, sym::proc_macro_derive]
        .iter().any(|kind| attr.check_name(*kind))
}

crate fn collect_derives(cx: &mut ExtCtxt<'_>, attrs: &mut Vec<ast::Attribute>) -> Vec<ast::Path> {
    let mut result = Vec::new();
    attrs.retain(|attr| {
        if attr.path != sym::derive {
            return true;
        }
        if !attr.is_meta_item_list() {
            cx.struct_span_err(attr.span, "malformed `derive` attribute input")
                .span_suggestion(
                    attr.span,
                    "missing traits to be derived",
                    "#[derive(Trait1, Trait2, ...)]".to_owned(),
                    Applicability::HasPlaceholders,
                ).emit();
            return false;
        }

        match attr.parse_list(cx.parse_sess,
                              |parser| parser.parse_path_allowing_meta(PathStyle::Mod)) {
            Ok(traits) => {
                result.extend(traits);
                true
            }
            Err(mut e) => {
                e.emit();
                false
            }
        }
    });
    result
}

crate fn add_derived_markers<T: HasAttrs>(
    cx: &mut ExtCtxt<'_>, span: Span, traits: &[ast::Path], item: &mut T
) {
    let (mut names, mut pretty_name) = (FxHashSet::default(), String::new());
    for (i, path) in traits.iter().enumerate() {
        if i > 0 {
            pretty_name.push_str(", ");
        }
        pretty_name.push_str(&path.to_string());
        names.insert(unwrap_or!(path.segments.get(0), continue).ident.name);
    }

    let span = span.fresh_expansion(cx.current_expansion.id, ExpnInfo::allow_unstable(
        ExpnKind::Macro(MacroKind::Derive, Symbol::intern(&pretty_name)), span,
        cx.parse_sess.edition, cx.allow_derive_markers.clone(),
    ));

    item.visit_attrs(|attrs| {
        if names.contains(&sym::Eq) && names.contains(&sym::PartialEq) {
            let meta = cx.meta_word(span, sym::structural_match);
            attrs.push(cx.attribute(span, meta));
        }
        if names.contains(&sym::Copy) {
            let meta = cx.meta_word(span, sym::rustc_copy_clone_marker);
            attrs.push(cx.attribute(span, meta));
        }
    });
}
