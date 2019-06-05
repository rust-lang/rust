use crate::proc_macro_impl::EXEC_STRATEGY;
use crate::proc_macro_server;

use errors::FatalError;
use rustc_data_structures::sync::Lrc;
use syntax::ast::{self, ItemKind, Attribute, Mac};
use syntax::attr::{mark_used, mark_known};
use syntax::source_map::Span;
use syntax::ext::base::*;
use syntax::parse;
use syntax::parse::token;
use syntax::tokenstream;
use syntax::visit::Visitor;
use syntax_pos::DUMMY_SP;

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
