use crate::base::{self, *};
use crate::proc_macro_server;

use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::{CanSynthesizeMissingTokens, TokenStream, TokenTree};
use rustc_ast::{self as ast, *};
use rustc_data_structures::sync::Lrc;
use rustc_errors::{struct_span_err, Applicability, ErrorReported};
use rustc_lexer::is_ident;
use rustc_parse::nt_to_tokenstream;
use rustc_span::symbol::sym;
use rustc_span::{Span, DUMMY_SP};

const EXEC_STRATEGY: pm::bridge::server::SameThread = pm::bridge::server::SameThread;

pub struct BangProcMacro {
    pub client: pm::bridge::client::Client<fn(pm::TokenStream) -> pm::TokenStream>,
}

impl base::ProcMacro for BangProcMacro {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
    ) -> Result<TokenStream, ErrorReported> {
        let server = proc_macro_server::Rustc::new(ecx);
        self.client.run(&EXEC_STRATEGY, server, input, ecx.ecfg.proc_macro_backtrace).map_err(|e| {
            let mut err = ecx.struct_span_err(span, "proc macro panicked");
            if let Some(s) = e.as_str() {
                err.help(&format!("message: {}", s));
            }
            err.emit();
            ErrorReported
        })
    }
}

pub struct AttrProcMacro {
    pub client: pm::bridge::client::Client<fn(pm::TokenStream, pm::TokenStream) -> pm::TokenStream>,
}

impl base::AttrProcMacro for AttrProcMacro {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorReported> {
        let server = proc_macro_server::Rustc::new(ecx);
        self.client
            .run(&EXEC_STRATEGY, server, annotation, annotated, ecx.ecfg.proc_macro_backtrace)
            .map_err(|e| {
                let mut err = ecx.struct_span_err(span, "custom attribute panicked");
                if let Some(s) = e.as_str() {
                    err.help(&format!("message: {}", s));
                }
                err.emit();
                ErrorReported
            })
    }
}

pub struct ProcMacroDerive {
    pub client: pm::bridge::client::Client<fn(pm::TokenStream) -> pm::TokenStream>,
}

impl MultiItemModifier for ProcMacroDerive {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        _meta_item: &ast::MetaItem,
        item: Annotatable,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        // We need special handling for statement items
        // (e.g. `fn foo() { #[derive(Debug)] struct Bar; }`)
        let mut is_stmt = false;
        let item = match item {
            Annotatable::Item(item) => token::NtItem(item),
            Annotatable::Stmt(stmt) => {
                is_stmt = true;
                assert!(stmt.is_item());

                // A proc macro can't observe the fact that we're passing
                // them an `NtStmt` - it can only see the underlying tokens
                // of the wrapped item
                token::NtStmt(stmt.into_inner())
            }
            _ => unreachable!(),
        };
        let input = if item.pretty_printing_compatibility_hack() {
            TokenTree::token(token::Interpolated(Lrc::new(item)), DUMMY_SP).into()
        } else {
            nt_to_tokenstream(&item, &ecx.sess.parse_sess, CanSynthesizeMissingTokens::Yes)
        };

        let server = proc_macro_server::Rustc::new(ecx);
        let stream =
            match self.client.run(&EXEC_STRATEGY, server, input, ecx.ecfg.proc_macro_backtrace) {
                Ok(stream) => stream,
                Err(e) => {
                    let mut err = ecx.struct_span_err(span, "proc-macro derive panicked");
                    if let Some(s) = e.as_str() {
                        err.help(&format!("message: {}", s));
                    }
                    err.emit();
                    return ExpandResult::Ready(vec![]);
                }
            };

        let error_count_before = ecx.sess.parse_sess.span_diagnostic.err_count();
        let mut parser =
            rustc_parse::stream_to_parser(&ecx.sess.parse_sess, stream, Some("proc-macro derive"));
        let mut items = vec![];

        loop {
            match parser.parse_item() {
                Ok(None) => break,
                Ok(Some(item)) => {
                    if is_stmt {
                        items.push(Annotatable::Stmt(P(ecx.stmt_item(span, item))));
                    } else {
                        items.push(Annotatable::Item(item));
                    }
                }
                Err(mut err) => {
                    err.emit();
                    break;
                }
            }
        }

        // fail if there have been errors emitted
        if ecx.sess.parse_sess.span_diagnostic.err_count() > error_count_before {
            ecx.struct_span_err(span, "proc-macro derive produced unparseable tokens").emit();
        }

        ExpandResult::Ready(items)
    }
}

crate fn collect_derives(cx: &mut ExtCtxt<'_>, attrs: &mut Vec<ast::Attribute>) -> Vec<ast::Path> {
    let mut result = Vec::new();
    attrs.retain(|attr| {
        if !attr.has_name(sym::derive) {
            return true;
        }

        // 1) First let's ensure that it's a meta item.
        let nmis = match attr.meta_item_list() {
            None => {
                cx.struct_span_err(attr.span, "malformed `derive` attribute input")
                    .span_suggestion(
                        attr.span,
                        "missing traits to be derived",
                        "#[derive(Trait1, Trait2, ...)]".to_owned(),
                        Applicability::HasPlaceholders,
                    )
                    .emit();
                return false;
            }
            Some(x) => x,
        };

        let mut error_reported_filter_map = false;
        let mut error_reported_map = false;
        let traits = nmis
            .into_iter()
            // 2) Moreover, let's ensure we have a path and not `#[derive("foo")]`.
            .filter_map(|nmi| match nmi {
                NestedMetaItem::Literal(lit) => {
                    error_reported_filter_map = true;
                    let mut err = struct_span_err!(
                        cx.sess,
                        lit.span,
                        E0777,
                        "expected path to a trait, found literal",
                    );
                    let token = lit.token.to_string();
                    if token.starts_with('"')
                        && token.len() > 2
                        && is_ident(&token[1..token.len() - 1])
                    {
                        err.help(&format!("try using `#[derive({})]`", &token[1..token.len() - 1]));
                    } else {
                        err.help("for example, write `#[derive(Debug)]` for `Debug`");
                    }
                    err.emit();
                    None
                }
                NestedMetaItem::MetaItem(mi) => Some(mi),
            })
            // 3) Finally, we only accept `#[derive($path_0, $path_1, ..)]`
            // but not e.g. `#[derive($path_0 = "value", $path_1(abc))]`.
            // In this case we can still at least determine that the user
            // wanted this trait to be derived, so let's keep it.
            .map(|mi| {
                let mut traits_dont_accept = |title, action| {
                    error_reported_map = true;
                    let sp = mi.span.with_lo(mi.path.span.hi());
                    cx.struct_span_err(sp, title)
                        .span_suggestion(
                            sp,
                            action,
                            String::new(),
                            Applicability::MachineApplicable,
                        )
                        .emit();
                };
                match &mi.kind {
                    MetaItemKind::List(..) => traits_dont_accept(
                        "traits in `#[derive(...)]` don't accept arguments",
                        "remove the arguments",
                    ),
                    MetaItemKind::NameValue(..) => traits_dont_accept(
                        "traits in `#[derive(...)]` don't accept values",
                        "remove the value",
                    ),
                    MetaItemKind::Word => {}
                }
                mi.path
            });

        result.extend(traits);
        !error_reported_filter_map && !error_reported_map
    });
    result
}
