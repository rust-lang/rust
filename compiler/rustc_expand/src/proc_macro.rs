use crate::base::{self, *};
use crate::errors;
use crate::proc_macro_server;

use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::sync::Lrc;
use rustc_errors::ErrorGuaranteed;
use rustc_parse::parser::ForceCollect;
use rustc_session::config::ProcMacroExecutionStrategy;
use rustc_span::profiling::SpannedEventArgRecorder;
use rustc_span::{Span, DUMMY_SP};

struct CrossbeamMessagePipe<T> {
    tx: crossbeam_channel::Sender<T>,
    rx: crossbeam_channel::Receiver<T>,
}

impl<T> pm::bridge::server::MessagePipe<T> for CrossbeamMessagePipe<T> {
    fn new() -> (Self, Self) {
        let (tx1, rx1) = crossbeam_channel::bounded(1);
        let (tx2, rx2) = crossbeam_channel::bounded(1);
        (CrossbeamMessagePipe { tx: tx1, rx: rx2 }, CrossbeamMessagePipe { tx: tx2, rx: rx1 })
    }

    fn send(&mut self, value: T) {
        self.tx.send(value).unwrap();
    }

    fn recv(&mut self) -> Option<T> {
        self.rx.recv().ok()
    }
}

fn exec_strategy(ecx: &ExtCtxt<'_>) -> impl pm::bridge::server::ExecutionStrategy {
    pm::bridge::server::MaybeCrossThread::<CrossbeamMessagePipe<_>>::new(
        ecx.sess.opts.unstable_opts.proc_macro_execution_strategy
            == ProcMacroExecutionStrategy::CrossThread,
    )
}

pub struct BangProcMacro {
    pub client: pm::bridge::client::Client<pm::TokenStream, pm::TokenStream>,
}

impl base::BangProcMacro for BangProcMacro {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        let _timer =
            ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                recorder.record_arg_with_span(ecx.expansion_descr(), span);
            });

        let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
        let strategy = exec_strategy(ecx);
        let server = proc_macro_server::Rustc::new(ecx);
        self.client.run(&strategy, server, input, proc_macro_backtrace).map_err(|e| {
            ecx.sess.emit_err(errors::ProcMacroPanicked {
                span,
                message: e
                    .as_str()
                    .map(|message| errors::ProcMacroPanickedHelp { message: message.into() }),
            })
        })
    }
}

pub struct AttrProcMacro {
    pub client: pm::bridge::client::Client<(pm::TokenStream, pm::TokenStream), pm::TokenStream>,
}

impl base::AttrProcMacro for AttrProcMacro {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        let _timer =
            ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                recorder.record_arg_with_span(ecx.expansion_descr(), span);
            });

        let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
        let strategy = exec_strategy(ecx);
        let server = proc_macro_server::Rustc::new(ecx);
        self.client.run(&strategy, server, annotation, annotated, proc_macro_backtrace).map_err(
            |e| {
                let mut err = ecx.struct_span_err(span, "custom attribute panicked");
                if let Some(s) = e.as_str() {
                    err.help(&format!("message: {}", s));
                }
                err.emit()
            },
        )
    }
}

pub struct DeriveProcMacro {
    pub client: pm::bridge::client::Client<pm::TokenStream, pm::TokenStream>,
}

impl MultiItemModifier for DeriveProcMacro {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        _meta_item: &ast::MetaItem,
        item: Annotatable,
        _is_derive_const: bool,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        // We need special handling for statement items
        // (e.g. `fn foo() { #[derive(Debug)] struct Bar; }`)
        let is_stmt = matches!(item, Annotatable::Stmt(..));
        let hack = crate::base::ann_pretty_printing_compatibility_hack(&item, &ecx.sess.parse_sess);
        let input = if hack {
            let nt = match item {
                Annotatable::Item(item) => token::NtItem(item),
                Annotatable::Stmt(stmt) => token::NtStmt(stmt),
                _ => unreachable!(),
            };
            TokenStream::token_alone(token::Interpolated(Lrc::new(nt)), DUMMY_SP)
        } else {
            item.to_tokens()
        };

        let stream = {
            let _timer =
                ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                    recorder.record_arg_with_span(ecx.expansion_descr(), span);
                });
            let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
            let strategy = exec_strategy(ecx);
            let server = proc_macro_server::Rustc::new(ecx);
            match self.client.run(&strategy, server, input, proc_macro_backtrace) {
                Ok(stream) => stream,
                Err(e) => {
                    let mut err = ecx.struct_span_err(span, "proc-macro derive panicked");
                    if let Some(s) = e.as_str() {
                        err.help(&format!("message: {}", s));
                    }
                    err.emit();
                    return ExpandResult::Ready(vec![]);
                }
            }
        };

        let error_count_before = ecx.sess.parse_sess.span_diagnostic.err_count();
        let mut parser =
            rustc_parse::stream_to_parser(&ecx.sess.parse_sess, stream, Some("proc-macro derive"));
        let mut items = vec![];

        loop {
            match parser.parse_item(ForceCollect::No) {
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
            ecx.sess.emit_err(errors::ProcMacroDeriveTokens { span });
        }

        ExpandResult::Ready(items)
    }
}
