use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::ErrorGuaranteed;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::config::ProcMacroExecutionStrategy;
use rustc_span::Span;
use rustc_span::profiling::SpannedEventArgRecorder;
use {rustc_ast as ast, rustc_proc_macro as pm};

use crate::base::{self, *};
use crate::{errors, proc_macro_server};

struct MessagePipe<T> {
    tx: std::sync::mpsc::SyncSender<T>,
    rx: std::sync::mpsc::Receiver<T>,
}

impl<T> pm::bridge::server::MessagePipe<T> for MessagePipe<T> {
    fn new() -> (Self, Self) {
        let (tx1, rx1) = std::sync::mpsc::sync_channel(1);
        let (tx2, rx2) = std::sync::mpsc::sync_channel(1);
        (MessagePipe { tx: tx1, rx: rx2 }, MessagePipe { tx: tx2, rx: rx1 })
    }

    fn send(&mut self, value: T) {
        self.tx.send(value).unwrap();
    }

    fn recv(&mut self) -> Option<T> {
        self.rx.recv().ok()
    }
}

fn exec_strategy(ecx: &ExtCtxt<'_>) -> impl pm::bridge::server::ExecutionStrategy + 'static {
    pm::bridge::server::MaybeCrossThread::<MessagePipe<_>>::new(
        ecx.sess.opts.unstable_opts.proc_macro_execution_strategy
            == ProcMacroExecutionStrategy::CrossThread,
    )
}

pub struct BangProcMacro {
    pub client: pm::bridge::client::Client<pm::TokenStream, pm::TokenStream>,
}

impl base::BangProcMacro for BangProcMacro {
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        let _timer =
            ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                recorder.record_arg_with_span(ecx.sess.source_map(), ecx.expansion_descr(), span);
            });

        let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
        let strategy = exec_strategy(ecx);
        let server = proc_macro_server::Rustc::new(ecx);
        self.client.run(&strategy, server, input, proc_macro_backtrace).map_err(|e| {
            ecx.dcx().emit_err(errors::ProcMacroPanicked {
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
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        let _timer =
            ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                recorder.record_arg_with_span(ecx.sess.source_map(), ecx.expansion_descr(), span);
            });

        let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
        let strategy = exec_strategy(ecx);
        let server = proc_macro_server::Rustc::new(ecx);
        self.client.run(&strategy, server, annotation, annotated, proc_macro_backtrace).map_err(
            |e| {
                ecx.dcx().emit_err(errors::CustomAttributePanicked {
                    span,
                    message: e.as_str().map(|message| errors::CustomAttributePanickedHelp {
                        message: message.into(),
                    }),
                })
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

        // We used to have an alternative behaviour for crates that needed it.
        // We had a lint for a long time, but now we just emit a hard error.
        // Eventually we might remove the special case hard error check
        // altogether. See #73345.
        crate::base::ann_pretty_printing_compatibility_hack(&item, &ecx.sess.psess);
        let input = item.to_tokens();
        let stream = {
            let _timer =
                ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
                    recorder.record_arg_with_span(
                        ecx.sess.source_map(),
                        ecx.expansion_descr(),
                        span,
                    );
                });
            let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
            let strategy = exec_strategy(ecx);
            let server = proc_macro_server::Rustc::new(ecx);
            match self.client.run(&strategy, server, input, proc_macro_backtrace) {
                Ok(stream) => stream,
                Err(e) => {
                    ecx.dcx().emit_err({
                        errors::ProcMacroDerivePanicked {
                            span,
                            message: e.as_str().map(|message| {
                                errors::ProcMacroDerivePanickedHelp { message: message.into() }
                            }),
                        }
                    });
                    return ExpandResult::Ready(vec![]);
                }
            }
        };

        let error_count_before = ecx.dcx().err_count();
        let mut parser = Parser::new(&ecx.sess.psess, stream, Some("proc-macro derive"));
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
                Err(err) => {
                    err.emit();
                    break;
                }
            }
        }

        // fail if there have been errors emitted
        if ecx.dcx().err_count() > error_count_before {
            ecx.dcx().emit_err(errors::ProcMacroDeriveTokens { span });
        }

        ExpandResult::Ready(items)
    }
}
