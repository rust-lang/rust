use std::cell::Cell;
use std::ptr::NonNull;

use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::svh::Svh;
use rustc_errors::ErrorGuaranteed;
use rustc_middle::ty::{self, TyCtxt};
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::Session;
use rustc_session::config::ProcMacroExecutionStrategy;
use rustc_span::profiling::SpannedEventArgRecorder;
use rustc_span::{LocalExpnId, Span};
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

pub fn exec_strategy(sess: &Session) -> impl pm::bridge::server::ExecutionStrategy + 'static {
    pm::bridge::server::MaybeCrossThread::<MessagePipe<_>>::new(
        sess.opts.unstable_opts.proc_macro_execution_strategy
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
        let strategy = exec_strategy(ecx.sess);
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
        let strategy = exec_strategy(ecx.sess);
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
        let _timer = ecx.sess.prof.generic_activity_with_arg_recorder(
            "expand_derive_proc_macro_outer",
            |recorder| {
                recorder.record_arg_with_span(ecx.sess.source_map(), ecx.expansion_descr(), span);
            },
        );

        // We need special handling for statement items
        // (e.g. `fn foo() { #[derive(Debug)] struct Bar; }`)
        let is_stmt = matches!(item, Annotatable::Stmt(..));

        // We used to have an alternative behaviour for crates that needed it.
        // We had a lint for a long time, but now we just emit a hard error.
        // Eventually we might remove the special case hard error check
        // altogether. See #73345.
        crate::base::ann_pretty_printing_compatibility_hack(&item, &ecx.sess.psess);
        let input = item.to_tokens();
        let res = ty::tls::with(|tcx| {
            let input = tcx.arena.alloc(input) as &TokenStream;
            let invoc_id = ecx.current_expansion.id;
            let invoc_expn_data = invoc_id.expn_data();

            assert_eq!(invoc_expn_data.call_site, span);

            // FIXME(pr-time): Is this the correct way to check for incremental compilation (as
            // well as for `cache_proc_macros`)?
            if tcx.sess.opts.incremental.is_some()
                && tcx.sess.opts.unstable_opts.cache_derive_macros
            {
                // FIXME(pr-time): Just using the crate hash to notice when the proc-macro code has
                // changed. How to *correctly* depend on exactly the macro definition?
                // I.e., depending on the crate hash is just a HACK, and ideally the dependency would be
                // more narrow.
                let macro_def_id = invoc_expn_data.macro_def_id.unwrap();
                let proc_macro_crate_hash = tcx.crate_hash(macro_def_id.krate);

                let key = (invoc_id, proc_macro_crate_hash, input);

                enter_context((ecx, self.client), move || tcx.derive_macro_expansion(key).cloned())
            } else {
                expand_derive_macro(tcx, invoc_id, input, ecx, self.client).cloned()
            }
        });

        let Ok(output) = res else {
            // error will already have been emitted
            return ExpandResult::Ready(vec![]);
        };

        let error_count_before = ecx.dcx().err_count();
        let mut parser = Parser::new(&ecx.sess.psess, output, Some("proc-macro derive"));
        let mut items = vec![];

        loop {
            match parser.parse_item(ForceCollect::No) {
                Ok(None) => break,
                Ok(Some(item)) => {
                    if is_stmt {
                        items.push(Annotatable::Stmt(Box::new(ecx.stmt_item(span, item))));
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

/// Provide a query for computing the output of a derive macro.
pub(super) fn provide_derive_macro_expansion<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (LocalExpnId, Svh, &'tcx TokenStream),
) -> Result<&'tcx TokenStream, ()> {
    let (invoc_id, _macro_crate_hash, input) = key;

    with_context(|(ecx, client)| expand_derive_macro(tcx, invoc_id, input, ecx, *client))
}

type CLIENT = pm::bridge::client::Client<pm::TokenStream, pm::TokenStream>;

fn expand_derive_macro<'tcx>(
    tcx: TyCtxt<'tcx>,
    invoc_id: LocalExpnId,
    input: &'tcx TokenStream,
    ecx: &mut ExtCtxt<'_>,
    client: CLIENT,
) -> Result<&'tcx TokenStream, ()> {
    let invoc_expn_data = invoc_id.expn_data();
    let span = invoc_expn_data.call_site;
    let event_arg = invoc_expn_data.kind.descr();
    let _timer = tcx.sess.prof.generic_activity_with_arg_recorder(
        "expand_derive_proc_macro_inner",
        |recorder| {
            recorder.record_arg_with_span(tcx.sess.source_map(), event_arg.clone(), span);
        },
    );

    let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
    let strategy = crate::proc_macro::exec_strategy(tcx.sess);
    let server = crate::proc_macro_server::Rustc::new(ecx);

    match client.run(&strategy, server, input.clone(), proc_macro_backtrace) {
        Ok(stream) => Ok(tcx.arena.alloc(stream) as &TokenStream),
        Err(e) => {
            tcx.dcx().emit_err({
                errors::ProcMacroDerivePanicked {
                    span,
                    message: e.as_str().map(|message| errors::ProcMacroDerivePanickedHelp {
                        message: message.into(),
                    }),
                }
            });
            Err(())
        }
    }
}

// based on rust/compiler/rustc_middle/src/ty/context/tls.rs
thread_local! {
    /// A thread local variable that stores a pointer to the current `CONTEXT`.
    static TLV: Cell<(*mut (), Option<CLIENT>)> = const { Cell::new((std::ptr::null_mut(), None)) };
}

/// Sets `context` as the new current `CONTEXT` for the duration of the function `f`.
#[inline]
pub(crate) fn enter_context<'a, F, R>(context: (&mut ExtCtxt<'a>, CLIENT), f: F) -> R
where
    F: FnOnce() -> R,
{
    let (ectx, client) = context;
    let erased = (ectx as *mut _ as *mut (), Some(client));
    TLV.with(|tlv| {
        let old = tlv.replace(erased);
        let _reset = rustc_data_structures::defer(move || tlv.set(old));
        f()
    })
}

/// Allows access to the current `CONTEXT`.
/// Panics if there is no `CONTEXT` available.
#[inline]
#[track_caller]
fn with_context<F, R>(f: F) -> R
where
    F: for<'a, 'b> FnOnce(&'b mut (&mut ExtCtxt<'a>, CLIENT)) -> R,
{
    let (ectx, client_opt) = TLV.get();
    let ectx = NonNull::new(ectx).expect("no CONTEXT stored in tls");

    // We could get an `CONTEXT` pointer from another thread.
    // Ensure that `CONTEXT` is `DynSync`.
    // FIXME(pr-time): we should not be able to?
    // sync::assert_dyn_sync::<CONTEXT<'_>>();

    // prevent double entering, as that would allow creating two `&mut ExtCtxt`s
    // FIXME(pr-time): probably use a RefCell instead (which checks this properly)?
    TLV.with(|tlv| {
        let old = tlv.replace((std::ptr::null_mut(), None));
        let _reset = rustc_data_structures::defer(move || tlv.set(old));
        let ectx = {
            let mut casted = ectx.cast::<ExtCtxt<'_>>();
            unsafe { casted.as_mut() }
        };

        f(&mut (ectx, client_opt.unwrap()))
    })
}
