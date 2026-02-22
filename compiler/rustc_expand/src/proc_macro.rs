use rustc_ast::tokenstream::TokenStream;
use rustc_errors::ErrorGuaranteed;
use rustc_middle::ty::{self, TyCtxt};
use rustc_parse::parser::{AllowConstBlockItems, ForceCollect, Parser};
use rustc_session::Session;
use rustc_session::config::ProcMacroExecutionStrategy;
use rustc_span::profiling::SpannedEventArgRecorder;
use rustc_span::{LocalExpnId, Span};
use {rustc_ast as ast, rustc_proc_macro as pm};

use crate::base::{self, *};
use crate::{errors, proc_macro_server};

fn exec_strategy(sess: &Session) -> impl pm::bridge::server::ExecutionStrategy + 'static {
    pm::bridge::server::MaybeCrossThread {
        cross_thread: sess.opts.unstable_opts.proc_macro_execution_strategy
            == ProcMacroExecutionStrategy::CrossThread,
    }
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
        let mut server = proc_macro_server::Rustc::new(ecx);
        let input = server.ts_rustc_to_pm(input);
        let output = self.client.run(&strategy, &mut server, input, proc_macro_backtrace);

        match output {
            Ok(stream) => Ok(server.ts_pm_to_rustc(stream)),
            Err(e) => Err(ecx.dcx().emit_err(errors::ProcMacroPanicked {
                span,
                message: e
                    .as_str()
                    .map(|message| errors::ProcMacroPanickedHelp { message: message.into() }),
            })),
        }
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
        let mut server = proc_macro_server::Rustc::new(ecx);
        let annotation = server.ts_rustc_to_pm(annotation);
        let annotated = server.ts_rustc_to_pm(annotated);
        let output =
            self.client.run(&strategy, &mut server, annotation, annotated, proc_macro_backtrace);
        match output {
            Ok(stream) => Ok(server.ts_pm_to_rustc(stream)),
            Err(e) => Err(ecx.dcx().emit_err(errors::CustomAttributePanicked {
                span,
                message: e
                    .as_str()
                    .map(|message| errors::CustomAttributePanickedHelp { message: message.into() }),
            })),
        }
    }
}

pub struct DeriveProcMacro {
    pub client: DeriveClient,
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

        let input = item.to_tokens();

        let invoc_id = ecx.current_expansion.id;

        let res = if ecx.sess.opts.incremental.is_some()
            && ecx.sess.opts.unstable_opts.cache_proc_macros
        {
            ty::tls::with(|tcx| {
                let input = &*tcx.arena.alloc(input);
                let key: (LocalExpnId, &TokenStream) = (invoc_id, input);

                QueryDeriveExpandCtx::enter(ecx, self.client, move || {
                    tcx.derive_macro_expansion(key).cloned()
                })
            })
        } else {
            expand_derive_macro(invoc_id, input, ecx, self.client)
        };

        let Ok(output) = res else {
            // error will already have been emitted
            return ExpandResult::Ready(vec![]);
        };

        let error_count_before = ecx.dcx().err_count();
        let mut parser = Parser::new(&ecx.sess.psess, output, Some("proc-macro derive"));
        let mut items = vec![];

        loop {
            match parser.parse_item(
                ForceCollect::No,
                if is_stmt { AllowConstBlockItems::No } else { AllowConstBlockItems::Yes },
            ) {
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
    key: (LocalExpnId, &'tcx TokenStream),
) -> Result<&'tcx TokenStream, ()> {
    let (invoc_id, input) = key;

    // Make sure that we invalidate the query when the crate defining the proc macro changes
    let _ = tcx.crate_hash(invoc_id.expn_data().macro_def_id.unwrap().krate);

    QueryDeriveExpandCtx::with(|ecx, client| {
        expand_derive_macro(invoc_id, input.clone(), ecx, client).map(|ts| &*tcx.arena.alloc(ts))
    })
}

type DeriveClient = pm::bridge::client::Client<pm::TokenStream, pm::TokenStream>;

fn expand_derive_macro(
    invoc_id: LocalExpnId,
    input: TokenStream,
    ecx: &mut ExtCtxt<'_>,
    client: DeriveClient,
) -> Result<TokenStream, ()> {
    let _timer =
        ecx.sess.prof.generic_activity_with_arg_recorder("expand_proc_macro", |recorder| {
            let invoc_expn_data = invoc_id.expn_data();
            let span = invoc_expn_data.call_site;
            let event_arg = invoc_expn_data.kind.descr();
            recorder.record_arg_with_span(ecx.sess.source_map(), event_arg.clone(), span);
        });

    let proc_macro_backtrace = ecx.ecfg.proc_macro_backtrace;
    let strategy = exec_strategy(ecx.sess);
    let mut server = proc_macro_server::Rustc::new(ecx);
    let input = server.ts_rustc_to_pm(input);

    match client.run(&strategy, &mut server, input, proc_macro_backtrace) {
        Ok(stream) => Ok(server.ts_pm_to_rustc(stream)),
        Err(e) => {
            let invoc_expn_data = invoc_id.expn_data();
            let span = invoc_expn_data.call_site;
            ecx.dcx().emit_err({
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

/// Stores the context necessary to expand a derive proc macro via a query.
struct QueryDeriveExpandCtx {
    /// Type-erased version of `&mut ExtCtxt`
    expansion_ctx: *mut (),
    client: DeriveClient,
}

impl QueryDeriveExpandCtx {
    /// Store the extension context and the client into the thread local value.
    /// It will be accessible via the `with` method while `f` is active.
    fn enter<F, R>(ecx: &mut ExtCtxt<'_>, client: DeriveClient, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // We need erasure to get rid of the lifetime
        let ctx = Self { expansion_ctx: ecx as *mut _ as *mut (), client };
        DERIVE_EXPAND_CTX.set(&ctx, || f())
    }

    /// Accesses the thread local value of the derive expansion context.
    /// Must be called while the `enter` function is active.
    fn with<F, R>(f: F) -> R
    where
        F: for<'a, 'b> FnOnce(&'b mut ExtCtxt<'a>, DeriveClient) -> R,
    {
        DERIVE_EXPAND_CTX.with(|ctx| {
            let ectx = {
                let casted = ctx.expansion_ctx.cast::<ExtCtxt<'_>>();
                // SAFETY: We can only get the value from `with` while the `enter` function
                // is active (on the callstack), and that function's signature ensures that the
                // lifetime is valid.
                // If `with` is called at some other time, it will panic due to usage of
                // `scoped_tls::with`.
                unsafe { casted.as_mut().unwrap() }
            };

            f(ectx, ctx.client)
        })
    }
}

// When we invoke a query to expand a derive proc macro, we need to provide it with the expansion
// context and derive Client. We do that using a thread-local.
scoped_tls::scoped_thread_local!(static DERIVE_EXPAND_CTX: QueryDeriveExpandCtx);
