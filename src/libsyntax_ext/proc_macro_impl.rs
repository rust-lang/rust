use crate::proc_macro_server;

use errors::FatalError;
use syntax::source_map::Span;
use syntax::ext::base::{self, *};
use syntax::tokenstream::TokenStream;

pub const EXEC_STRATEGY: proc_macro::bridge::server::SameThread =
    proc_macro::bridge::server::SameThread;

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
