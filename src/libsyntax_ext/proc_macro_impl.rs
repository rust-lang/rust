// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use errors::FatalError;

use syntax::source_map::Span;
use syntax::ext::base::*;
use syntax::tokenstream::TokenStream;
use syntax::ext::base;

pub const EXEC_STRATEGY: ::proc_macro::bridge::server::SameThread =
    ::proc_macro::bridge::server::SameThread;

pub struct AttrProcMacro {
    pub client: ::proc_macro::bridge::client::Client<
        fn(::proc_macro::TokenStream, ::proc_macro::TokenStream) -> ::proc_macro::TokenStream,
    >,
}

impl base::AttrProcMacro for AttrProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   annotation: TokenStream,
                   annotated: TokenStream)
                   -> TokenStream {
        let server = ::proc_macro::rustc::Rustc;
        let res = ::proc_macro::__internal::set_sess(ecx, || {
            self.client.run(&EXEC_STRATEGY, server, annotation, annotated)
        });

        match res {
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
    pub client: ::proc_macro::bridge::client::Client<
        fn(::proc_macro::TokenStream) -> ::proc_macro::TokenStream,
    >,
}

impl base::ProcMacro for BangProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   input: TokenStream)
                   -> TokenStream {
        let server = ::proc_macro::rustc::Rustc;
        let res = ::proc_macro::__internal::set_sess(ecx, || {
            self.client.run(&EXEC_STRATEGY, server, input)
        });

        match res {
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
