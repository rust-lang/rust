// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::panic;

use errors::FatalError;

use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::tokenstream::TokenStream;
use syntax::ext::base;

use proc_macro::TokenStream as TsShim;
use proc_macro::__internal;

pub struct AttrProcMacro {
    pub inner: fn(TsShim, TsShim) -> TsShim,
}

impl base::AttrProcMacro for AttrProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   annotation: TokenStream,
                   annotated: TokenStream)
                   -> TokenStream {
        let annotation = __internal::token_stream_wrap(annotation);
        let annotated = __internal::token_stream_wrap(annotated);

        let res = __internal::set_parse_sess(&ecx.parse_sess, || {
            panic::catch_unwind(panic::AssertUnwindSafe(|| (self.inner)(annotation, annotated)))
        });

        match res {
            Ok(stream) => __internal::token_stream_inner(stream),
            Err(e) => {
                let msg = "custom attribute panicked";
                let mut err = ecx.struct_span_fatal(span, msg);
                if let Some(s) = e.downcast_ref::<String>() {
                    err.help(&format!("message: {}", s));
                }
                if let Some(s) = e.downcast_ref::<&'static str>() {
                    err.help(&format!("message: {}", s));
                }

                err.emit();
                panic!(FatalError);
            }
        }
    }
}
