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

use rustc_macro::{TokenStream, __internal};
use syntax::ast::{self, ItemKind};
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::fold::{self, Folder};
use errors::FatalError;

pub struct CustomDerive {
    inner: fn(TokenStream) -> TokenStream,
}

impl CustomDerive {
    pub fn new(inner: fn(TokenStream) -> TokenStream) -> CustomDerive {
        CustomDerive { inner: inner }
    }
}

impl MultiItemModifier for CustomDerive {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              _meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Vec<Annotatable> {
        let item = match item {
            Annotatable::Item(item) => item,
            Annotatable::ImplItem(_) |
            Annotatable::TraitItem(_) => {
                ecx.span_err(span, "custom derive attributes may only be \
                                    applied to struct/enum items");
                return Vec::new()
            }
        };
        match item.node {
            ItemKind::Struct(..) |
            ItemKind::Enum(..) => {}
            _ => {
                ecx.span_err(span, "custom derive attributes may only be \
                                    applied to struct/enum items");
                return Vec::new()
            }
        }

        let input = __internal::new_token_stream(item);
        let res = __internal::set_parse_sess(&ecx.parse_sess, || {
            let inner = self.inner;
            panic::catch_unwind(panic::AssertUnwindSafe(|| inner(input)))
        });
        let item = match res {
            Ok(stream) => __internal::token_stream_items(stream),
            Err(e) => {
                let msg = "custom derive attribute panicked";
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
        };

        // Right now we have no knowledge of spans at all in custom derive
        // macros, everything is just parsed as a string. Reassign all spans to
        // the #[derive] attribute for better errors here.
        item.into_iter().flat_map(|item| {
            ChangeSpan { span: span }.fold_item(item)
        }).map(Annotatable::Item).collect()
    }
}

struct ChangeSpan { span: Span }

impl Folder for ChangeSpan {
    fn new_span(&mut self, _sp: Span) -> Span {
        self.span
    }

    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
}
