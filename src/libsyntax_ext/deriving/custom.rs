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
use proc_macro::{TokenStream, __internal};
use syntax::ast::{self, ItemKind, Attribute, Mac};
use syntax::attr::{mark_used, mark_known};
use syntax::codemap::Span;
use syntax::ext::base::*;
use syntax::fold::Folder;
use syntax::visit::Visitor;

struct MarkAttrs<'a>(&'a [ast::Name]);

impl<'a> Visitor<'a> for MarkAttrs<'a> {
    fn visit_attribute(&mut self, attr: &Attribute) {
        if self.0.contains(&attr.name()) {
            mark_used(attr);
            mark_known(attr);
        }
    }

    fn visit_mac(&mut self, _mac: &Mac) {}
}

pub struct CustomDerive {
    inner: fn(TokenStream) -> TokenStream,
    attrs: Vec<ast::Name>,
}

impl CustomDerive {
    pub fn new(inner: fn(TokenStream) -> TokenStream, attrs: Vec<ast::Name>) -> CustomDerive {
        CustomDerive { inner: inner, attrs: attrs }
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
            ItemKind::Enum(..) => {},
            _ => {
                ecx.span_err(span, "custom derive attributes may only be \
                                    applied to struct/enum items");
                return Vec::new()
            }
        }

        // Mark attributes as known, and used.
        MarkAttrs(&self.attrs).visit_item(&item);

        let input = __internal::new_token_stream(ecx.resolver.eliminate_crate_var(item.clone()));
        let res = __internal::set_parse_sess(&ecx.parse_sess, || {
            let inner = self.inner;
            panic::catch_unwind(panic::AssertUnwindSafe(|| inner(input)))
        });

        let stream = match res {
            Ok(stream) => stream,
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

        let new_items = __internal::set_parse_sess(&ecx.parse_sess, || {
            match __internal::token_stream_parse_items(stream) {
                Ok(new_items) => new_items,
                Err(_) => {
                    // FIXME: handle this better
                    let msg = "custom derive produced unparseable tokens";
                    ecx.struct_span_fatal(span, msg).emit();
                    panic!(FatalError);
                }
            }
        });

        let mut res = vec![Annotatable::Item(item)];
        // Reassign spans of all expanded items to the input `item`
        // for better errors here.
        res.extend(new_items.into_iter().flat_map(|item| {
            ChangeSpan { span: span }.fold_item(item)
        }).map(Annotatable::Item));
        res
    }
}
