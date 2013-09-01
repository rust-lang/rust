// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Deprecated #[auto_encode] and #[auto_decode] syntax extensions

use ast;
use codemap::Span;
use ext::base::*;

pub fn expand_auto_encode(
    cx: @ExtCtxt,
    span: Span,
    _mitem: @ast::MetaItem,
    in_items: ~[@ast::item]
) -> ~[@ast::item] {
    cx.span_err(span, "`#[auto_encode]` is deprecated, use `#[deriving(Encodable)]` instead");
    in_items
}

pub fn expand_auto_decode(
    cx: @ExtCtxt,
    span: Span,
    _mitem: @ast::MetaItem,
    in_items: ~[@ast::item]
) -> ~[@ast::item] {
    cx.span_err(span, "`#[auto_decode]` is deprecated, use `#[deriving(Decodable)]` instead");
    in_items
}
