// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
The compiler code necessary to implement the #[deriving] extensions.


FIXME (#2810)--Hygiene. Search for "__" strings (in other files too).
We also assume "extra" is the standard library, and "std" is the core
library.

*/

use ast::{item, lit, MetaItem, MetaList, MetaNameValue, MetaWord};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap::span;

pub mod clone;
pub mod iter_bytes;
pub mod encodable;
pub mod decodable;
pub mod rand;
pub mod to_str;
pub mod zero;

#[path="cmp/mod.rs"]
pub mod cmp;

pub mod generic;

pub fn expand_meta_deriving(cx: @ExtCtxt,
                            _span: span,
                            mitem: @MetaItem,
                            in_items: ~[@item])
                         -> ~[@item] {
    match mitem.node {
        MetaNameValue(_, ref l) => {
            cx.span_err(l.span, "unexpected value in `deriving`");
            in_items
        }
        MetaWord(_) | MetaList(_, []) => {
            cx.span_warn(mitem.span, "empty trait list in `deriving`");
            in_items
        }
        MetaList(_, ref titems) => {
            do titems.rev_iter().fold(in_items) |in_items, &titem| {
                let (name, options) = match titem.node {
                    MetaNameValue(name, ref lit) => (name, Lit(lit)),
                    MetaList(name, ref list) => (name, List(*list)),
                    MetaWord(name) => (name, NoOptions),
                };

                macro_rules! expand(($func:path) => ($func(cx, titem.span, options,
                                                           titem, in_items)));
                match name.as_slice() {
                    "Clone" => expand!(clone::expand_deriving_clone),
                    "DeepClone" => expand!(clone::expand_deriving_deep_clone),

                    "IterBytes" => expand!(iter_bytes::expand_deriving_iter_bytes),

                    "Encodable" => expand!(encodable::expand_deriving_encodable),
                    "Decodable" => expand!(decodable::expand_deriving_decodable),

                    "Eq" => expand!(cmp::eq::expand_deriving_eq),
                    "TotalEq" => expand!(cmp::totaleq::expand_deriving_totaleq),
                    "Ord" => expand!(cmp::ord::expand_deriving_ord),
                    "TotalOrd" => expand!(cmp::totalord::expand_deriving_totalord),

                    "Rand" => expand!(rand::expand_deriving_rand),

                    "ToStr" => expand!(to_str::expand_deriving_to_str),
                    "Zero" => expand!(zero::expand_deriving_zero),

                    _ => {
                        cx.span_err(titem.span, fmt!("unknown `deriving` trait: `%s`", name));
                        in_items
                    }
                }
            }
        }
    }
}

/// Summary of `#[deriving(SomeTrait(foo, option="bar"))]` and
/// `#[deriving(SomeTrait="foo")]`.
pub enum DerivingOptions<'self> {
    NoOptions,
    Lit(&'self lit),
    List(&'self [@MetaItem])
}

impl<'self> DerivingOptions<'self> {
    /// Emits a warning when there are options and the deriving
    /// implementation doesn't use them (i.e. it warns if `self` is
    /// not `NoOptions`.)
    pub fn unused_options_maybe_warn(&self, cx: @ExtCtxt, span: span, deriving_name: &str) {
        match *self {
            NoOptions => {},
            _ => {
                cx.span_warn(span,
                             fmt!("`#[deriving(%s)]` does not use any options.", deriving_name));
            }
        }
    }
}
