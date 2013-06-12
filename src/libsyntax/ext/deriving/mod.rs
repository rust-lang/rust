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

use core::prelude::*;
use core::iterator::IteratorUtil;

use ast::{enum_def, ident, item, Generics, meta_item, struct_def};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap::span;

pub mod clone;
pub mod iter_bytes;
pub mod encodable;
pub mod decodable;
pub mod rand;
pub mod to_str;

#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/totaleq.rs"]
pub mod totaleq;
#[path="cmp/ord.rs"]
pub mod ord;
#[path="cmp/totalord.rs"]
pub mod totalord;


pub mod generic;

pub type ExpandDerivingStructDefFn<'self> = &'self fn(@ExtCtxt,
                                                       span,
                                                       x: &struct_def,
                                                       ident,
                                                       y: &Generics)
                                                 -> @item;
pub type ExpandDerivingEnumDefFn<'self> = &'self fn(@ExtCtxt,
                                                    span,
                                                    x: &enum_def,
                                                    ident,
                                                    y: &Generics)
                                                 -> @item;

pub fn expand_meta_deriving(cx: @ExtCtxt,
                            _span: span,
                            mitem: @meta_item,
                            in_items: ~[@item])
                         -> ~[@item] {
    use ast::{meta_list, meta_name_value, meta_word};

    match mitem.node {
        meta_name_value(_, ref l) => {
            cx.span_err(l.span, "unexpected value in `deriving`");
            in_items
        }
        meta_word(_) | meta_list(_, []) => {
            cx.span_warn(mitem.span, "empty trait list in `deriving`");
            in_items
        }
        meta_list(_, ref titems) => {
            do titems.rev_iter().fold(in_items) |in_items, &titem| {
                match titem.node {
                    meta_name_value(tname, _) |
                    meta_list(tname, _) |
                    meta_word(tname) => {
                        macro_rules! expand(($func:path) => ($func(cx, titem.span,
                                                                   titem, in_items)));
                        match tname.as_slice() {
                            "Clone" => expand!(clone::expand_deriving_clone),
                            "DeepClone" => expand!(clone::expand_deriving_deep_clone),

                            "IterBytes" => expand!(iter_bytes::expand_deriving_iter_bytes),

                            "Encodable" => expand!(encodable::expand_deriving_encodable),
                            "Decodable" => expand!(decodable::expand_deriving_decodable),

                            "Eq" => expand!(eq::expand_deriving_eq),
                            "TotalEq" => expand!(totaleq::expand_deriving_totaleq),
                            "Ord" => expand!(ord::expand_deriving_ord),
                            "TotalOrd" => expand!(totalord::expand_deriving_totalord),

                            "Rand" => expand!(rand::expand_deriving_rand),

                            "ToStr" => expand!(to_str::expand_deriving_to_str),

                            ref tname => {
                                cx.span_err(titem.span, fmt!("unknown \
                                    `deriving` trait: `%s`", *tname));
                                in_items
                            }
                        }
                    }
                }
            }
        }
    }
}
