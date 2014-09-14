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
The compiler code necessary to implement the `#[deriving]` extensions.


FIXME (#2810): hygiene. Search for "__" strings (in other files too).
We also assume "extra" is the standard library, and "std" is the core
library.

*/

use ast::{Item, MetaItem, MetaList, MetaNameValue, MetaWord};
use ext::base::ExtCtxt;
use codemap::Span;
use ptr::P;

pub mod bounds;
pub mod clone;
pub mod encodable;
pub mod decodable;
pub mod hash;
pub mod rand;
pub mod show;
pub mod zero;
pub mod default;
pub mod primitive;

#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/totaleq.rs"]
pub mod totaleq;
#[path="cmp/ord.rs"]
pub mod ord;
#[path="cmp/totalord.rs"]
pub mod totalord;


pub mod generic;

pub fn expand_meta_deriving(cx: &mut ExtCtxt,
                            _span: Span,
                            mitem: &MetaItem,
                            item: &Item,
                            push: |P<Item>|) {
    match mitem.node {
        MetaNameValue(_, ref l) => {
            cx.span_err(l.span, "unexpected value in `deriving`");
        }
        MetaWord(_) => {
            cx.span_warn(mitem.span, "empty trait list in `deriving`");
        }
        MetaList(_, ref titems) if titems.len() == 0 => {
            cx.span_warn(mitem.span, "empty trait list in `deriving`");
        }
        MetaList(_, ref titems) => {
            for titem in titems.iter().rev() {
                match titem.node {
                    MetaNameValue(ref tname, _) |
                    MetaList(ref tname, _) |
                    MetaWord(ref tname) => {
                        macro_rules! expand(($func:path) => ($func(cx, titem.span,
                                                                   &**titem, item,
                                                                   |i| push(i))));
                        match tname.get() {
                            "Clone" => expand!(clone::expand_deriving_clone),

                            "Hash" => expand!(hash::expand_deriving_hash),

                            "Encodable" => expand!(encodable::expand_deriving_encodable),
                            "Decodable" => expand!(decodable::expand_deriving_decodable),

                            "PartialEq" => expand!(eq::expand_deriving_eq),
                            "Eq" => expand!(totaleq::expand_deriving_totaleq),
                            "PartialOrd" => expand!(ord::expand_deriving_ord),
                            "Ord" => expand!(totalord::expand_deriving_totalord),

                            "Rand" => expand!(rand::expand_deriving_rand),

                            "Show" => expand!(show::expand_deriving_show),

                            "Zero" => expand!(zero::expand_deriving_zero),
                            "Default" => expand!(default::expand_deriving_default),

                            "FromPrimitive" => expand!(primitive::expand_deriving_from_primitive),

                            "Send" => expand!(bounds::expand_deriving_bound),
                            "Sync" => expand!(bounds::expand_deriving_bound),
                            "Copy" => expand!(bounds::expand_deriving_bound),

                            ref tname => {
                                cx.span_err(titem.span,
                                            format!("unknown `deriving` \
                                                     trait: `{}`",
                                                    *tname).as_slice());
                            }
                        };
                    }
                }
            }
        }
    }
}
