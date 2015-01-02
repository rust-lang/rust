// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The compiler code necessary to implement the `#[deriving]` extensions.
//!
//! FIXME (#2810): hygiene. Search for "__" strings (in other files too). We also assume "extra" is
//! the standard library, and "std" is the core library.

use attr::{mk_attr_id, AttrMetaMethods};
use ast::{AttrStyle, Attribute_, Item, MetaItem, MetaList, MetaNameValue, MetaWord};
use ext::base::ExtCtxt;
use codemap::{Span, Spanned};
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
                            mut push: Box<FnMut(P<Item>)>) {
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
                let mut attrs = Vec::new();
                match titem.node {
                    MetaNameValue(_, _) |
                    MetaWord(_) => { }
                    MetaList(_, ref alist) => {
                        if alist.len() != 1 {
                            cx.span_err(titem.span, "unexpected syntax in `deriving`")
                        } else {
                            let alist = &alist[0];
                            if alist.name().get() == "attributes" {
                                match alist.meta_item_list() {
                                    None => cx.span_err(alist.span,
                                                        "unexpected syntax in `deriving`"),
                                    Some(metas) => {
                                        for meta in metas.iter() {
                                            attrs.push(Spanned {
                                                span: meta.span,
                                                node: Attribute_ {
                                                    id: mk_attr_id(),
                                                    style: AttrStyle::AttrOuter,
                                                    value: meta.clone() /* bad clone */,
                                                    is_sugared_doc: false
                                                }
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // so we don't need to clone in the closure below. we also can't move,
                // unfortunately, since it would capture the outer `push` as well.
                let mut attrs = Some(attrs);

                match titem.node {
                    MetaNameValue(ref tname, _) |
                    MetaList(ref tname, _) |
                    MetaWord(ref tname) => {
                        // note: for the unwrap to succeed, this macro can't be evaluated more than
                        // once in any control flow branch. don't use it twice in a match arm
                        // below!
                        macro_rules! expand(
                            ($func:path) =>
                            ($func(cx, titem.span, &**titem, item,
                                   |i| push.call_mut((i.map(|mut i| {
                                       i.attrs.extend(attrs.take().unwrap().into_iter());
                                       i
                                   }),)))));
                        match tname.get() {
                            // if you change this list without updating reference.md, cmr will be
                            // sad
                            "Clone" => expand!(clone::expand_deriving_clone),

                            "Hash" => expand!(hash::expand_deriving_hash),

                            "RustcEncodable" => {
                                expand!(encodable::expand_deriving_rustc_encodable)
                            }
                            "RustcDecodable" => {
                                expand!(decodable::expand_deriving_rustc_decodable)
                            }
                            "Encodable" => {
                                cx.span_warn(titem.span,
                                             "deriving(Encodable) is deprecated \
                                              in favor of deriving(RustcEncodable)");

                                expand!(encodable::expand_deriving_encodable)
                            }
                            "Decodable" => {
                                cx.span_warn(titem.span,
                                             "deriving(Decodable) is deprecated \
                                              in favor of deriving(RustcDecodable)");

                                expand!(decodable::expand_deriving_decodable)
                            }

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
                                                    *tname)[]);
                            }
                        };
                    }
                }
            }
        }
    }
}
