// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The compiler code necessary to implement the `#[derive]` extensions.
//!
//! FIXME (#2810): hygiene. Search for "__" strings (in other files too). We also assume "extra" is
//! the standard library, and "std" is the core library.

use ast::{Item, MetaItem, MetaList, MetaNameValue, MetaWord};
use ext::base::ExtCtxt;
use codemap::Span;
use ptr::P;

macro_rules! pathvec {
    ($($x:ident)::+) => (
        vec![ $( stringify!($x) ),+ ]
    )
}

macro_rules! path {
    ($($x:tt)*) => (
        ::ext::deriving::generic::ty::Path::new( pathvec!( $($x)* ) )
    )
}

macro_rules! pathvec_std {
    ($cx:expr, $first:ident :: $($rest:ident)::+) => (
        if $cx.use_std {
            pathvec!(std :: $($rest)::+)
        } else {
            pathvec!($first :: $($rest)::+)
        }
    )
}

macro_rules! path_std {
    ($($x:tt)*) => (
        ::ext::deriving::generic::ty::Path::new( pathvec_std!( $($x)* ) )
    )
}

pub mod bounds;
pub mod clone;
pub mod encodable;
pub mod decodable;
pub mod hash;
pub mod rand;
pub mod show;
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

pub fn expand_deprecated_deriving(cx: &mut ExtCtxt,
                                  span: Span,
                                  _: &MetaItem,
                                  _: &Item,
                                  _: &mut FnMut(P<Item>)) {
    cx.span_err(span, "`deriving` has been renamed to `derive`");
}

pub fn expand_meta_derive(cx: &mut ExtCtxt,
                          _span: Span,
                          mitem: &MetaItem,
                          item: &Item,
                          push: &mut FnMut(P<Item>)) {
    match mitem.node {
        MetaNameValue(_, ref l) => {
            cx.span_err(l.span, "unexpected value in `derive`");
        }
        MetaWord(_) => {
            cx.span_warn(mitem.span, "empty trait list in `derive`");
        }
        MetaList(_, ref titems) if titems.len() == 0 => {
            cx.span_warn(mitem.span, "empty trait list in `derive`");
        }
        MetaList(_, ref titems) => {
            for titem in titems.iter().rev() {
                match titem.node {
                    MetaNameValue(ref tname, _) |
                    MetaList(ref tname, _) |
                    MetaWord(ref tname) => {
                        macro_rules! expand {
                            ($func:path) => ($func(cx, titem.span, &**titem, item,
                                                   |i| push(i)))
                        }

                        match &tname[..] {
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
                                             "derive(Encodable) is deprecated \
                                              in favor of derive(RustcEncodable)");

                                expand!(encodable::expand_deriving_encodable)
                            }
                            "Decodable" => {
                                cx.span_warn(titem.span,
                                             "derive(Decodable) is deprecated \
                                              in favor of derive(RustcDecodable)");

                                expand!(decodable::expand_deriving_decodable)
                            }

                            "PartialEq" => expand!(eq::expand_deriving_eq),
                            "Eq" => expand!(totaleq::expand_deriving_totaleq),
                            "PartialOrd" => expand!(ord::expand_deriving_ord),
                            "Ord" => expand!(totalord::expand_deriving_totalord),

                            "Rand" => expand!(rand::expand_deriving_rand),

                            "Show" => {
                                cx.span_warn(titem.span,
                                             "derive(Show) is deprecated \
                                              in favor of derive(Debug)");

                                expand!(show::expand_deriving_show)
                            },

                            "Debug" => expand!(show::expand_deriving_show),

                            "Default" => expand!(default::expand_deriving_default),

                            "FromPrimitive" => expand!(primitive::expand_deriving_from_primitive),

                            "Send" => expand!(bounds::expand_deriving_bound),
                            "Sync" => expand!(bounds::expand_deriving_bound),
                            "Copy" => expand!(bounds::expand_deriving_bound),

                            ref tname => {
                                cx.span_err(titem.span,
                                            &format!("unknown `derive` \
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
