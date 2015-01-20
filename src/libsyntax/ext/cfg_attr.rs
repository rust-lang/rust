// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, TraitItem, ImplItem};
use attr;
use codemap::Span;
use ext::base::{Annotatable, ExtCtxt};
use ext::build::AstBuilder;
use ptr::P;

macro_rules! fold_annotatable {
    ($ann:expr, $item:ident => $oper:expr) => (
        match $ann {
            Annotatable::Item(it) => {
                let mut $item = (*it).clone();
                $oper;
                Annotatable::Item(P($item))
            }
            Annotatable::TraitItem(it) => {
                match it {
                    TraitItem::RequiredMethod(mut $item) => {
                        $oper;
                        Annotatable::TraitItem(TraitItem::RequiredMethod($item))
                    }
                    TraitItem::ProvidedMethod(pm) => {
                        let mut $item = (*pm).clone();
                        $oper;
                        Annotatable::TraitItem(TraitItem::ProvidedMethod(P($item)))
                    }
                    TraitItem::TypeTraitItem(at) => {
                        let mut $item = (*at).clone();
                        $oper;
                        Annotatable::TraitItem(TraitItem::TypeTraitItem(P($item)))
                    }
                }
            }
            Annotatable::ImplItem(it) => {
                match it {
                    ImplItem::MethodImplItem(pm) => {
                        let mut $item = (*pm).clone();
                        $oper;
                        Annotatable::ImplItem(ImplItem::MethodImplItem(P($item)))
                    }
                    ImplItem::TypeImplItem(at) => {
                        let mut $item = (*at).clone();
                        $oper;
                        Annotatable::ImplItem(ImplItem::TypeImplItem(P($item)))
                    }
                }
            }
        }
    );
}

pub fn expand(cx: &mut ExtCtxt, sp: Span, mi: &ast::MetaItem, ann: Annotatable) -> Annotatable {
    let (cfg, attr) = match mi.node {
        ast::MetaList(_, ref mis) if mis.len() == 2 => (&mis[0], &mis[1]),
        _ => {
            cx.span_err(sp, "expected `#[cfg_attr(<cfg pattern>, <attr>)]`");
            return ann;
        }
    };

    if attr::cfg_matches(&cx.parse_sess.span_diagnostic, cx.cfg.as_slice(), &**cfg) {
        let attr = cx.attribute(attr.span, attr.clone());
        fold_annotatable!(ann, item => item.attrs.push(attr))
    } else {
        ann
    }
}
