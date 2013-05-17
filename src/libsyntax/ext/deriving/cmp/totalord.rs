// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast::{meta_item, item, expr};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use core::cmp::{Ordering, Equal, Less, Greater};

pub fn expand_deriving_totalord(cx: @ExtCtxt,
                                span: span,
                                mitem: @meta_item,
                                in_items: ~[@item]) -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["core", "cmp", "TotalOrd"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "cmp",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[borrowed_self()],
                ret_ty: Literal(Path::new(~["core", "cmp", "Ordering"])),
                const_nonmatching: false,
                combine_substructure: cs_cmp
            }
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items,
                            &trait_def)
}


pub fn ordering_const(cx: @ExtCtxt, span: span, cnst: Ordering) -> @expr {
    let cnst = match cnst {
        Less => "Less",
        Equal => "Equal",
        Greater => "Greater"
    };
    cx.expr_path(
        cx.path_global(span,
                       ~[cx.ident_of("core"),
                         cx.ident_of("cmp"),
                         cx.ident_of(cnst)]))
}

pub fn cs_cmp(cx: @ExtCtxt, span: span,
              substr: &Substructure) -> @expr {

    cs_same_method_fold(
        // foldr (possibly) nests the matches in lexical_ordering better
        false,
        |cx, span, old, new| {
            cx.expr_call_global(span,
                                ~[cx.ident_of("core"),
                                  cx.ident_of("cmp"),
                                  cx.ident_of("lexical_ordering")],
                                ~[old, new])
        },
        ordering_const(cx, span, Equal),
        |cx, span, list, _| {
            match list {
                // an earlier nonmatching variant is Less than a
                // later one
                [(self_var, _, _),
                 (other_var, _, _)] => ordering_const(cx, span,
                                                      self_var.cmp(&other_var)),
                _ => cx.span_bug(span, "Not exactly 2 arguments in `deriving(TotalOrd)`")
            }
        },
        cx, span, substr)
}
