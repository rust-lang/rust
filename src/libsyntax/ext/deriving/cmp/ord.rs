// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use ast::{meta_item, item, expr_if, expr};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_ord(cx: @ExtCtxt,
                           span: span,
                           mitem: @meta_item,
                           in_items: ~[@item]) -> ~[@item] {
    macro_rules! md (
        ($name:expr, $less:expr, $equal:expr) => {
            MethodDef {
                name: $name,
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[borrowed_self()],
                ret_ty: Literal(Path::new(~["bool"])),
                const_nonmatching: false,
                combine_substructure: |cx, span, substr|
                    cs_ord($less, $equal, cx, span, substr)
            }
        }
    );



    let trait_def = TraitDef {
        path: Path::new(~["core", "cmp", "Ord"]),
        // XXX: Ord doesn't imply Eq yet
        additional_bounds: ~[Literal(Path::new(~["core", "cmp", "Eq"]))],
        generics: LifetimeBounds::empty(),
        methods: ~[
            md!("lt", true,  false),
            md!("le", true,  true),
            md!("gt", false, false),
            md!("ge", false, true)
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items,
                            &trait_def)
}

/// `less`: is this `lt` or `le`? `equal`: is this `le` or `ge`?
fn cs_ord(less: bool, equal: bool,
          cx: @ExtCtxt, span: span,
          substr: &Substructure) -> @expr {
    let binop = if less {
        cx.ident_of("lt")
    } else {
        cx.ident_of("gt")
    };
    let false_blk_expr = cx.mk_block(span,
                                         ~[], ~[],
                                         Some(cx.mk_bool(span, false)));
    let base = cx.mk_bool(span, equal);

    cs_fold(
        false, // need foldr,
        |cx, span, subexpr, self_f, other_fs| {
            /*

            build up a series of nested ifs from the inside out to get
            lexical ordering (hence foldr), i.e.

            ```
            if self.f1 `binop` other.f1 {
                true
            } else if self.f1 == other.f1 {
                if self.f2 `binop` other.f2 {
                    true
                } else if self.f2 == other.f2 {
                    `equal`
                } else {
                    false
                }
            } else {
                false
            }
            ```

            The inner "`equal`" case is only reached if the two
            items have all fields equal.
            */
            if other_fs.len() != 1 {
                cx.span_bug(span, "Not exactly 2 arguments in `deriving(Ord)`");
            }

            let cmp = cx.mk_method_call(span,
                                            self_f, cx.ident_of("eq"), other_fs.to_owned());
            let subexpr = cx.mk_simple_block(span, subexpr);
            let elseif = expr_if(cmp, subexpr, Some(false_blk_expr));
            let elseif = cx.mk_expr(span, elseif);

            let cmp = cx.mk_method_call(span,
                                            self_f, binop, other_fs.to_owned());
            let true_blk = cx.mk_simple_block(span,
                                                  cx.mk_bool(span, true));
            let if_ = expr_if(cmp, true_blk, Some(elseif));

            cx.mk_expr(span, if_)
        },
        base,
        |cx, span, args, _| {
            // nonmatching enums, order by the order the variants are
            // written
            match args {
                [(self_var, _, _),
                 (other_var, _, _)] =>
                    cx.mk_bool(span,
                                   if less {
                                       self_var < other_var
                                   } else {
                                       self_var > other_var
                                   }),
                _ => cx.span_bug(span, "Not exactly 2 arguments in `deriving(Ord)`")
            }
        },
        cx, span, substr)
}
