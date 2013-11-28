// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{MetaItem, item, Expr};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_to_str(cx: @ExtCtxt,
                              span: Span,
                              mitem: @MetaItem,
                              in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["std", "to_str", "ToStr"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "to_str",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Ptr(~Literal(Path::new_local("str")), Send),
                inline: false,
                const_nonmatching: false,
                combine_substructure: to_str_substructure
            }
        ]
    };
    trait_def.expand(cx, span, mitem, in_items)
}

// It used to be the case that this deriving implementation invoked
// std::repr::repr_to_str, but this isn't sufficient because it
// doesn't invoke the to_str() method on each field. Hence we mirror
// the logic of the repr_to_str() method, but with tweaks to call to_str()
// on sub-fields.
fn to_str_substructure(cx: @ExtCtxt, span: Span,
                       substr: &Substructure) -> @Expr {
    let to_str = cx.ident_of("to_str");

    let doit = |start: &str, end: @str, name: ast::Ident,
                fields: &[FieldInfo]| {
        if fields.len() == 0 {
            cx.expr_str_uniq(span, cx.str_of(name))
        } else {
            let buf = cx.ident_of("buf");
            let start = cx.str_of(name) + start;
            let init = cx.expr_str_uniq(span, start.to_managed());
            let mut stmts = ~[cx.stmt_let(span, true, buf, init)];
            let push_str = cx.ident_of("push_str");

            let push = |s: @Expr| {
                let ebuf = cx.expr_ident(span, buf);
                let call = cx.expr_method_call(span, ebuf, push_str, ~[s]);
                stmts.push(cx.stmt_expr(call));
            };

            for (i, &FieldInfo {name, span, self_, _}) in fields.iter().enumerate() {
                if i > 0 {
                    push(cx.expr_str(span, @", "));
                }
                match name {
                    None => {}
                    Some(id) => {
                        let name = cx.str_of(id) + ": ";
                        push(cx.expr_str(span, name.to_managed()));
                    }
                }
                push(cx.expr_method_call(span, self_, to_str, ~[]));
            }
            push(cx.expr_str(span, end));

            cx.expr_block(cx.block(span, stmts, Some(cx.expr_ident(span, buf))))
        }
    };

    return match *substr.fields {
        Struct(ref fields) => {
            if fields.len() == 0 || fields[0].name.is_none() {
                doit("(", @")", substr.type_ident, *fields)
            } else {
                doit("{", @"}", substr.type_ident, *fields)
            }
        }

        EnumMatching(_, variant, ref fields) => {
            match variant.node.kind {
                ast::tuple_variant_kind(*) =>
                    doit("(", @")", variant.node.name, *fields),
                ast::struct_variant_kind(*) =>
                    doit("{", @"}", variant.node.name, *fields),
            }
        }

        _ => cx.bug("expected Struct or EnumMatching in deriving(ToStr)")
    };
}
