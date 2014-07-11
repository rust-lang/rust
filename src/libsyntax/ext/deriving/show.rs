// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{MetaItem, Item, Expr};
use codemap::Span;
use ext::format;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use parse::token;

use std::collections::HashMap;
use std::string::String;
use std::gc::Gc;

pub fn expand_deriving_show(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: Gc<MetaItem>,
                            item: Gc<Item>,
                            push: |Gc<Item>|) {
    // &mut ::std::fmt::Formatter
    let fmtr = Ptr(box Literal(Path::new(vec!("std", "fmt", "Formatter"))),
                   Borrowed(None, ast::MutMutable));

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "fmt", "Show")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "fmt",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(fmtr),
                ret_ty: Literal(Path::new(vec!("std", "fmt", "Result"))),
                attributes: Vec::new(),
                combine_substructure: combine_substructure(|a, b, c| {
                    show_substructure(a, b, c)
                })
            }
        )
    };
    trait_def.expand(cx, mitem, item, push)
}

/// We construct a format string and then defer to std::fmt, since that
/// knows what's up with formatting and so on.
fn show_substructure(cx: &mut ExtCtxt, span: Span,
                     substr: &Substructure) -> Gc<Expr> {
    // build `<name>`, `<name>({}, {}, ...)` or `<name> { <field>: {},
    // <field>: {}, ... }` based on the "shape".
    //
    // Easy start: they all start with the name.
    let name = match *substr.fields {
        Struct(_) => substr.type_ident,
        EnumMatching(_, v, _) => v.node.name,
        EnumNonMatchingCollapsed(..) | StaticStruct(..) | StaticEnum(..) => {
            cx.span_bug(span, "nonsensical .fields in `#[deriving(Show)]`")
        }
    };

    let mut format_string = String::from_str(token::get_ident(name).get());
    // the internal fields we're actually formatting
    let mut exprs = Vec::new();

    // Getting harder... making the format string:
    match *substr.fields {
        // unit struct/nullary variant: no work necessary!
        Struct(ref fields) if fields.len() == 0 => {}
        EnumMatching(_, _, ref fields) if fields.len() == 0 => {}

        Struct(ref fields) | EnumMatching(_, _, ref fields) => {
            if fields.get(0).name.is_none() {
                // tuple struct/"normal" variant

                format_string.push_str("(");

                for (i, field) in fields.iter().enumerate() {
                    if i != 0 { format_string.push_str(", "); }

                    format_string.push_str("{}");

                    exprs.push(field.self_);
                }

                format_string.push_str(")");
            } else {
                // normal struct/struct variant

                format_string.push_str(" {{");

                for (i, field) in fields.iter().enumerate() {
                    if i != 0 { format_string.push_str(","); }

                    let name = token::get_ident(field.name.unwrap());
                    format_string.push_str(" ");
                    format_string.push_str(name.get());
                    format_string.push_str(": {}");

                    exprs.push(field.self_);
                }

                format_string.push_str(" }}");
            }
        }
        _ => unreachable!()
    }

    // AST construction!
    // we're basically calling
    //
    // format_arg_method!(fmt, write_fmt, "<format_string>", exprs...)
    //
    // but doing it directly via ext::format.
    let formatter = substr.nonself_args[0];

    let meth = cx.ident_of("write_fmt");
    let s = token::intern_and_get_ident(format_string.as_slice());
    let format_string = cx.expr_str(span, s);

    // phew, not our responsibility any more!
    format::expand_preparsed_format_args(cx, span,
                                         format::MethodCall(formatter, meth),
                                         format_string, exprs, Vec::new(),
                                         HashMap::new())
}
