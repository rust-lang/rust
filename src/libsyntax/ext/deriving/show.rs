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

use parse::token;

use collections::HashMap;

pub fn expand_deriving_show(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: @MetaItem,
                            item: @Item,
                            push: |@Item|) {
    // &mut ::std::fmt::Formatter
    let fmtr = Ptr(~Literal(Path::new(~["std", "fmt", "Formatter"])),
                   Borrowed(None, ast::MutMutable));

    let trait_def = TraitDef {
        span: span,
        attributes: ~[],
        path: Path::new(~["std", "fmt", "Show"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "fmt",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[fmtr],
                ret_ty: Literal(Path::new(~["std", "fmt", "Result"])),
                inline: false,
                const_nonmatching: false,
                combine_substructure: show_substructure
            }
        ]
    };
    trait_def.expand(cx, mitem, item, push)
}

// we construct a format string and then defer to std::fmt, since that
// knows what's up with formatting at so on.
fn show_substructure(cx: &mut ExtCtxt, span: Span,
                     substr: &Substructure) -> @Expr {
    // build `<name>`, `<name>({}, {}, ...)` or `<name> { <field>: {},
    // <field>: {}, ... }` based on the "shape".
    //
    // Easy start: they all start with the name.
    let name = match *substr.fields {
        Struct(_) => substr.type_ident,
        EnumMatching(_, v, _) => v.node.name,

        EnumNonMatching(..) | StaticStruct(..) | StaticEnum(..) => {
            cx.span_bug(span, "nonsensical .fields in `#[deriving(Show)]`")
        }
    };

    let mut format_string = token::get_ident(name).get().to_owned();
    // the internal fields we're actually formatting
    let mut exprs = ~[];

    // Getting harder... making the format string:
    match *substr.fields {
        // unit struct/nullary variant: no work necessary!
        Struct(ref fields) if fields.len() == 0 => {}
        EnumMatching(_, _, ref fields) if fields.len() == 0 => {}

        Struct(ref fields) | EnumMatching(_, _, ref fields) => {
            if fields[0].name.is_none() {
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

                format_string.push_str(" \\{");

                for (i, field) in fields.iter().enumerate() {
                    if i != 0 { format_string.push_str(","); }

                    let name = token::get_ident(field.name.unwrap());
                    format_string.push_str(" ");
                    format_string.push_str(name.get());
                    format_string.push_str(": {}");

                    exprs.push(field.self_);
                }

                format_string.push_str(" \\}");
            }
        }
        _ => unreachable!()
    }

    // AST construction!
    // we're basically calling
    //
    // format_arg!(|__args| ::std::fmt::write(fmt.buf, __args), "<format_string>", exprs...)
    //
    // but doing it directly via ext::format.
    let formatter = substr.nonself_args[0];
    let buf = cx.expr_field_access(span, formatter, cx.ident_of("buf"));

    let std_write = ~[cx.ident_of("std"), cx.ident_of("fmt"), cx.ident_of("write")];
    let args = cx.ident_of("__args");
    let write_call = cx.expr_call_global(span, std_write, ~[buf, cx.expr_ident(span, args)]);
    let format_closure = cx.lambda_expr(span, ~[args], write_call);

    let s = token::intern_and_get_ident(format_string);
    let format_string = cx.expr_str(span, s);

    // phew, not our responsibility any more!
    format::expand_preparsed_format_args(cx, span,
                                         format_closure,
                                         format_string, exprs, HashMap::new())
}
