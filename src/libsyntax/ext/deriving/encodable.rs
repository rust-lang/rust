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

The compiler code necessary to implement the #[deriving(Encodable)]
(and Decodable, in decodable.rs) extension.  The idea here is that
type-defining items may be tagged with #[deriving(Encodable,
Decodable)].

For example, a type like:

    #[deriving(Encodable, Decodable)]
    struct Node {id: uint}

would generate two implementations like:

impl<S:serialize::Encoder> Encodable<S> for Node {
    fn encode(&self, s: &S) {
        s.emit_struct("Node", 1, || {
            s.emit_field("id", 0, || s.emit_uint(self.id))
        })
    }
}

impl<D:Decoder> Decodable for node_id {
    fn decode(d: &D) -> Node {
        d.read_struct("Node", 1, || {
            Node {
                id: d.read_field(~"x", 0, || decode(d))
            }
        })
    }
}

Other interesting scenarios are whe the item has type parameters or
references other non-built-in types.  A type definition like:

    #[deriving(Encodable, Decodable)]
    struct spanned<T> {node: T, span: Span}

would yield functions like:

    impl<
        S: Encoder,
        T: Encodable<S>
    > spanned<T>: Encodable<S> {
        fn encode<S:Encoder>(s: &S) {
            s.emit_rec(|| {
                s.emit_field("node", 0, || self.node.encode(s));
                s.emit_field("span", 1, || self.span.encode(s));
            })
        }
    }

    impl<
        D: Decoder,
        T: Decodable<D>
    > spanned<T>: Decodable<D> {
        fn decode(d: &D) -> spanned<T> {
            d.read_rec(|| {
                {
                    node: d.read_field(~"node", 0, || decode(d)),
                    span: d.read_field(~"span", 1, || decode(d)),
                }
            })
        }
    }
*/

use ast::{MetaItem, Item, Expr, MutMutable};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use parse::token;

pub fn expand_deriving_encodable(cx: &mut ExtCtxt,
                                 span: Span,
                                 mitem: @MetaItem,
                                 in_items: ~[@Item]) -> ~[@Item] {
    let trait_def = TraitDef {
        span: span,
        path: Path::new_(~["serialize", "Encodable"], None,
                         ~[~Literal(Path::new_local("__E"))], true),
        additional_bounds: ~[],
        generics: LifetimeBounds {
            lifetimes: ~[],
            bounds: ~[("__E", ~[Path::new(~["serialize", "Encoder"])])],
        },
        methods: ~[
            MethodDef {
                name: "encode",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[Ptr(~Literal(Path::new_local("__E")),
                            Borrowed(None, MutMutable))],
                ret_ty: nil_ty(),
                inline: false,
                const_nonmatching: true,
                combine_substructure: encodable_substructure,
            },
        ]
    };

    trait_def.expand(cx, mitem, in_items)
}

fn encodable_substructure(cx: &mut ExtCtxt, trait_span: Span,
                          substr: &Substructure) -> @Expr {
    let encoder = substr.nonself_args[0];
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = cx.ident_of("_e");
    let blkencoder = cx.expr_ident(trait_span, blkarg);
    let encode = cx.ident_of("encode");

    return match *substr.fields {
        Struct(ref fields) => {
            let emit_struct_field = cx.ident_of("emit_struct_field");
            let mut stmts = ~[];
            for (i, &FieldInfo {
                    name,
                    self_,
                    span,
                    ..
                }) in fields.iter().enumerate() {
                let name = match name {
                    Some(id) => token::get_ident(id.name),
                    None => {
                        token::intern_and_get_ident(format!("_field{}", i))
                    }
                };
                let enc = cx.expr_method_call(span, self_, encode, ~[blkencoder]);
                let lambda = cx.lambda_expr_1(span, enc, blkarg);
                let call = cx.expr_method_call(span, blkencoder,
                                               emit_struct_field,
                                               ~[cx.expr_str(span, name),
                                                 cx.expr_uint(span, i),
                                                 lambda]);
                stmts.push(cx.stmt_expr(call));
            }

            let blk = cx.lambda_stmts_1(trait_span, stmts, blkarg);
            cx.expr_method_call(trait_span,
                                encoder,
                                cx.ident_of("emit_struct"),
                                ~[
                cx.expr_str(trait_span,
                            token::get_ident(substr.type_ident.name)),
                cx.expr_uint(trait_span, fields.len()),
                blk
            ])
        }

        EnumMatching(idx, variant, ref fields) => {
            // We're not generating an AST that the borrow checker is expecting,
            // so we need to generate a unique local variable to take the
            // mutable loan out on, otherwise we get conflicts which don't
            // actually exist.
            let me = cx.stmt_let(trait_span, false, blkarg, encoder);
            let encoder = cx.expr_ident(trait_span, blkarg);
            let emit_variant_arg = cx.ident_of("emit_enum_variant_arg");
            let mut stmts = ~[];
            for (i, &FieldInfo { self_, span, .. }) in fields.iter().enumerate() {
                let enc = cx.expr_method_call(span, self_, encode, ~[blkencoder]);
                let lambda = cx.lambda_expr_1(span, enc, blkarg);
                let call = cx.expr_method_call(span, blkencoder,
                                               emit_variant_arg,
                                               ~[cx.expr_uint(span, i),
                                                 lambda]);
                stmts.push(cx.stmt_expr(call));
            }

            let blk = cx.lambda_stmts_1(trait_span, stmts, blkarg);
            let name = cx.expr_str(trait_span,
                                   token::get_ident(variant.node.name.name));
            let call = cx.expr_method_call(trait_span, blkencoder,
                                           cx.ident_of("emit_enum_variant"),
                                           ~[name,
                                             cx.expr_uint(trait_span, idx),
                                             cx.expr_uint(trait_span, fields.len()),
                                             blk]);
            let blk = cx.lambda_expr_1(trait_span, call, blkarg);
            let ret = cx.expr_method_call(trait_span,
                                          encoder,
                                          cx.ident_of("emit_enum"),
                                          ~[
                cx.expr_str(trait_span,
                            token::get_ident(substr.type_ident.name)),
                blk
            ]);
            cx.expr_block(cx.block(trait_span, ~[me], Some(ret)))
        }

        _ => cx.bug("expected Struct or EnumMatching in deriving(Encodable)")
    };
}
