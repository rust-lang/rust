// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::spanned;
use codemap::BytePos;
use parse::common::*; //resolve bug?
use parse::token;
use parse::parser::Parser;

// a parser that can parse attributes.
pub trait parser_attr {
    fn parse_outer_attributes(&self) -> ~[ast::attribute];
    fn parse_attribute(&self, style: ast::attr_style) -> ast::attribute;
    fn parse_attribute_naked(
        &self,
        style: ast::attr_style,
        lo: BytePos
    ) -> ast::attribute;
    fn parse_inner_attrs_and_next(&self) ->
        (~[ast::attribute], ~[ast::attribute]);
    fn parse_meta_item(&self) -> @ast::meta_item;
    fn parse_meta_seq(&self) -> ~[@ast::meta_item];
    fn parse_optional_meta(&self) -> ~[@ast::meta_item];
}

impl parser_attr for Parser {

    // Parse attributes that appear before an item
    fn parse_outer_attributes(&self) -> ~[ast::attribute] {
        let mut attrs: ~[ast::attribute] = ~[];
        loop {
            match *self.token {
              token::POUND => {
                if self.look_ahead(1u) != token::LBRACKET {
                    break;
                }
                attrs += ~[self.parse_attribute(ast::attr_outer)];
              }
              token::DOC_COMMENT(s) => {
                let attr = ::attr::mk_sugared_doc_attr(
                    copy *self.id_to_str(s),
                    self.span.lo,
                    self.span.hi
                );
                if attr.node.style != ast::attr_outer {
                  self.fatal("expected outer comment");
                }
                attrs += ~[attr];
                self.bump();
              }
              _ => break
            }
        }
        return attrs;
    }

    // matches attribute = # attribute_naked
    fn parse_attribute(&self, style: ast::attr_style) -> ast::attribute {
        let lo = self.span.lo;
        self.expect(&token::POUND);
        return self.parse_attribute_naked(style, lo);
    }

    // matches attribute_naked = [ meta_item ]
    fn parse_attribute_naked(&self, style: ast::attr_style, lo: BytePos) ->
        ast::attribute {
        self.expect(&token::LBRACKET);
        let meta_item = self.parse_meta_item();
        self.expect(&token::RBRACKET);
        let hi = self.span.hi;
        return spanned(lo, hi, ast::attribute_ { style: style,
                                                 value: meta_item,
                                                 is_sugared_doc: false });
    }

    // Parse attributes that appear after the opening of an item, each
    // terminated by a semicolon. In addition to a vector of inner attributes,
    // this function also returns a vector that may contain the first outer
    // attribute of the next item (since we can't know whether the attribute
    // is an inner attribute of the containing item or an outer attribute of
    // the first contained item until we see the semi).

    // matches inner_attrs* outer_attr?
    // you can make the 'next' field an Option, but the result is going to be
    // more useful as a vector.
    fn parse_inner_attrs_and_next(&self) ->
        (~[ast::attribute], ~[ast::attribute]) {
        let mut inner_attrs: ~[ast::attribute] = ~[];
        let mut next_outer_attrs: ~[ast::attribute] = ~[];
        loop {
            match *self.token {
              token::POUND => {
                if self.look_ahead(1u) != token::LBRACKET {
                    // This is an extension
                    break;
                }
                let attr = self.parse_attribute(ast::attr_inner);
                if *self.token == token::SEMI {
                    self.bump();
                    inner_attrs += ~[attr];
                } else {
                    // It's not really an inner attribute
                    let outer_attr =
                        spanned(attr.span.lo, attr.span.hi,
                            ast::attribute_ { style: ast::attr_outer,
                                              value: attr.node.value,
                                              is_sugared_doc: false });
                    next_outer_attrs += ~[outer_attr];
                    break;
                }
              }
              token::DOC_COMMENT(s) => {
                let attr = ::attr::mk_sugared_doc_attr(
                    copy *self.id_to_str(s),
                    self.span.lo,
                    self.span.hi
                );
                self.bump();
                if attr.node.style == ast::attr_inner {
                  inner_attrs += ~[attr];
                } else {
                  next_outer_attrs += ~[attr];
                  break;
                }
              }
              _ => break
            }
        }
        (inner_attrs, next_outer_attrs)
    }

    // matches meta_item = IDENT
    // | IDENT = lit
    // | IDENT meta_seq
    fn parse_meta_item(&self) -> @ast::meta_item {
        let lo = self.span.lo;
        let name = self.id_to_str(self.parse_ident());
        match *self.token {
            token::EQ => {
                self.bump();
                let lit = self.parse_lit();
                let hi = self.span.hi;
                @spanned(lo, hi, ast::meta_name_value(name, lit))
            }
            token::LPAREN => {
                let inner_items = self.parse_meta_seq();
                let hi = self.span.hi;
                @spanned(lo, hi, ast::meta_list(name, inner_items))
            }
            _ => {
                let hi = self.last_span.hi;
                @spanned(lo, hi, ast::meta_word(name))
            }
        }
    }

    // matches meta_seq = ( COMMASEP(meta_item) )
    fn parse_meta_seq(&self) -> ~[@ast::meta_item] {
        copy self.parse_seq(
            &token::LPAREN,
            &token::RPAREN,
            seq_sep_trailing_disallowed(token::COMMA),
            |p| p.parse_meta_item()
        ).node
    }

    fn parse_optional_meta(&self) -> ~[@ast::meta_item] {
        match *self.token {
            token::LPAREN => self.parse_meta_seq(),
            _ => ~[]
        }
    }
}
