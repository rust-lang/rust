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
    fn parse_outer_attributes(&self) -> ~[ast::Attribute];
    fn parse_attribute(&self, style: ast::AttrStyle) -> ast::Attribute;
    fn parse_attribute_naked(
        &self,
        style: ast::AttrStyle,
        lo: BytePos
    ) -> ast::Attribute;
    fn parse_inner_attrs_and_next(&self) ->
        (~[ast::Attribute], ~[ast::Attribute]);
    fn parse_meta_item(&self) -> @ast::MetaItem;
    fn parse_meta_seq(&self) -> ~[@ast::MetaItem];
    fn parse_optional_meta(&self) -> ~[@ast::MetaItem];
}

impl parser_attr for Parser {

    // Parse attributes that appear before an item
    fn parse_outer_attributes(&self) -> ~[ast::Attribute] {
        let mut attrs: ~[ast::Attribute] = ~[];
        loop {
            match *self.token {
              token::POUND => {
                if self.look_ahead(1, |t| *t != token::LBRACKET) {
                    break;
                }
                attrs.push(self.parse_attribute(ast::AttrOuter));
              }
              token::DOC_COMMENT(s) => {
                let attr = ::attr::mk_sugared_doc_attr(
                    self.id_to_str(s),
                    self.span.lo,
                    self.span.hi
                );
                if attr.node.style != ast::AttrOuter {
                  self.fatal("expected outer comment");
                }
                attrs.push(attr);
                self.bump();
              }
              _ => break
            }
        }
        return attrs;
    }

    // matches attribute = # attribute_naked
    fn parse_attribute(&self, style: ast::AttrStyle) -> ast::Attribute {
        let lo = self.span.lo;
        self.expect(&token::POUND);
        return self.parse_attribute_naked(style, lo);
    }

    // matches attribute_naked = [ meta_item ]
    fn parse_attribute_naked(&self, style: ast::AttrStyle, lo: BytePos) ->
        ast::Attribute {
        self.expect(&token::LBRACKET);
        let meta_item = self.parse_meta_item();
        self.expect(&token::RBRACKET);
        let hi = self.span.hi;
        return spanned(lo, hi, ast::Attribute_ { style: style,
                                                 value: meta_item, is_sugared_doc: false }); }

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
        (~[ast::Attribute], ~[ast::Attribute]) {
        let mut inner_attrs: ~[ast::Attribute] = ~[];
        let mut next_outer_attrs: ~[ast::Attribute] = ~[];
        loop {
            match *self.token {
              token::POUND => {
                if self.look_ahead(1, |t| *t != token::LBRACKET) {
                    // This is an extension
                    break;
                }
                let attr = self.parse_attribute(ast::AttrInner);
                if *self.token == token::SEMI {
                    self.bump();
                    inner_attrs.push(attr);
                } else {
                    // It's not really an inner attribute
                    let outer_attr =
                        spanned(attr.span.lo, attr.span.hi,
                            ast::Attribute_ { style: ast::AttrOuter,
                                              value: attr.node.value,
                                              is_sugared_doc: false });
                    next_outer_attrs.push(outer_attr);
                    break;
                }
              }
              token::DOC_COMMENT(s) => {
                let attr = ::attr::mk_sugared_doc_attr(
                    self.id_to_str(s),
                    self.span.lo,
                    self.span.hi
                );
                self.bump();
                if attr.node.style == ast::AttrInner {
                  inner_attrs.push(attr);
                } else {
                  next_outer_attrs.push(attr);
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
    fn parse_meta_item(&self) -> @ast::MetaItem {
        let lo = self.span.lo;
        let name = self.id_to_str(self.parse_ident());
        match *self.token {
            token::EQ => {
                self.bump();
                let lit = self.parse_lit();
                let hi = self.span.hi;
                @spanned(lo, hi, ast::MetaNameValue(name, lit))
            }
            token::LPAREN => {
                let inner_items = self.parse_meta_seq();
                let hi = self.span.hi;
                @spanned(lo, hi, ast::MetaList(name, inner_items))
            }
            _ => {
                let hi = self.last_span.hi;
                @spanned(lo, hi, ast::MetaWord(name))
            }
        }
    }

    // matches meta_seq = ( COMMASEP(meta_item) )
    fn parse_meta_seq(&self) -> ~[@ast::MetaItem] {
        self.parse_seq(&token::LPAREN,
                       &token::RPAREN,
                       seq_sep_trailing_disallowed(token::COMMA),
                       |p| p.parse_meta_item()).node
    }

    fn parse_optional_meta(&self) -> ~[@ast::MetaItem] {
        match *self.token {
            token::LPAREN => self.parse_meta_seq(),
            _ => ~[]
        }
    }
}
