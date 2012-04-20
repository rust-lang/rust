import either::{either, left, right};
import common::{parse_seq,
                seq_sep,
                expect,
                parse_ident,
                spanned};
import parser::{parse_lit,
                parse_syntax_ext_naked};

export attr_or_ext;
export parse_outer_attributes;
export parse_outer_attrs_or_ext;
export parse_inner_attrs_and_next;
export parse_optional_meta;

// A type to distingush between the parsing of item attributes or syntax
// extensions, which both begin with token.POUND
type attr_or_ext = option<either<[ast::attribute], @ast::expr>>;

fn parse_outer_attrs_or_ext(
    p: parser,
    first_item_attrs: [ast::attribute]) -> attr_or_ext {
    let expect_item_next = vec::is_not_empty(first_item_attrs);
    if p.token == token::POUND {
        let lo = p.span.lo;
        if p.look_ahead(1u) == token::LBRACKET {
            p.bump();
            let first_attr = parse_attribute_naked(p, ast::attr_outer, lo);
            ret some(left([first_attr] + parse_outer_attributes(p)));
        } else if !(p.look_ahead(1u) == token::LT
                    || p.look_ahead(1u) == token::LBRACKET
                    || expect_item_next) {
            p.bump();
            ret some(right(parse_syntax_ext_naked(p, lo)));
        } else { ret none; }
    } else { ret none; }
}

// Parse attributes that appear before an item
fn parse_outer_attributes(p: parser) -> [ast::attribute] {
    let mut attrs: [ast::attribute] = [];
    while p.token == token::POUND {
        attrs += [parse_attribute(p, ast::attr_outer)];
    }
    ret attrs;
}

fn parse_attribute(p: parser, style: ast::attr_style) -> ast::attribute {
    let lo = p.span.lo;
    expect(p, token::POUND);
    ret parse_attribute_naked(p, style, lo);
}

fn parse_attribute_naked(p: parser, style: ast::attr_style, lo: uint) ->
   ast::attribute {
    expect(p, token::LBRACKET);
    let meta_item = parse_meta_item(p);
    expect(p, token::RBRACKET);
    let mut hi = p.span.hi;
    ret spanned(lo, hi, {style: style, value: *meta_item});
}

// Parse attributes that appear after the opening of an item, each terminated
// by a semicolon. In addition to a vector of inner attributes, this function
// also returns a vector that may contain the first outer attribute of the
// next item (since we can't know whether the attribute is an inner attribute
// of the containing item or an outer attribute of the first contained item
// until we see the semi).
fn parse_inner_attrs_and_next(p: parser) ->
   {inner: [ast::attribute], next: [ast::attribute]} {
    let mut inner_attrs: [ast::attribute] = [];
    let mut next_outer_attrs: [ast::attribute] = [];
    while p.token == token::POUND {
        if p.look_ahead(1u) != token::LBRACKET {
            // This is an extension
            break;
        }
        let attr = parse_attribute(p, ast::attr_inner);
        if p.token == token::SEMI {
            p.bump();
            inner_attrs += [attr];
        } else {
            // It's not really an inner attribute
            let outer_attr =
                spanned(attr.span.lo, attr.span.hi,
                        {style: ast::attr_outer, value: attr.node.value});
            next_outer_attrs += [outer_attr];
            break;
        }
    }
    ret {inner: inner_attrs, next: next_outer_attrs};
}

fn parse_meta_item(p: parser) -> @ast::meta_item {
    let lo = p.span.lo;
    let ident = parse_ident(p);
    alt p.token {
      token::EQ {
        p.bump();
        let lit = parse_lit(p);
        let mut hi = p.span.hi;
        ret @spanned(lo, hi, ast::meta_name_value(ident, lit));
      }
      token::LPAREN {
        let inner_items = parse_meta_seq(p);
        let mut hi = p.span.hi;
        ret @spanned(lo, hi, ast::meta_list(ident, inner_items));
      }
      _ {
        let mut hi = p.span.hi;
        ret @spanned(lo, hi, ast::meta_word(ident));
      }
    }
}

fn parse_meta_seq(p: parser) -> [@ast::meta_item] {
    ret parse_seq(token::LPAREN, token::RPAREN, seq_sep(token::COMMA),
                  parse_meta_item, p).node;
}

fn parse_optional_meta(p: parser) -> [@ast::meta_item] {
    alt p.token { token::LPAREN { ret parse_meta_seq(p); } _ { ret []; } }
}
