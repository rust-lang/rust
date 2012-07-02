import either::{either, left, right};
import ast_util::spanned;
import common::{parser_common, seq_sep_trailing_disallowed};

export attr_or_ext;
export parser_attr;

// A type to distingush between the parsing of item attributes or syntax
// extensions, which both begin with token.POUND
type attr_or_ext = option<either<~[ast::attribute], @ast::expr>>;

impl parser_attr for parser {

    fn parse_outer_attrs_or_ext(first_item_attrs: ~[ast::attribute])
        -> attr_or_ext
    {
        let expect_item_next = vec::is_not_empty(first_item_attrs);
        alt self.token {
          token::POUND {
            let lo = self.span.lo;
            if self.look_ahead(1u) == token::LBRACKET {
                self.bump();
                let first_attr =
                    self.parse_attribute_naked(ast::attr_outer, lo);
                ret some(left(vec::append(~[first_attr],
                                          self.parse_outer_attributes())));
            } else if !(self.look_ahead(1u) == token::LT
                        || self.look_ahead(1u) == token::LBRACKET
                        || self.look_ahead(1u) == token::POUND
                        || expect_item_next) {
                self.bump();
                ret some(right(self.parse_syntax_ext_naked(lo)));
            } else { ret none; }
        }
        token::DOC_COMMENT(_) {
          ret some(left(self.parse_outer_attributes()));
        }
        _ {
          ret none;
        }
      }
    }

    // Parse attributes that appear before an item
    fn parse_outer_attributes() -> ~[ast::attribute] {
        let mut attrs: ~[ast::attribute] = ~[];
        loop {
            alt copy self.token {
              token::POUND {
                if self.look_ahead(1u) != token::LBRACKET {
                    break;
                }
                attrs += [self.parse_attribute(ast::attr_outer)]/~;
              }
              token::DOC_COMMENT(s) {
                let attr = ::attr::mk_sugared_doc_attr(
                        *self.get_str(s), self.span.lo, self.span.hi);
                if attr.node.style != ast::attr_outer {
                  self.fatal("expected outer comment");
                }
                attrs += [attr]/~;
                self.bump();
              }
              _ {
                break;
              }
            }
        }
        ret attrs;
    }

    fn parse_attribute(style: ast::attr_style) -> ast::attribute {
        let lo = self.span.lo;
        self.expect(token::POUND);
        ret self.parse_attribute_naked(style, lo);
    }

    fn parse_attribute_naked(style: ast::attr_style, lo: uint) ->
        ast::attribute {
        self.expect(token::LBRACKET);
        let meta_item = self.parse_meta_item();
        self.expect(token::RBRACKET);
        let mut hi = self.span.hi;
        ret spanned(lo, hi, {style: style, value: *meta_item,
                             is_sugared_doc: false});
    }

    // Parse attributes that appear after the opening of an item, each
    // terminated by a semicolon. In addition to a vector of inner attributes,
    // this function also returns a vector that may contain the first outer
    // attribute of the next item (since we can't know whether the attribute
    // is an inner attribute of the containing item or an outer attribute of
    // the first contained item until we see the semi).
    fn parse_inner_attrs_and_next() ->
        {inner: ~[ast::attribute], next: ~[ast::attribute]} {
        let mut inner_attrs: ~[ast::attribute] = ~[];
        let mut next_outer_attrs: ~[ast::attribute] = ~[];
        loop {
            alt copy self.token {
              token::POUND {
                if self.look_ahead(1u) != token::LBRACKET {
                    // This is an extension
                    break;
                }
                let attr = self.parse_attribute(ast::attr_inner);
                if self.token == token::SEMI {
                    self.bump();
                    inner_attrs += [attr]/~;
                } else {
                    // It's not really an inner attribute
                    let outer_attr =
                        spanned(attr.span.lo, attr.span.hi,
                            {style: ast::attr_outer, value: attr.node.value,
                             is_sugared_doc: false});
                    next_outer_attrs += [outer_attr]/~;
                    break;
                }
              }
              token::DOC_COMMENT(s) {
                let attr = ::attr::mk_sugared_doc_attr(
                        *self.get_str(s), self.span.lo, self.span.hi);
                self.bump();
                if attr.node.style == ast::attr_inner {
                  inner_attrs += [attr]/~;
                } else {
                  next_outer_attrs += [attr]/~;
                  break;
                }
              }
              _ {
                break;
              }
            }
        }
        ret {inner: inner_attrs, next: next_outer_attrs};
    }

    fn parse_meta_item() -> @ast::meta_item {
        let lo = self.span.lo;
        let ident = self.parse_ident();
        alt self.token {
          token::EQ {
            self.bump();
            let lit = self.parse_lit();
            let mut hi = self.span.hi;
            ret @spanned(lo, hi, ast::meta_name_value(ident, lit));
          }
          token::LPAREN {
            let inner_items = self.parse_meta_seq();
            let mut hi = self.span.hi;
            ret @spanned(lo, hi, ast::meta_list(ident, inner_items));
          }
          _ {
            let mut hi = self.span.hi;
            ret @spanned(lo, hi, ast::meta_word(ident));
          }
        }
    }

    fn parse_meta_seq() -> ~[@ast::meta_item] {
        ret self.parse_seq(token::LPAREN, token::RPAREN,
                           seq_sep_trailing_disallowed(token::COMMA),
                           |p| p.parse_meta_item()).node;
    }

    fn parse_optional_meta() -> ~[@ast::meta_item] {
        alt self.token { token::LPAREN { ret self.parse_meta_seq(); }
                         _ { ret ~[]; } }
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
