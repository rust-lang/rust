// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Parsing pipes protocols from token trees.

use ast_util;
use ext::pipes::proto::*;
use parse::common::SeqSep;
use parse::parser;
use parse::token;

use core::prelude::*;

pub trait proto_parser {
    fn parse_proto(&self, id: ~str) -> protocol;
    fn parse_state(&self, proto: protocol);
    fn parse_message(&self, state: state);
}

impl proto_parser for parser::Parser {
    fn parse_proto(&self, id: ~str) -> protocol {
        let proto = protocol(id, *self.span);

        self.parse_seq_to_before_end(
            &token::EOF,
            SeqSep {
                sep: None,
                trailing_sep_allowed: false,
            },
            |self| self.parse_state(proto)
        );

        return proto;
    }

    fn parse_state(&self, proto: protocol) {
        let id = self.parse_ident();
        let name = copy *self.interner.get(id);

        self.expect(&token::COLON);
        let dir = match copy *self.token {
            token::IDENT(n, _) => self.interner.get(n),
            _ => fail!()
        };
        self.bump();
        let dir = match dir {
          @~"send" => send,
          @~"recv" => recv,
          _ => fail!()
        };

        let generics = if *self.token == token::LT {
            self.parse_generics()
        } else {
            ast_util::empty_generics()
        };

        let state = proto.add_state_poly(name, id, dir, generics);

        // parse the messages
        self.parse_unspanned_seq(
            &token::LBRACE,
            &token::RBRACE,
            SeqSep {
                sep: Some(token::COMMA),
                trailing_sep_allowed: true,
            },
            |self| self.parse_message(state)
        );
    }

    fn parse_message(&self, state: state) {
        let mname = copy *self.interner.get(self.parse_ident());

        let args = if *self.token == token::LPAREN {
            self.parse_unspanned_seq(
                &token::LPAREN,
                &token::RPAREN,
                SeqSep {
                    sep: Some(token::COMMA),
                    trailing_sep_allowed: true,
                },
                |p| p.parse_ty(false)
            )
        }
        else { ~[] };

        self.expect(&token::RARROW);

        let next = match *self.token {
          token::IDENT(_, _) => {
            let name = copy *self.interner.get(self.parse_ident());
            let ntys = if *self.token == token::LT {
                self.parse_unspanned_seq(
                    &token::LT,
                    &token::GT,
                    SeqSep {
                        sep: Some(token::COMMA),
                        trailing_sep_allowed: true,
                    },
                    |p| p.parse_ty(false)
                )
            }
            else { ~[] };
            Some(next_state {state: name, tys: ntys})
          }
          token::NOT => {
            // -> !
            self.bump();
            None
          }
          _ => self.fatal(~"invalid next state")
        };

        state.add_message(mname, *self.span, args, next);

    }
}
