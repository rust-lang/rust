// Parsing pipes protocols from token trees.

import parse::parser;
import ast::ident;
import parse::token;

import pipec::*;

impl proto_parser for parser {
    fn parse_proto(id: ident) -> protocol {
        let proto = protocol(id);

        self.parse_seq_to_before_end(token::EOF,
                                     {sep: none, trailing_sep_allowed: false},
                                     |self| self.parse_state(proto));

        ret proto;
    }

    fn parse_state(proto: protocol) {
        let id = self.parse_ident();
        self.expect(token::COLON);
        let dir = alt copy self.token {
          token::IDENT(n, _) {
            self.get_str(n)
          }
          _ { fail }
        };
        self.bump();
        let dir = alt dir {
          @~"send" { send }
          @~"recv" { recv }
          _ { fail }
        };

        let typarms = if self.token == token::LT {
            self.parse_ty_params()
        }
        else { ~[] };

        let state = proto.add_state_poly(id, dir, typarms);

        // parse the messages
        self.parse_unspanned_seq(
            token::LBRACE, token::RBRACE,
            {sep: some(token::COMMA), trailing_sep_allowed: true},
            |self| {
                let mname = self.parse_ident();

                let args = if self.token == token::LPAREN {
                    self.parse_unspanned_seq(token::LPAREN,
                                             token::RPAREN,
                                             {sep: some(token::COMMA),
                                              trailing_sep_allowed: true},
                                             |p| p.parse_ty(false))
                }
                else { ~[] };

                self.expect(token::RARROW);

                let next = self.parse_ident();

                let ntys = if self.token == token::LT {
                    self.parse_unspanned_seq(token::LT,
                                             token::GT,
                                             {sep: some(token::COMMA),
                                              trailing_sep_allowed: true},
                                             |p| p.parse_ty(false))
                }
                else { ~[] };

                state.add_message(mname, args, next, ntys);

            });
    }
}
