// Parsing pipes protocols from token trees.

use parse::parser;
use parse::token;

use pipec::*;

trait proto_parser {
    fn parse_proto(id: ~str) -> protocol;
    fn parse_state(proto: protocol);
}

impl parser: proto_parser {
    fn parse_proto(id: ~str) -> protocol {
        let proto = protocol(id, self.span);

        self.parse_seq_to_before_end(token::EOF,
                                     {sep: None, trailing_sep_allowed: false},
                                     |self| self.parse_state(proto));

        return proto;
    }

    fn parse_state(proto: protocol) {
        let id = self.parse_ident();
        let name = *self.interner.get(id);

        self.expect(token::COLON);
        let dir = match copy self.token {
          token::IDENT(n, _) => self.interner.get(n),
          _ => fail
        };
        self.bump();
        let dir = match dir {
          @~"send" => send,
          @~"recv" => recv,
          _ => fail
        };

        let typarms = if self.token == token::LT {
            self.parse_ty_params()
        }
        else { ~[] };

        let state = proto.add_state_poly(name, id, dir, typarms);

        // parse the messages
        self.parse_unspanned_seq(
            token::LBRACE, token::RBRACE,
            {sep: Some(token::COMMA), trailing_sep_allowed: true},
            |self| self.parse_message(state));
    }

    fn parse_message(state: state) {
        let mname = *self.interner.get(self.parse_ident());

        let args = if self.token == token::LPAREN {
            self.parse_unspanned_seq(token::LPAREN,
                                     token::RPAREN,
                                     {sep: Some(token::COMMA),
                                      trailing_sep_allowed: true},
                                     |p| p.parse_ty(false))
        }
        else { ~[] };

        self.expect(token::RARROW);

        let next = match copy self.token {
          token::IDENT(_, _) => {
            let name = *self.interner.get(self.parse_ident());
            let ntys = if self.token == token::LT {
                self.parse_unspanned_seq(token::LT,
                                         token::GT,
                                         {sep: Some(token::COMMA),
                                          trailing_sep_allowed: true},
                                         |p| p.parse_ty(false))
            }
            else { ~[] };
            Some({state: name, tys: ntys})
          }
          token::NOT => {
            // -> !
            self.bump();
            None
          }
          _ => self.fatal(~"invalid next state")
        };

        state.add_message(mname, copy self.span, args, next);

    }
}
