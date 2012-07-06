// Parsing pipes protocols from token trees.

import parse::parser;
import ast::ident;
import parse::token;

import pipec::*;

impl proto_parser for parser {
    fn parse_proto(id: ident) -> protocol {
        let proto = protocol(id);

        self.expect(token::LBRACE);

        while self.token != token::RBRACE {
            self.parse_state(proto);
        }

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
          @"send" { send }
          @"recv" { recv }
          _ { fail }
        };

        let state = proto.add_state(id, dir);
        // TODO: add typarams too.

        self.expect(token::LBRACE);

        while self.token != token::RBRACE {
            let mname = self.parse_ident();

            // TODO: parse data

            self.expect(token::RARROW);

            let next = self.parse_ident();
            // TODO: parse next types

            state.add_message(mname, ~[], next, ~[]);

            alt copy self.token {
              token::COMMA { self.bump() }
              token::RBRACE { }
              _ { fail }
            }
        }
        self.bump();
    }
}
