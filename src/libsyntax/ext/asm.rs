// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * Inline assembly support.
 */

use ast;
use codemap::span;
use ext::base;
use ext::base::*;
use parse;
use parse::token;

use std::vec;

enum State {
    Asm,
    Outputs,
    Inputs,
    Clobbers,
    Options
}

fn next_state(s: State) -> Option<State> {
    match s {
        Asm      => Some(Outputs),
        Outputs  => Some(Inputs),
        Inputs   => Some(Clobbers),
        Clobbers => Some(Options),
        Options  => None
    }
}

pub fn expand_asm(cx: @ExtCtxt, sp: span, tts: &[ast::token_tree])
               -> base::MacResult {
    let p = parse::new_parser_from_tts(cx.parse_sess(),
                                       cx.cfg(),
                                       vec::to_owned(tts));

    let mut asm = @"";
    let mut outputs = ~[];
    let mut inputs = ~[];
    let mut cons = ~"";
    let mut volatile = false;
    let mut alignstack = false;
    let mut dialect = ast::asm_att;

    let mut state = Asm;

    // Not using labeled break to get us through one round of bootstrapping.
    let mut continue = true;
    while continue {
        match state {
            Asm => {
                asm = expr_to_str(cx, p.parse_expr(),
                                  ~"inline assembly must be a string literal.");
            }
            Outputs => {
                while *p.token != token::EOF &&
                      *p.token != token::COLON &&
                      *p.token != token::MOD_SEP {

                    if outputs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let constraint = p.parse_str();
                    p.expect(&token::LPAREN);
                    let out = p.parse_expr();
                    p.expect(&token::RPAREN);

                    let out = @ast::expr {
                        id: cx.next_id(),
                        span: out.span,
                        node: ast::expr_addr_of(ast::m_mutbl, out)
                    };

                    outputs.push((constraint, out));
                }
            }
            Inputs => {
                while *p.token != token::EOF &&
                      *p.token != token::COLON &&
                      *p.token != token::MOD_SEP {

                    if inputs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let constraint = p.parse_str();
                    p.expect(&token::LPAREN);
                    let in = p.parse_expr();
                    p.expect(&token::RPAREN);

                    inputs.push((constraint, in));
                }
            }
            Clobbers => {
                let mut clobs = ~[];
                while *p.token != token::EOF &&
                      *p.token != token::COLON &&
                      *p.token != token::MOD_SEP {

                    if clobs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let clob = fmt!("~{%s}", p.parse_str());
                    clobs.push(clob);
                }

                cons = clobs.connect(",");
            }
            Options => {
                let option = p.parse_str();

                if "volatile" == option {
                    volatile = true;
                } else if "alignstack" == option {
                    alignstack = true;
                } else if "intel" == option {
                    dialect = ast::asm_intel;
                }

                if *p.token == token::COMMA {
                    p.eat(&token::COMMA);
                }
            }
        }

        while *p.token == token::COLON   ||
              *p.token == token::MOD_SEP ||
              *p.token == token::EOF {
            state = if *p.token == token::COLON {
                p.bump();
                match next_state(state) {
                    Some(x) => x,
                    None    => {
                        continue = false;
                        break
                    }
                }
            } else if *p.token == token::MOD_SEP {
                p.bump();
                let s = match next_state(state) {
                    Some(x) => x,
                    None    => {
                        continue = false;
                        break
                    }
                };
                match next_state(s) {
                    Some(x) => x,
                    None    => {
                        continue = false;
                        break
                    }
                }
            } else if *p.token == token::EOF {
                continue = false;
                break;
            } else {
               state
            };
        }
    }

    MRExpr(@ast::expr {
        id: cx.next_id(),
        node: ast::expr_inline_asm(ast::inline_asm {
            asm: asm,
            clobbers: cons.to_managed(),
            inputs: inputs,
            outputs: outputs,
            volatile: volatile,
            alignstack: alignstack,
            dialect: dialect
        }),
        span: sp
    })
}
