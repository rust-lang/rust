// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
use codemap::Span;
use ext::base;
use ext::base::*;
use parse;
use parse::token::InternedString;
use parse::token;

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

pub fn expand_asm(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
               -> base::MacResult {
    let mut p = parse::new_parser_from_tts(cx.parse_sess(),
                                           cx.cfg(),
                                           tts.to_owned());

    let mut asm = InternedString::new("");
    let mut asm_str_style = None;
    let mut outputs = ~[];
    let mut inputs = ~[];
    let mut cons = ~"";
    let mut volatile = false;
    let mut alignstack = false;
    let mut dialect = ast::AsmAtt;

    let mut state = Asm;

    // Not using labeled break to get us through one round of bootstrapping.
    let mut continue_ = true;
    while continue_ {
        match state {
            Asm => {
                let (s, style) = match expr_to_str(cx, p.parse_expr(),
                                                   "inline assembly must be a string literal.") {
                    Some((s, st)) => (s, st),
                    // let compilation continue
                    None => return MacResult::dummy_expr(sp),
                };
                asm = s;
                asm_str_style = Some(style);
            }
            Outputs => {
                while p.token != token::EOF &&
                      p.token != token::COLON &&
                      p.token != token::MOD_SEP {

                    if outputs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let (constraint, _str_style) = p.parse_str();

                    if constraint.get().starts_with("+") {
                        cx.span_unimpl(p.last_span,
                                       "'+' (read+write) output operand constraint modifier");
                    } else if !constraint.get().starts_with("=") {
                        cx.span_err(p.last_span, "output operand constraint lacks '='");
                    }

                    p.expect(&token::LPAREN);
                    let out = p.parse_expr();
                    p.expect(&token::RPAREN);

                    outputs.push((constraint, out));
                }
            }
            Inputs => {
                while p.token != token::EOF &&
                      p.token != token::COLON &&
                      p.token != token::MOD_SEP {

                    if inputs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let (constraint, _str_style) = p.parse_str();

                    if constraint.get().starts_with("=") {
                        cx.span_err(p.last_span, "input operand constraint contains '='");
                    } else if constraint.get().starts_with("+") {
                        cx.span_err(p.last_span, "input operand constraint contains '+'");
                    }

                    p.expect(&token::LPAREN);
                    let input = p.parse_expr();
                    p.expect(&token::RPAREN);

                    inputs.push((constraint, input));
                }
            }
            Clobbers => {
                let mut clobs = ~[];
                while p.token != token::EOF &&
                      p.token != token::COLON &&
                      p.token != token::MOD_SEP {

                    if clobs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let (s, _str_style) = p.parse_str();
                    let clob = format!("~\\{{}\\}", s);
                    clobs.push(clob);
                }

                cons = clobs.connect(",");
            }
            Options => {
                let (option, _str_style) = p.parse_str();

                if option.equiv(&("volatile")) {
                    volatile = true;
                } else if option.equiv(&("alignstack")) {
                    alignstack = true;
                } else if option.equiv(&("intel")) {
                    dialect = ast::AsmIntel;
                }

                if p.token == token::COMMA {
                    p.eat(&token::COMMA);
                }
            }
        }

        while p.token == token::COLON   ||
              p.token == token::MOD_SEP ||
              p.token == token::EOF {
            state = if p.token == token::COLON {
                p.bump();
                match next_state(state) {
                    Some(x) => x,
                    None    => {
                        continue_ = false;
                        break
                    }
                }
            } else if p.token == token::MOD_SEP {
                p.bump();
                let s = match next_state(state) {
                    Some(x) => x,
                    None    => {
                        continue_ = false;
                        break
                    }
                };
                match next_state(s) {
                    Some(x) => x,
                    None    => {
                        continue_ = false;
                        break
                    }
                }
            } else if p.token == token::EOF {
                continue_ = false;
                break;
            } else {
               state
            };
        }
    }

    MRExpr(@ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprInlineAsm(ast::InlineAsm {
            asm: token::intern_and_get_ident(asm.get()),
            asm_str_style: asm_str_style.unwrap(),
            clobbers: token::intern_and_get_ident(cons),
            inputs: inputs,
            outputs: outputs,
            volatile: volatile,
            alignstack: alignstack,
            dialect: dialect
        }),
        span: sp
    })
}
