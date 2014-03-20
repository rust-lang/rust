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

use std::vec::Vec;

enum State {
    Asm,
    Outputs,
    Inputs,
    Clobbers,
    Options,
    StateNone
}

impl State {
    fn next(&self) -> State {
        match *self {
            Asm       => Outputs,
            Outputs   => Inputs,
            Inputs    => Clobbers,
            Clobbers  => Options,
            Options   => StateNone,
            StateNone => StateNone
        }
    }
}

static OPTIONS: &'static [&'static str] = &["volatile", "alignstack", "intel"];

pub fn expand_asm(cx: &mut ExtCtxt, sp: Span, tts: &[ast::TokenTree])
               -> base::MacResult {
    let mut p = parse::new_parser_from_tts(cx.parse_sess(),
                                           cx.cfg(),
                                           tts.iter()
                                              .map(|x| (*x).clone())
                                              .collect());

    let mut asm = InternedString::new("");
    let mut asm_str_style = None;
    let mut outputs = Vec::new();
    let mut inputs = Vec::new();
    let mut cons = ~"";
    let mut volatile = false;
    let mut alignstack = false;
    let mut dialect = ast::AsmAtt;

    let mut state = Asm;

    let mut read_write_operands = Vec::new();

    'statement: loop {
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

                    let span = p.last_span;

                    p.expect(&token::LPAREN);
                    let out = p.parse_expr();
                    p.expect(&token::RPAREN);

                    // Expands a read+write operand into two operands.
                    //
                    // Use '+' modifier when you want the same expression
                    // to be both an input and an output at the same time.
                    // It's the opposite of '=&' which means that the memory
                    // cannot be shared with any other operand (usually when
                    // a register is clobbered early.)
                    let output = match constraint.get().slice_shift_char() {
                        (Some('='), _) => None,
                        (Some('+'), operand) => {
                            // Save a reference to the output
                            read_write_operands.push((outputs.len(), out));
                            Some(token::intern_and_get_ident("=" + operand))
                        }
                        _ => {
                            cx.span_err(span, "output operand constraint lacks '=' or '+'");
                            None
                        }
                    };

                    outputs.push((output.unwrap_or(constraint), out));
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
                let mut clobs = Vec::new();
                while p.token != token::EOF &&
                      p.token != token::COLON &&
                      p.token != token::MOD_SEP {

                    if clobs.len() != 0 {
                        p.eat(&token::COMMA);
                    }

                    let (s, _str_style) = p.parse_str();
                    let clob = format!("~\\{{}\\}", s);
                    clobs.push(clob);

                    if OPTIONS.iter().any(|opt| s.equiv(opt)) {
                        cx.span_warn(p.last_span, "expected a clobber, but found an option");
                    }
                }

                cons = clobs.connect(",");
            }
            Options => {
                let (option, _str_style) = p.parse_str();

                if option.equiv(&("volatile")) {
                    // Indicates that the inline assembly has side effects
                    // and must not be optimized out along with its outputs.
                    volatile = true;
                } else if option.equiv(&("alignstack")) {
                    alignstack = true;
                } else if option.equiv(&("intel")) {
                    dialect = ast::AsmIntel;
                } else {
                    cx.span_warn(p.last_span, "unrecognized option");
                }

                if p.token == token::COMMA {
                    p.eat(&token::COMMA);
                }
            }
            StateNone => ()
        }

        loop {
            // MOD_SEP is a double colon '::' without space in between.
            // When encountered, the state must be advanced twice.
            match (&p.token, state.next(), state.next().next()) {
                (&token::COLON, StateNone, _)   |
                (&token::MOD_SEP, _, StateNone) => {
                    p.bump();
                    break 'statement;
                }
                (&token::COLON, st, _)   |
                (&token::MOD_SEP, _, st) => {
                    p.bump();
                    state = st;
                }
                (&token::EOF, _, _) => break 'statement,
                _ => break
            }
        }
    }

    // Append an input operand, with the form of ("0", expr)
    // that links to an output operand.
    for &(i, out) in read_write_operands.iter() {
        inputs.push((token::intern_and_get_ident(i.to_str()), out));
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
