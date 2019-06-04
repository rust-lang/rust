// Inline assembly support.
//
use State::*;

use rustc_data_structures::thin_vec::ThinVec;

use errors::DiagnosticBuilder;

use syntax::ast;
use syntax::ext::base::{self, *};
use syntax::feature_gate;
use syntax::parse;
use syntax::parse::token::{self, Token};
use syntax::ptr::P;
use syntax::symbol::{kw, sym, Symbol};
use syntax::ast::AsmDialect;
use syntax_pos::Span;
use syntax::tokenstream;
use syntax::{span_err, struct_span_err};

enum State {
    Asm,
    Outputs,
    Inputs,
    Clobbers,
    Options,
    StateNone,
}

impl State {
    fn next(&self) -> State {
        match *self {
            Asm => Outputs,
            Outputs => Inputs,
            Inputs => Clobbers,
            Clobbers => Options,
            Options => StateNone,
            StateNone => StateNone,
        }
    }
}

const OPTIONS: &[Symbol] = &[sym::volatile, sym::alignstack, sym::intel];

pub fn expand_asm<'cx>(cx: &'cx mut ExtCtxt<'_>,
                       sp: Span,
                       tts: &[tokenstream::TokenTree])
                       -> Box<dyn base::MacResult + 'cx> {
    if !cx.ecfg.enable_asm() {
        feature_gate::emit_feature_err(&cx.parse_sess,
                                       sym::asm,
                                       sp,
                                       feature_gate::GateIssue::Language,
                                       feature_gate::EXPLAIN_ASM);
    }

    let mut inline_asm = match parse_inline_asm(cx, sp, tts) {
        Ok(Some(inline_asm)) => inline_asm,
        Ok(None) => return DummyResult::expr(sp),
        Err(mut err) => {
            err.emit();
            return DummyResult::expr(sp);
        }
    };

    // If there are no outputs, the inline assembly is executed just for its side effects,
    // so ensure that it is volatile
    if inline_asm.outputs.is_empty() {
        inline_asm.volatile = true;
    }

    MacEager::expr(P(ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprKind::InlineAsm(P(inline_asm)),
        span: sp,
        attrs: ThinVec::new(),
    }))
}

fn parse_inline_asm<'a>(
    cx: &mut ExtCtxt<'a>,
    sp: Span,
    tts: &[tokenstream::TokenTree],
) -> Result<Option<ast::InlineAsm>, DiagnosticBuilder<'a>> {
    // Split the tts before the first colon, to avoid `asm!("x": y)`  being
    // parsed as `asm!(z)` with `z = "x": y` which is type ascription.
    let first_colon = tts.iter()
        .position(|tt| {
            match *tt {
                tokenstream::TokenTree::Token(Token { kind: token::Colon, .. }) |
                tokenstream::TokenTree::Token(Token { kind: token::ModSep, .. }) => true,
                _ => false,
            }
        })
        .unwrap_or(tts.len());
    let mut p = cx.new_parser_from_tts(&tts[first_colon..]);
    let mut asm = kw::Invalid;
    let mut asm_str_style = None;
    let mut outputs = Vec::new();
    let mut inputs = Vec::new();
    let mut clobs = Vec::new();
    let mut volatile = false;
    let mut alignstack = false;
    let mut dialect = AsmDialect::Att;

    let mut state = Asm;

    'statement: loop {
        match state {
            Asm => {
                if asm_str_style.is_some() {
                    // If we already have a string with instructions,
                    // ending up in Asm state again is an error.
                    return Err(struct_span_err!(
                        cx.parse_sess.span_diagnostic,
                        sp,
                        E0660,
                        "malformed inline assembly"
                    ));
                }
                // Nested parser, stop before the first colon (see above).
                let mut p2 = cx.new_parser_from_tts(&tts[..first_colon]);

                if p2.token == token::Eof {
                    let mut err =
                        cx.struct_span_err(sp, "macro requires a string literal as an argument");
                    err.span_label(sp, "string literal required");
                    return Err(err);
                }

                let expr = p2.parse_expr()?;
                let (s, style) =
                    match expr_to_string(cx, expr, "inline assembly must be a string literal") {
                        Some((s, st)) => (s, st),
                        None => return Ok(None),
                    };

                // This is most likely malformed.
                if p2.token != token::Eof {
                    let mut extra_tts = p2.parse_all_token_trees()?;
                    extra_tts.extend(tts[first_colon..].iter().cloned());
                    p = parse::stream_to_parser(
                        cx.parse_sess,
                        extra_tts.into_iter().collect(),
                        Some("inline assembly"),
                    );
                }

                asm = s;
                asm_str_style = Some(style);
            }
            Outputs => {
                while p.token != token::Eof && p.token != token::Colon && p.token != token::ModSep {
                    if !outputs.is_empty() {
                        p.eat(&token::Comma);
                    }

                    let (constraint, _) = p.parse_str()?;

                    let span = p.prev_span;

                    p.expect(&token::OpenDelim(token::Paren))?;
                    let expr = p.parse_expr()?;
                    p.expect(&token::CloseDelim(token::Paren))?;

                    // Expands a read+write operand into two operands.
                    //
                    // Use '+' modifier when you want the same expression
                    // to be both an input and an output at the same time.
                    // It's the opposite of '=&' which means that the memory
                    // cannot be shared with any other operand (usually when
                    // a register is clobbered early.)
                    let constraint_str = constraint.as_str();
                    let mut ch = constraint_str.chars();
                    let output = match ch.next() {
                        Some('=') => None,
                        Some('+') => {
                            Some(Symbol::intern(&format!("={}", ch.as_str())))
                        }
                        _ => {
                            span_err!(cx, span, E0661,
                                                    "output operand constraint lacks '=' or '+'");
                            None
                        }
                    };

                    let is_rw = output.is_some();
                    let is_indirect = constraint_str.contains("*");
                    outputs.push(ast::InlineAsmOutput {
                        constraint: output.unwrap_or(constraint),
                        expr,
                        is_rw,
                        is_indirect,
                    });
                }
            }
            Inputs => {
                while p.token != token::Eof && p.token != token::Colon && p.token != token::ModSep {
                    if !inputs.is_empty() {
                        p.eat(&token::Comma);
                    }

                    let (constraint, _) = p.parse_str()?;

                    if constraint.as_str().starts_with("=") {
                        span_err!(cx, p.prev_span, E0662,
                                                "input operand constraint contains '='");
                    } else if constraint.as_str().starts_with("+") {
                        span_err!(cx, p.prev_span, E0663,
                                                "input operand constraint contains '+'");
                    }

                    p.expect(&token::OpenDelim(token::Paren))?;
                    let input = p.parse_expr()?;
                    p.expect(&token::CloseDelim(token::Paren))?;

                    inputs.push((constraint, input));
                }
            }
            Clobbers => {
                while p.token != token::Eof && p.token != token::Colon && p.token != token::ModSep {
                    if !clobs.is_empty() {
                        p.eat(&token::Comma);
                    }

                    let (s, _) = p.parse_str()?;

                    if OPTIONS.iter().any(|&opt| s == opt) {
                        cx.span_warn(p.prev_span, "expected a clobber, found an option");
                    } else if s.as_str().starts_with("{") || s.as_str().ends_with("}") {
                        span_err!(cx, p.prev_span, E0664,
                                                "clobber should not be surrounded by braces");
                    }

                    clobs.push(s);
                }
            }
            Options => {
                let (option, _) = p.parse_str()?;

                if option == sym::volatile {
                    // Indicates that the inline assembly has side effects
                    // and must not be optimized out along with its outputs.
                    volatile = true;
                } else if option == sym::alignstack {
                    alignstack = true;
                } else if option == sym::intel {
                    dialect = AsmDialect::Intel;
                } else {
                    cx.span_warn(p.prev_span, "unrecognized option");
                }

                if p.token == token::Comma {
                    p.eat(&token::Comma);
                }
            }
            StateNone => (),
        }

        loop {
            // MOD_SEP is a double colon '::' without space in between.
            // When encountered, the state must be advanced twice.
            match (&p.token.kind, state.next(), state.next().next()) {
                (&token::Colon, StateNone, _) |
                (&token::ModSep, _, StateNone) => {
                    p.bump();
                    break 'statement;
                }
                (&token::Colon, st, _) |
                (&token::ModSep, _, st) => {
                    p.bump();
                    state = st;
                }
                (&token::Eof, ..) => break 'statement,
                _ => break,
            }
        }
    }

    Ok(Some(ast::InlineAsm {
        asm,
        asm_str_style: asm_str_style.unwrap(),
        outputs,
        inputs,
        clobbers: clobs,
        volatile,
        alignstack,
        dialect,
        ctxt: cx.backtrace(),
    }))
}
