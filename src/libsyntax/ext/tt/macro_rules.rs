// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, TokenTree};
use codemap::{Span, DUMMY_SP};
use ext::base::{ExtCtxt, MacResult, SyntaxExtension};
use ext::base::{NormalTT, TTMacroExpander};
use ext::tt::macro_parser::{Success, Error, Failure};
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use ext::tt::macro_parser::parse;
use parse::lexer::new_tt_reader;
use parse::parser::Parser;
use parse::token::{self, special_idents, gensym_ident, NtTT, Token};
use parse::token::Token::*;
use print;
use ptr::P;

use util::small_vector::SmallVector;

use std::cell::RefCell;
use std::rc::Rc;
use std::iter::once;

struct ParserAnyMacro<'a> {
    parser: RefCell<Parser<'a>>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: ast::Ident
}

impl<'a> ParserAnyMacro<'a> {
    /// Make sure we don't have any tokens left to parse, so we don't
    /// silently drop anything. `allow_semi` is so that "optional"
    /// semicolons at the end of normal expressions aren't complained
    /// about e.g. the semicolon in `macro_rules! kapow { () => {
    /// panic!(); } }` doesn't get picked up by .parse_expr(), but it's
    /// allowed to be there.
    fn ensure_complete_parse(&self, allow_semi: bool) {
        let mut parser = self.parser.borrow_mut();
        if allow_semi && parser.token == token::Semi {
            panictry!(parser.bump())
        }
        if parser.token != token::Eof {
            let token_str = parser.this_token_to_string();
            let msg = format!("macro expansion ignores token `{}` and any \
                               following",
                              token_str);
            let span = parser.span;
            parser.span_err(span, &msg[..]);

            let msg = format!("caused by the macro expansion here; the usage \
                               of `{}` is likely invalid in this context",
                               self.macro_ident);
            parser.span_note(self.site_span, &msg[..]);
        }
    }
}

impl<'a> MacResult for ParserAnyMacro<'a> {
    fn make_expr(self: Box<ParserAnyMacro<'a>>) -> Option<P<ast::Expr>> {
        let ret = panictry!(self.parser.borrow_mut().parse_expr());
        self.ensure_complete_parse(true);
        Some(ret)
    }
    fn make_pat(self: Box<ParserAnyMacro<'a>>) -> Option<P<ast::Pat>> {
        let ret = panictry!(self.parser.borrow_mut().parse_pat());
        self.ensure_complete_parse(false);
        Some(ret)
    }
    fn make_items(self: Box<ParserAnyMacro<'a>>) -> Option<SmallVector<P<ast::Item>>> {
        let mut ret = SmallVector::zero();
        while let Some(item) = panictry!(self.parser.borrow_mut().parse_item()) {
            ret.push(item);
        }
        self.ensure_complete_parse(false);
        Some(ret)
    }

    fn make_impl_items(self: Box<ParserAnyMacro<'a>>)
                       -> Option<SmallVector<P<ast::ImplItem>>> {
        let mut ret = SmallVector::zero();
        loop {
            let mut parser = self.parser.borrow_mut();
            match parser.token {
                token::Eof => break,
                _ => ret.push(panictry!(parser.parse_impl_item()))
            }
        }
        self.ensure_complete_parse(false);
        Some(ret)
    }

    fn make_stmts(self: Box<ParserAnyMacro<'a>>)
                 -> Option<SmallVector<P<ast::Stmt>>> {
        let mut ret = SmallVector::zero();
        loop {
            let mut parser = self.parser.borrow_mut();
            match parser.token {
                token::Eof => break,
                _ => match parser.parse_stmt() {
                    Ok(maybe_stmt) => match maybe_stmt {
                        Some(stmt) => ret.push(stmt),
                        None => (),
                    },
                    Err(_) => break,
                }
            }
        }
        self.ensure_complete_parse(false);
        Some(ret)
    }

    fn make_ty(self: Box<ParserAnyMacro<'a>>) -> Option<P<ast::Ty>> {
        let ret = panictry!(self.parser.borrow_mut().parse_ty());
        self.ensure_complete_parse(true);
        Some(ret)
    }
}

struct MacroRulesMacroExpander {
    name: ast::Ident,
    imported_from: Option<ast::Ident>,
    lhses: Vec<Rc<NamedMatch>>,
    rhses: Vec<Rc<NamedMatch>>,
}

impl TTMacroExpander for MacroRulesMacroExpander {
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   arg: &[ast::TokenTree])
                   -> Box<MacResult+'cx> {
        generic_extension(cx,
                          sp,
                          self.name,
                          self.imported_from,
                          arg,
                          &self.lhses,
                          &self.rhses)
    }
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(cx: &'cx ExtCtxt,
                          sp: Span,
                          name: ast::Ident,
                          imported_from: Option<ast::Ident>,
                          arg: &[ast::TokenTree],
                          lhses: &[Rc<NamedMatch>],
                          rhses: &[Rc<NamedMatch>])
                          -> Box<MacResult+'cx> {
    if cx.trace_macros() {
        println!("{}! {{ {} }}",
                 name,
                 print::pprust::tts_to_string(arg));
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_fail_spot = DUMMY_SP;
    let mut best_fail_msg = "internal error: ran no matchers".to_string();

    for (i, lhs) in lhses.iter().enumerate() { // try each arm's matchers
        match **lhs {
          MatchedNonterminal(NtTT(ref lhs_tt)) => {
            let lhs_tt = match **lhs_tt {
                TokenTree::Delimited(_, ref delim) => &delim.tts[..],
                _ => panic!(cx.span_fatal(sp, "malformed macro lhs"))
            };

            match TokenTree::parse(cx, lhs_tt, arg) {
              Success(named_matches) => {
                let rhs = match *rhses[i] {
                    // okay, what's your transcriber?
                    MatchedNonterminal(NtTT(ref tt)) => {
                        match **tt {
                            // ignore delimiters
                            TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                            _ => panic!(cx.span_fatal(sp, "macro rhs must be delimited")),
                        }
                    },
                    _ => cx.span_bug(sp, "bad thing in rhs")
                };
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let trncbr = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                           Some(named_matches),
                                           imported_from,
                                           rhs);
                let mut p = Parser::new(cx.parse_sess(), cx.cfg(), Box::new(trncbr));
                panictry!(p.check_unknown_macro_variable());
                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return Box::new(ParserAnyMacro {
                    parser: RefCell::new(p),

                    // Pass along the original expansion site and the name of the macro
                    // so we can print a useful error message if the parse of the expanded
                    // macro leaves unparsed tokens.
                    site_span: sp,
                    macro_ident: name
                })
              }
              Failure(sp, ref msg) => if sp.lo >= best_fail_spot.lo {
                best_fail_spot = sp;
                best_fail_msg = (*msg).clone();
              },
              Error(err_sp, ref msg) => {
                panic!(cx.span_fatal(err_sp.substitute_dummy(sp), &msg[..]))
              }
            }
          }
          _ => cx.bug("non-matcher found in parsed lhses")
        }
    }

    panic!(cx.span_fatal(best_fail_spot.substitute_dummy(sp), &best_fail_msg[..]));
}

// Note that macro-by-example's input is also matched against a token tree:
//                   $( $lhs:tt => $rhs:tt );+
//
// Holy self-referential!

/// Converts a `macro_rules!` invocation into a syntax extension.
pub fn compile<'cx>(cx: &'cx mut ExtCtxt,
                    def: &ast::MacroDef) -> SyntaxExtension {

    let lhs_nm =  gensym_ident("lhs");
    let rhs_nm =  gensym_ident("rhs");

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let match_lhs_tok = MatchNt(lhs_nm, special_idents::tt, token::Plain, token::Plain);
    let match_rhs_tok = MatchNt(rhs_nm, special_idents::tt, token::Plain, token::Plain);
    let argument_gram = vec!(
        TokenTree::Sequence(DUMMY_SP,
                   Rc::new(ast::SequenceRepetition {
                       tts: vec![
                           TokenTree::Token(DUMMY_SP, match_lhs_tok),
                           TokenTree::Token(DUMMY_SP, token::FatArrow),
                           TokenTree::Token(DUMMY_SP, match_rhs_tok)],
                       separator: Some(token::Semi),
                       op: ast::OneOrMore,
                       num_captures: 2
                   })),
        //to phase into semicolon-termination instead of
        //semicolon-separation
        TokenTree::Sequence(DUMMY_SP,
                   Rc::new(ast::SequenceRepetition {
                       tts: vec![TokenTree::Token(DUMMY_SP, token::Semi)],
                       separator: None,
                       op: ast::ZeroOrMore,
                       num_captures: 0
                   })));


    // Parse the macro_rules! invocation (`none` is for no interpolations):
    let arg_reader = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                   None,
                                   None,
                                   def.body.clone());

    let argument_map = match parse(cx.parse_sess(),
                                   cx.cfg(),
                                   arg_reader,
                                   &argument_gram) {
        Success(m) => m,
        Failure(sp, str) | Error(sp, str) => {
            panic!(cx.parse_sess().span_diagnostic
                     .span_fatal(sp.substitute_dummy(def.span), &str[..]));
        }
    };

    // Extract the arguments:
    let lhses = match **argument_map.get(&lhs_nm.name).unwrap() {
        MatchedSeq(ref s, _) => /* FIXME (#2543) */ (*s).clone(),
        _ => cx.span_bug(def.span, "wrong-structured lhs")
    };

    for lhs in &lhses {
        check_lhs_nt_follows(cx, lhs, def.span);
    }

    let rhses = match **argument_map.get(&rhs_nm.name).unwrap() {
        MatchedSeq(ref s, _) => /* FIXME (#2543) */ (*s).clone(),
        _ => cx.span_bug(def.span, "wrong-structured rhs")
    };

    let exp: Box<_> = Box::new(MacroRulesMacroExpander {
        name: def.ident,
        imported_from: def.imported_from,
        lhses: lhses,
        rhses: rhses,
    });

    NormalTT(exp, Some(def.span), def.allow_internal_unstable)
}

fn check_lhs_nt_follows(cx: &mut ExtCtxt, lhs: &NamedMatch, sp: Span) {
    // lhs is going to be like MatchedNonterminal(NtTT(TokenTree::Delimited(...))), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    match lhs {
        &MatchedNonterminal(NtTT(ref inner)) => match &**inner {
            &TokenTree::Delimited(_, ref tts) => {
                check_matcher(cx, tts.tts.iter(), &Eof);
            },
            tt @ &TokenTree::Sequence(..) => {
                check_matcher(cx, Some(tt).into_iter(), &Eof);
            },
            _ => cx.span_err(sp, "Invalid macro matcher; matchers must be contained \
               in balanced delimiters or a repetition indicator")
        },
        _ => cx.span_bug(sp, "wrong-structured lhs for follow check (didn't find a \
           MatchedNonterminal)")
    };
    // we don't abort on errors on rejection, the driver will do that for us
    // after parsing/expansion. we can report every error in every macro this way.
}

// returns the last token that was checked, for TokenTree::Sequence. this gets used later on.
fn check_matcher<'a, I>(cx: &mut ExtCtxt, matcher: I, follow: &Token)
-> Option<(Span, Token)> where I: Iterator<Item=&'a TokenTree> {
    use print::pprust::token_to_string;

    let mut last = None;

    // 2. For each token T in M:
    let mut tokens = matcher.peekable();
    while let Some(token) = tokens.next() {
        last = match *token {
            TokenTree::Token(sp, MatchNt(ref name, ref frag_spec, _, _)) => {
                // ii. If T is a simple NT, look ahead to the next token T' in
                // M. If T' is in the set FOLLOW(NT), continue. Else; reject.
                if can_be_followed_by_any(&frag_spec.name.as_str()) {
                    continue
                } else {
                    let next_token = match tokens.peek() {
                        // If T' closes a complex NT, replace T' with F
                        Some(&&TokenTree::Token(_, CloseDelim(_))) => follow.clone(),
                        Some(&&TokenTree::Token(_, ref tok)) => tok.clone(),
                        Some(&&TokenTree::Sequence(sp, _)) => {
                            // Be conservative around sequences: to be
                            // more specific, we would need to
                            // consider FIRST sets, but also the
                            // possibility that the sequence occurred
                            // zero times (in which case we need to
                            // look at the token that follows the
                            // sequence, which may itself be a sequence,
                            // and so on).
                            cx.span_err(sp,
                                        &format!("`${0}:{1}` is followed by a \
                                                  sequence repetition, which is not \
                                                  allowed for `{1}` fragments",
                                                 name, frag_spec)
                                        );
                            Eof
                        },
                        // die next iteration
                        Some(&&TokenTree::Delimited(_, ref delim)) => delim.close_token(),
                        // else, we're at the end of the macro or sequence
                        None => follow.clone()
                    };

                    let tok = if let TokenTree::Token(_, ref tok) = *token {
                        tok
                    } else {
                        unreachable!()
                    };

                    // If T' is in the set FOLLOW(NT), continue. Else, reject.
                    match (&next_token, is_in_follow(cx, &next_token, &frag_spec.name.as_str())) {
                        (_, Err(msg)) => {
                            cx.span_err(sp, &msg);
                            continue
                        }
                        (&Eof, _) => return Some((sp, tok.clone())),
                        (_, Ok(true)) => continue,
                        (next, Ok(false)) => {
                            cx.span_err(sp, &format!("`${0}:{1}` is followed by `{2}`, which \
                                                      is not allowed for `{1}` fragments",
                                                     name, frag_spec,
                                                     token_to_string(next)));
                            continue
                        },
                    }
                }
            },
            TokenTree::Sequence(sp, ref seq) => {
                // iii. Else, T is a complex NT.
                match seq.separator {
                    // If T has the form $(...)U+ or $(...)U* for some token U,
                    // run the algorithm on the contents with F set to U. If it
                    // accepts, continue, else, reject.
                    Some(ref u) => {
                        let last = check_matcher(cx, seq.tts.iter(), u);
                        match last {
                            // Since the delimiter isn't required after the last
                            // repetition, make sure that the *next* token is
                            // sane. This doesn't actually compute the FIRST of
                            // the rest of the matcher yet, it only considers
                            // single tokens and simple NTs. This is imprecise,
                            // but conservatively correct.
                            Some((span, tok)) => {
                                let fol = match tokens.peek() {
                                    Some(&&TokenTree::Token(_, ref tok)) => tok.clone(),
                                    Some(&&TokenTree::Delimited(_, ref delim)) =>
                                        delim.close_token(),
                                    Some(_) => {
                                        cx.span_err(sp, "sequence repetition followed by \
                                                another sequence repetition, which is not allowed");
                                        Eof
                                    },
                                    None => Eof
                                };
                                check_matcher(cx, once(&TokenTree::Token(span, tok.clone())),
                                              &fol)
                            },
                            None => last,
                        }
                    },
                    // If T has the form $(...)+ or $(...)*, run the algorithm
                    // on the contents with F set to the token following the
                    // sequence. If it accepts, continue, else, reject.
                    None => {
                        let fol = match tokens.peek() {
                            Some(&&TokenTree::Token(_, ref tok)) => tok.clone(),
                            Some(&&TokenTree::Delimited(_, ref delim)) => delim.close_token(),
                            Some(_) => {
                                cx.span_err(sp, "sequence repetition followed by another \
                                             sequence repetition, which is not allowed");
                                Eof
                            },
                            None => Eof
                        };
                        check_matcher(cx, seq.tts.iter(), &fol)
                    }
                }
            },
            TokenTree::Token(..) => {
                // i. If T is not an NT, continue.
                continue
            },
            TokenTree::Delimited(_, ref tts) => {
                // if we don't pass in that close delimiter, we'll incorrectly consider the matcher
                // `{ $foo:ty }` as having a follow that isn't `RBrace`
                check_matcher(cx, tts.tts.iter(), &tts.close_token())
            }
        }
    }
    last
}

/// True if a fragment of type `frag` can be followed by any sort of
/// token.  We use this (among other things) as a useful approximation
/// for when `frag` can be followed by a repetition like `$(...)*` or
/// `$(...)+`. In general, these can be a bit tricky to reason about,
/// so we adopt a conservative position that says that any fragment
/// specifier which consumes at most one token tree can be followed by
/// a fragment specifier (indeed, these fragments can be followed by
/// ANYTHING without fear of future compatibility hazards).
fn can_be_followed_by_any(frag: &str) -> bool {
    match frag {
        "item" |  // always terminated by `}` or `;`
        "block" | // exactly one token tree
        "ident" | // exactly one token tree
        "meta" |  // exactly one token tree
        "tt" =>    // exactly one token tree
            true,

        _ =>
            false,
    }
}

/// True if `frag` can legally be followed by the token `tok`. For
/// fragments that can consume an unbounded numbe of tokens, `tok`
/// must be within a well-defined follow set. This is intended to
/// guarantee future compatibility: for example, without this rule, if
/// we expanded `expr` to include a new binary operator, we might
/// break macros that were relying on that binary operator as a
/// separator.
fn is_in_follow(_: &ExtCtxt, tok: &Token, frag: &str) -> Result<bool, String> {
    if let &CloseDelim(_) = tok {
        // closing a token tree can never be matched by any fragment;
        // iow, we always require that `(` and `)` match, etc.
        Ok(true)
    } else {
        match frag {
            "item" => {
                // since items *must* be followed by either a `;` or a `}`, we can
                // accept anything after them
                Ok(true)
            },
            "block" => {
                // anything can follow block, the braces provide an easy boundary to
                // maintain
                Ok(true)
            },
            "stmt" | "expr"  => {
                match *tok {
                    FatArrow | Comma | Semi => Ok(true),
                    _ => Ok(false)
                }
            },
            "pat" => {
                match *tok {
                    FatArrow | Comma | Eq => Ok(true),
                    Ident(i, _) if i.name.as_str() == "if" || i.name.as_str() == "in" => Ok(true),
                    _ => Ok(false)
                }
            },
            "path" | "ty" => {
                match *tok {
                    Comma | FatArrow | Colon | Eq | Gt | Semi => Ok(true),
                    Ident(i, _) if i.name.as_str() == "as" => Ok(true),
                    _ => Ok(false)
                }
            },
            "ident" => {
                // being a single token, idents are harmless
                Ok(true)
            },
            "meta" | "tt" => {
                // being either a single token or a delimited sequence, tt is
                // harmless
                Ok(true)
            },
            _ => Err(format!("invalid fragment specifier `{}`", frag))
        }
    }
}
