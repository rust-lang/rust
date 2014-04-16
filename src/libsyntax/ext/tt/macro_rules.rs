// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Ident, Matcher_, Matcher, MatchTok, MatchNonterminal, MatchSeq};
use ast::{TTDelim};
use ast;
use codemap::{Span, Spanned, DUMMY_SP};
use ext::base::{ExtCtxt, MacResult, MacroDef};
use ext::base::{NormalTT, MacroExpander};
use ext::base;
use ext::tt::macro_parser::{Success, Error, Failure};
use ext::tt::macro_parser::{NamedMatch, MatchedSeq, MatchedNonterminal};
use ext::tt::macro_parser::{parse, parse_or_else};
use parse::lexer::new_tt_reader;
use parse::parser::Parser;
use parse::attr::ParserAttr;
use parse::token::{special_idents, gensym_ident};
use parse::token::{FAT_ARROW, SEMI, NtMatchers, NtTT, EOF};
use parse::token;
use print;
use util::small_vector::SmallVector;

use std::cell::RefCell;
use std::rc::Rc;

struct ParserAnyMacro<'a> {
    parser: RefCell<Parser<'a>>,
}

impl<'a> ParserAnyMacro<'a> {
    /// Make sure we don't have any tokens left to parse, so we don't
    /// silently drop anything. `allow_semi` is so that "optional"
    /// semilons at the end of normal expressions aren't complained
    /// about e.g. the semicolon in `macro_rules! kapow( () => {
    /// fail!(); } )` doesn't get picked up by .parse_expr(), but it's
    /// allowed to be there.
    fn ensure_complete_parse(&self, allow_semi: bool) {
        let mut parser = self.parser.borrow_mut();
        if allow_semi && parser.token == SEMI {
            parser.bump()
        }
        if parser.token != EOF {
            let token_str = parser.this_token_to_str();
            let msg = format!("macro expansion ignores token `{}` and any \
                               following",
                              token_str);
            let span = parser.span;
            parser.span_err(span, msg);
        }
    }
}

impl<'a> MacResult for ParserAnyMacro<'a> {
    fn make_expr(&self) -> Option<@ast::Expr> {
        let ret = self.parser.borrow_mut().parse_expr();
        self.ensure_complete_parse(true);
        Some(ret)
    }
    fn make_items(&self) -> Option<SmallVector<@ast::Item>> {
        let mut ret = SmallVector::zero();
        loop {
            let mut parser = self.parser.borrow_mut();
            let attrs = parser.parse_outer_attributes();
            match parser.parse_item(attrs) {
                Some(item) => ret.push(item),
                None => break
            }
        }
        self.ensure_complete_parse(false);
        Some(ret)
    }
    fn make_stmt(&self) -> Option<@ast::Stmt> {
        let attrs = self.parser.borrow_mut().parse_outer_attributes();
        let ret = self.parser.borrow_mut().parse_stmt(attrs);
        self.ensure_complete_parse(true);
        Some(ret)
    }
}

struct MacroRulesMacroExpander {
    name: Ident,
    lhses: Vec<Rc<NamedMatch>>,
    rhses: Vec<Rc<NamedMatch>>,
}

impl MacroExpander for MacroRulesMacroExpander {
    fn expand(&self,
              cx: &mut ExtCtxt,
              sp: Span,
              arg: &[ast::TokenTree])
              -> ~MacResult {
        generic_extension(cx,
                          sp,
                          self.name,
                          arg,
                          self.lhses.as_slice(),
                          self.rhses.as_slice())
    }
}

struct MacroRulesDefiner {
    def: RefCell<Option<MacroDef>>
}
impl MacResult for MacroRulesDefiner {
    fn make_def(&self) -> Option<MacroDef> {
        Some(self.def.borrow_mut().take().expect("MacroRulesDefiner expanded twice"))
    }
}

// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension(cx: &ExtCtxt,
                     sp: Span,
                     name: Ident,
                     arg: &[ast::TokenTree],
                     lhses: &[Rc<NamedMatch>],
                     rhses: &[Rc<NamedMatch>])
                     -> ~MacResult {
    if cx.trace_macros() {
        println!("{}! \\{ {} \\}",
                 token::get_ident(name),
                 print::pprust::tt_to_str(&TTDelim(Rc::new(arg.iter()
                                                              .map(|x| (*x).clone())
                                                              .collect()))));
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_fail_spot = DUMMY_SP;
    let mut best_fail_msg = "internal error: ran no matchers".to_owned();

    for (i, lhs) in lhses.iter().enumerate() { // try each arm's matchers
        match **lhs {
          MatchedNonterminal(NtMatchers(ref mtcs)) => {
            // `None` is because we're not interpolating
            let arg_rdr = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                        None,
                                        arg.iter()
                                           .map(|x| (*x).clone())
                                           .collect());
            match parse(cx.parse_sess(), cx.cfg(), arg_rdr, mtcs.as_slice()) {
              Success(named_matches) => {
                let rhs = match *rhses[i] {
                    // okay, what's your transcriber?
                    MatchedNonterminal(NtTT(tt)) => {
                        match *tt {
                            // cut off delimiters; don't parse 'em
                            TTDelim(ref tts) => {
                                (*tts).slice(1u,(*tts).len()-1u)
                                      .iter()
                                      .map(|x| (*x).clone())
                                      .collect()
                            }
                            _ => cx.span_fatal(
                                sp, "macro rhs must be delimited")
                        }
                    },
                    _ => cx.span_bug(sp, "bad thing in rhs")
                };
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let trncbr = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                           Some(named_matches),
                                           rhs);
                let p = Parser(cx.parse_sess(), cx.cfg(), ~trncbr);
                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return ~ParserAnyMacro {
                    parser: RefCell::new(p),
                } as ~MacResult
              }
              Failure(sp, ref msg) => if sp.lo >= best_fail_spot.lo {
                best_fail_spot = sp;
                best_fail_msg = (*msg).clone();
              },
              Error(sp, ref msg) => cx.span_fatal(sp, (*msg))
            }
          }
          _ => cx.bug("non-matcher found in parsed lhses")
        }
    }
    cx.span_fatal(best_fail_spot, best_fail_msg);
}

// this procedure performs the expansion of the
// macro_rules! macro. It parses the RHS and adds
// an extension to the current context.
pub fn add_new_extension(cx: &mut ExtCtxt,
                         sp: Span,
                         name: Ident,
                         arg: Vec<ast::TokenTree> )
                         -> ~base::MacResult {
    // these spans won't matter, anyways
    fn ms(m: Matcher_) -> Matcher {
        Spanned {
            node: m.clone(),
            span: DUMMY_SP
        }
    }

    let lhs_nm =  gensym_ident("lhs");
    let rhs_nm =  gensym_ident("rhs");

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:mtcs => $rhs:tt );+
    // ...quasiquoting this would be nice.
    let argument_gram = vec!(
        ms(MatchSeq(vec!(
            ms(MatchNonterminal(lhs_nm, special_idents::matchers, 0u)),
            ms(MatchTok(FAT_ARROW)),
            ms(MatchNonterminal(rhs_nm, special_idents::tt, 1u))), Some(SEMI), false, 0u, 2u)),
        //to phase into semicolon-termination instead of
        //semicolon-separation
        ms(MatchSeq(vec!(ms(MatchTok(SEMI))), None, true, 2u, 2u)));


    // Parse the macro_rules! invocation (`none` is for no interpolations):
    let arg_reader = new_tt_reader(&cx.parse_sess().span_diagnostic,
                                   None,
                                   arg.clone());
    let argument_map = parse_or_else(cx.parse_sess(),
                                     cx.cfg(),
                                     arg_reader,
                                     argument_gram);

    // Extract the arguments:
    let lhses = match **argument_map.get(&lhs_nm) {
        MatchedSeq(ref s, _) => /* FIXME (#2543) */ (*s).clone(),
        _ => cx.span_bug(sp, "wrong-structured lhs")
    };

    let rhses = match **argument_map.get(&rhs_nm) {
        MatchedSeq(ref s, _) => /* FIXME (#2543) */ (*s).clone(),
        _ => cx.span_bug(sp, "wrong-structured rhs")
    };

    let exp = ~MacroRulesMacroExpander {
        name: name,
        lhses: lhses,
        rhses: rhses,
    };

    ~MacroRulesDefiner {
        def: RefCell::new(Some(MacroDef {
            name: token::get_ident(name).to_str(),
            ext: NormalTT(exp, Some(sp))
        }))
    } as ~MacResult
}
