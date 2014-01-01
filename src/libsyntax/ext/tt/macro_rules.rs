// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Ident, matcher_, matcher, match_tok, match_nonterminal, match_seq};
use ast::{tt_delim};
use ast;
use codemap::{Span, Spanned, DUMMY_SP};
use ext::base::{AnyMacro, ExtCtxt, MacResult, MRAny, MRDef, MacroDef};
use ext::base::{NormalTT, SyntaxExpanderTTTrait};
use ext::base;
use ext::tt::macro_parser::{error};
use ext::tt::macro_parser::{named_match, matched_seq, matched_nonterminal};
use ext::tt::macro_parser::{parse, parse_or_else, success, failure};
use parse::lexer::{new_tt_reader, reader};
use parse::parser::Parser;
use parse::attr::parser_attr;
use parse::token::{get_ident_interner, special_idents, gensym_ident, ident_to_str};
use parse::token::{FAT_ARROW, SEMI, nt_matchers, nt_tt, EOF};
use print;
use util::small_vector::SmallVector;

struct ParserAnyMacro {
    parser: @Parser,
}

impl ParserAnyMacro {
    /// Make sure we don't have any tokens left to parse, so we don't
    /// silently drop anything. `allow_semi` is so that "optional"
    /// semilons at the end of normal expressions aren't complained
    /// about e.g. the semicolon in `macro_rules! kapow( () => {
    /// fail!(); } )` doesn't get picked up by .parse_expr(), but it's
    /// allowed to be there.
    fn ensure_complete_parse(&self, allow_semi: bool) {
        if allow_semi && *self.parser.token == SEMI {
            self.parser.bump()
        }
        if *self.parser.token != EOF {
            let msg = format!("macro expansion ignores token `{}` and any following",
                              self.parser.this_token_to_str());
            self.parser.span_err(*self.parser.span, msg);
        }
    }
}

impl AnyMacro for ParserAnyMacro {
    fn make_expr(&self) -> @ast::Expr {
        let ret = self.parser.parse_expr();
        self.ensure_complete_parse(true);
        ret
    }
    fn make_items(&self) -> SmallVector<@ast::item> {
        let mut ret = SmallVector::zero();
        loop {
            let attrs = self.parser.parse_outer_attributes();
            match self.parser.parse_item(attrs) {
                Some(item) => ret.push(item),
                None => break
            }
        }
        self.ensure_complete_parse(false);
        ret
    }
    fn make_stmt(&self) -> @ast::Stmt {
        let attrs = self.parser.parse_outer_attributes();
        let ret = self.parser.parse_stmt(attrs);
        self.ensure_complete_parse(true);
        ret
    }
}

struct MacroRulesSyntaxExpanderTTFun {
    name: Ident,
    lhses: @~[@named_match],
    rhses: @~[@named_match],
}

impl SyntaxExpanderTTTrait for MacroRulesSyntaxExpanderTTFun {
    fn expand(&self,
              cx: &mut ExtCtxt,
              sp: Span,
              arg: &[ast::token_tree],
              _: ast::SyntaxContext)
              -> MacResult {
        generic_extension(cx, sp, self.name, arg, *self.lhses, *self.rhses)
    }
}

// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension(cx: &ExtCtxt,
                     sp: Span,
                     name: Ident,
                     arg: &[ast::token_tree],
                     lhses: &[@named_match],
                     rhses: &[@named_match])
                     -> MacResult {
    if cx.trace_macros() {
        println!("{}! \\{ {} \\}",
                  cx.str_of(name),
                  print::pprust::tt_to_str(
                      &ast::tt_delim(@arg.to_owned()),
                      get_ident_interner()));
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_fail_spot = DUMMY_SP;
    let mut best_fail_msg = ~"internal error: ran no matchers";

    let s_d = cx.parse_sess().span_diagnostic;

    for (i, lhs) in lhses.iter().enumerate() { // try each arm's matchers
        match *lhs {
          @matched_nonterminal(nt_matchers(ref mtcs)) => {
            // `none` is because we're not interpolating
            let arg_rdr = new_tt_reader(
                s_d,
                None,
                arg.to_owned()
            ) as @mut reader;
            match parse(cx.parse_sess(), cx.cfg(), arg_rdr, *mtcs) {
              success(named_matches) => {
                let rhs = match rhses[i] {
                    // okay, what's your transcriber?
                    @matched_nonterminal(nt_tt(@ref tt)) => {
                        match (*tt) {
                            // cut off delimiters; don't parse 'em
                            tt_delim(ref tts) => {
                                (*tts).slice(1u,(*tts).len()-1u).to_owned()
                            }
                            _ => cx.span_fatal(
                                sp, "macro rhs must be delimited")
                        }
                    },
                    _ => cx.span_bug(sp, "bad thing in rhs")
                };
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let trncbr = new_tt_reader(s_d, Some(named_matches),
                                           rhs);
                let p = @Parser(cx.parse_sess(),
                                cx.cfg(),
                                trncbr as @mut reader);

                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return MRAny(@ParserAnyMacro {
                    parser: p,
                } as @AnyMacro)
              }
              failure(sp, ref msg) => if sp.lo >= best_fail_spot.lo {
                best_fail_spot = sp;
                best_fail_msg = (*msg).clone();
              },
              error(sp, ref msg) => cx.span_fatal(sp, (*msg))
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
                         arg: ~[ast::token_tree],
                         _: ast::SyntaxContext)
                         -> base::MacResult {
    // these spans won't matter, anyways
    fn ms(m: matcher_) -> matcher {
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
    let argument_gram = ~[
        ms(match_seq(~[
            ms(match_nonterminal(lhs_nm, special_idents::matchers, 0u)),
            ms(match_tok(FAT_ARROW)),
            ms(match_nonterminal(rhs_nm, special_idents::tt, 1u)),
        ], Some(SEMI), false, 0u, 2u)),
        //to phase into semicolon-termination instead of
        //semicolon-separation
        ms(match_seq(~[ms(match_tok(SEMI))], None, true, 2u, 2u))];


    // Parse the macro_rules! invocation (`none` is for no interpolations):
    let arg_reader = new_tt_reader(cx.parse_sess().span_diagnostic,
                                   None,
                                   arg.clone());
    let argument_map = parse_or_else(cx.parse_sess(),
                                     cx.cfg(),
                                     arg_reader as @mut reader,
                                     argument_gram);

    // Extract the arguments:
    let lhses = match *argument_map.get(&lhs_nm) {
        @matched_seq(ref s, _) => /* FIXME (#2543) */ @(*s).clone(),
        _ => cx.span_bug(sp, "wrong-structured lhs")
    };

    let rhses = match *argument_map.get(&rhs_nm) {
      @matched_seq(ref s, _) => /* FIXME (#2543) */ @(*s).clone(),
      _ => cx.span_bug(sp, "wrong-structured rhs")
    };

    let exp = @MacroRulesSyntaxExpanderTTFun {
        name: name,
        lhses: lhses,
        rhses: rhses,
    } as @SyntaxExpanderTTTrait;

    return MRDef(MacroDef {
        name: ident_to_str(&name),
        ext: NormalTT(exp, Some(sp))
    });
}
