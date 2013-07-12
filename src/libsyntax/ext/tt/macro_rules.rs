// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{ident, matcher_, matcher, match_tok, match_nonterminal, match_seq};
use ast::{tt_delim};
use ast;
use codemap::{span, spanned, dummy_sp};
use ext::base::{ExtCtxt, MacResult, MRAny, MRDef, MacroDef, NormalTT};
use ext::base;
use ext::tt::macro_parser::{error};
use ext::tt::macro_parser::{named_match, matched_seq, matched_nonterminal};
use ext::tt::macro_parser::{parse, parse_or_else, success, failure};
use parse::lexer::{new_tt_reader, reader};
use parse::parser::Parser;
use parse::token::{get_ident_interner, special_idents, gensym_ident, ident_to_str};
use parse::token::{FAT_ARROW, SEMI, nt_matchers, nt_tt};
use print;

use std::io;

pub fn add_new_extension(cx: @ExtCtxt,
                         sp: span,
                         name: ident,
                         arg: ~[ast::token_tree])
                      -> base::MacResult {
    // these spans won't matter, anyways
    fn ms(m: matcher_) -> matcher {
        spanned { node: copy m, span: dummy_sp() }
    }

    let lhs_nm =  gensym_ident("lhs");
    let rhs_nm =  gensym_ident("rhs");

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
    let arg_reader = new_tt_reader(copy cx.parse_sess().span_diagnostic,
                                   None, copy arg);
    let argument_map = parse_or_else(cx.parse_sess(),
                                     cx.cfg(),
                                     arg_reader as @reader,
                                     argument_gram);

    // Extract the arguments:
    let lhses = match *argument_map.get(&lhs_nm) {
        @matched_seq(ref s, _) => /* FIXME (#2543) */ @copy *s,
        _ => cx.span_bug(sp, "wrong-structured lhs")
    };

    let rhses = match *argument_map.get(&rhs_nm) {
      @matched_seq(ref s, _) => /* FIXME (#2543) */ @copy *s,
      _ => cx.span_bug(sp, "wrong-structured rhs")
    };

    // Given `lhses` and `rhses`, this is the new macro we create
    fn generic_extension(cx: @ExtCtxt, sp: span, name: ident,
                         arg: &[ast::token_tree],
                         lhses: &[@named_match], rhses: &[@named_match])
    -> MacResult {

        if cx.trace_macros() {
            io::println(fmt!("%s! { %s }",
                             cx.str_of(name),
                             print::pprust::tt_to_str(
                                 &ast::tt_delim(arg.to_owned()),
                                 get_ident_interner())));
        }

        // Which arm's failure should we report? (the one furthest along)
        let mut best_fail_spot = dummy_sp();
        let mut best_fail_msg = ~"internal error: ran no matchers";

        let s_d = cx.parse_sess().span_diagnostic;

        for lhses.iter().enumerate().advance |(i, lhs)| { // try each arm's matchers
            match *lhs {
              @matched_nonterminal(nt_matchers(ref mtcs)) => {
                // `none` is because we're not interpolating
                let arg_rdr = new_tt_reader(
                    s_d,
                    None,
                    arg.to_owned()
                ) as @reader;
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
                                    trncbr as @reader);

                    // Let the context choose how to interpret the result.
                    // Weird, but useful for X-macros.
                    return MRAny(|| p.parse_expr(),
                                  || p.parse_item(~[/* no attrs*/]),
                                  || p.parse_stmt(~[/* no attrs*/]));
                  }
                  failure(sp, ref msg) => if sp.lo >= best_fail_spot.lo {
                    best_fail_spot = sp;
                    best_fail_msg = copy *msg;
                  },
                  error(sp, ref msg) => cx.span_fatal(sp, (*msg))
                }
              }
              _ => cx.bug("non-matcher found in parsed lhses")
            }
        }
        cx.span_fatal(best_fail_spot, best_fail_msg);
    }

    let exp: @fn(@ExtCtxt, span, &[ast::token_tree]) -> MacResult =
        |cx, sp, arg| generic_extension(cx, sp, name, arg, *lhses, *rhses);

    return MRDef(MacroDef{
        name: ident_to_str(&name),
        ext: NormalTT(base::SyntaxExpanderTT{expander: exp, span: Some(sp)})
    });
}
