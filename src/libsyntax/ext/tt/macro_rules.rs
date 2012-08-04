import base::{ext_ctxt, mac_result, mr_expr, mr_def, expr_tt};
import codemap::span;
import ast::{ident, matcher_, matcher, match_tok,
             match_nonterminal, match_seq, tt_delim};
import parse::lexer::{new_tt_reader, tt_reader_as_reader, reader};
import parse::token::{FAT_ARROW, SEMI, LBRACE, RBRACE, nt_matchers, nt_tt};
import parse::parser::{parser, SOURCE_FILE};
import earley_parser::{parse, parse_or_else, success, failure, named_match,
                       matched_seq, matched_nonterminal};
import std::map::hashmap;

fn add_new_extension(cx: ext_ctxt, sp: span, name: ident,
                     arg: ~[ast::token_tree]) -> base::mac_result {
    // these spans won't matter, anyways
    fn ms(m: matcher_) -> matcher {
        {node: m, span: {lo: 0u, hi: 0u, expn_info: none}}
    }

    // The grammar for macro_rules! is:
    // $( $lhs:mtcs => $rhs:tt );+
    // ...quasiquoting this would be nice.
    let argument_gram = ~[
        ms(match_seq(~[
            ms(match_nonterminal(@~"lhs",@~"matchers", 0u)),
            ms(match_tok(FAT_ARROW)),
            ms(match_nonterminal(@~"rhs",@~"tt", 1u)),
        ], some(SEMI), false, 0u, 2u)),
        //to phase into semicolon-termination instead of
        //semicolon-separation
        ms(match_seq(~[ms(match_tok(SEMI))], none, true, 2u, 2u))];


    // Parse the macro_rules! invocation (`none` is for no interpolations):
    let arg_reader = new_tt_reader(cx.parse_sess().span_diagnostic,
                                   cx.parse_sess().interner, none, arg);
    let argument_map = parse_or_else(cx.parse_sess(), cx.cfg(),
                                     arg_reader as reader, argument_gram);

    // Extract the arguments:
    let lhses:~[@named_match] = alt argument_map.get(@~"lhs") {
      @matched_seq(s, sp) => s,
      _ => cx.span_bug(sp, ~"wrong-structured lhs")
    };
    let rhses:~[@named_match] = alt argument_map.get(@~"rhs") {
      @matched_seq(s, sp) => s,
      _ => cx.span_bug(sp, ~"wrong-structured rhs")
    };

    // Given `lhses` and `rhses`, this is the new macro we create
    fn generic_extension(cx: ext_ctxt, sp: span, arg: ~[ast::token_tree],
                         lhses: ~[@named_match], rhses: ~[@named_match])
    -> mac_result {
        // Which arm's failure should we report? (the one furthest along)
        let mut best_fail_spot = {lo: 0u, hi: 0u, expn_info: none};
        let mut best_fail_msg = ~"internal error: ran no matchers";

        let s_d = cx.parse_sess().span_diagnostic;
        let itr = cx.parse_sess().interner;

        for lhses.eachi() |i, lhs| { // try each arm's matchers
            alt lhs {
              @matched_nonterminal(nt_matchers(mtcs)) => {
                // `none` is because we're not interpolating
                let arg_rdr = new_tt_reader(s_d, itr, none, arg) as reader;
                alt parse(cx.parse_sess(), cx.cfg(), arg_rdr, mtcs) {
                  success(named_matches) => {
                    let rhs = alt rhses[i] { // okay, what's your transcriber?
                      @matched_nonterminal(nt_tt(@tt)) => tt,
                      _ => cx.span_bug(sp, ~"bad thing in rhs")
                    };
                    // rhs has holes ( `$id` and `$(...)` that need filled)
                    let trncbr = new_tt_reader(s_d, itr, some(named_matches),
                                               ~[rhs]);
                    let p = parser(cx.parse_sess(), cx.cfg(),
                                   trncbr as reader, SOURCE_FILE);
                    return mr_expr(p.parse_expr());
                  }
                  failure(sp, msg) => if sp.lo >= best_fail_spot.lo {
                    best_fail_spot = sp;
                    best_fail_msg = msg;
                  }
                }
              }
              _ => cx.bug(~"non-matcher found in parsed lhses")
            }
        }
        cx.span_fatal(best_fail_spot, best_fail_msg);
    }

    let exp = |cx, sp, arg| generic_extension(cx, sp, arg, lhses, rhses);

    return mr_def({
        ident: name,
        ext: expr_tt({expander: exp, span: some(sp)})
    });
}