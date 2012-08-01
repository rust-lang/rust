import base::{ext_ctxt, mac_result, mr_expr, mr_def, expr_tt};
import codemap::span;
import ast::{ident, matcher_, matcher, match_tok,
             match_nonterminal, match_seq, tt_delim};
import parse::lexer::{new_tt_reader, tt_reader_as_reader, reader};
import parse::token::{FAT_ARROW, SEMI, LBRACE, RBRACE, nt_matchers, nt_tt};
import parse::parser::{parser, SOURCE_FILE};
import earley_parser::{parse, success, failure, named_match,
                       matched_seq, matched_nonterminal};
import std::map::hashmap;



fn add_new_extension(cx: ext_ctxt, sp: span, name: ident,
                     arg: ~[ast::token_tree]) -> base::mac_result {
    // these spans won't matter, anyways
    fn ms(m: matcher_) -> matcher {
        {node: m, span: {lo: 0u, hi: 0u, expn_info: none}}
    }

    let argument_gram = ~[
        ms(match_seq(~[
            ms(match_nonterminal(@~"lhs",@~"matchers", 0u)),
            ms(match_tok(FAT_ARROW)),
            ms(match_nonterminal(@~"rhs",@~"tt", 1u)),
        ], some(SEMI), false, 0u, 2u))];

    let arg_reader = new_tt_reader(cx.parse_sess().span_diagnostic,
                                   cx.parse_sess().interner, none, arg);
    let arguments = alt parse(cx.parse_sess(), cx.cfg(),
                              arg_reader as reader, argument_gram) {
      success(m) { m }
      failure(sp, msg) { cx.span_fatal(sp, msg); }
    };

    let lhses = alt arguments.get(@~"lhs") {
      @matched_seq(s, sp) { s }
      _ { cx.span_bug(sp, ~"wrong-structured lhs") }
    };
    let rhses = alt arguments.get(@~"rhs") {
      @matched_seq(s, sp) { s }
      _ { cx.span_bug(sp, ~"wrong-structured rhs") }
    };

    fn generic_extension(cx: ext_ctxt, sp: span, arg: ~[ast::token_tree],
                         lhses: ~[@named_match], rhses: ~[@named_match])
    -> mac_result {
        let mut best_fail_spot = {lo: 0u, hi: 0u, expn_info: none};
        let mut best_fail_msg = ~"internal error: ran no matchers";

        let s_d = cx.parse_sess().span_diagnostic;
        let itr = cx.parse_sess().interner;

        for lhses.eachi() |i, lhs| {
            alt lhs {
              @matched_nonterminal(nt_matchers(mtcs)) {
                let arg_rdr = new_tt_reader(s_d, itr, none, arg) as reader;
                alt parse(cx.parse_sess(), cx.cfg(), arg_rdr, mtcs) {
                  success(m) {
                    let rhs = alt rhses[i] {
                      @matched_nonterminal(nt_tt(@tt)) { tt }
                      _ { cx.span_bug(sp, ~"bad thing in rhs") }
                    };
                    let trncbr = new_tt_reader(s_d, itr, some(m), ~[rhs]);
                    let p = parser(cx.parse_sess(), cx.cfg(),
                                   trncbr as reader, SOURCE_FILE);
                    ret mr_expr(p.parse_expr());
                  }
                  failure(sp, msg) {
                    if sp.lo >= best_fail_spot.lo {
                        best_fail_spot = sp; best_fail_msg = msg;
                    }
                  }
                }
              }
              _ { cx.bug(~"non-matcher found in parsed lhses"); }
            }
        }
        cx.span_fatal(best_fail_spot, best_fail_msg);
    }

    let exp = |cx, sp, arg| generic_extension(cx, sp, arg, lhses, rhses);

    ret mr_def({ident: name, ext: expr_tt({expander: exp, span: some(sp)})});
}