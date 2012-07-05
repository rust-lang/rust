// Earley-like parser for macros.
import parse::token;
import parse::token::{token, EOF, to_str, whole_nt};
import parse::lexer::*; //resolve bug?
//import parse::lexer::{reader, tt_reader, tt_reader_as_reader};
import parse::parser::{parser,SOURCE_FILE};
//import parse::common::parser_common;
import parse::common::*; //resolve bug?
import parse::parse_sess;
import dvec::{dvec, extensions};
import ast::{matcher, mtc_tok, mtc_rep, mtc_bb, ident};
import ast_util::mk_sp;
import std::map::{hashmap, box_str_hash};

/* This is an Earley-like parser, without support for nonterminals.  This
means that there are no completer or predictor rules, and therefore no need to
store one column per token: instead, there's a set of current Earley items and
a set of next ones. Instead of NTs, we have a special case for Kleene
star. The big-O, in pathological cases, is worse than traditional Earley
parsing, but it's an easier fit for Macro-by-Example-style rules, and I think
the overhead is lower. */


/* to avoid costly uniqueness checks, we require that `mtc_rep` always has a
nonempty body. */

enum matcher_pos_up { /* to break a circularity */
    matcher_pos_up(option<matcher_pos>)
}

fn is_some(&&mpu: matcher_pos_up) -> bool {
    alt mpu {
      matcher_pos_up(none) { false }
      _ { true }
    }
}

type matcher_pos = ~{
    elts: ~[ast::matcher], // maybe should be /&? Need to understand regions.
    sep: option<token>,
    mut idx: uint,
    mut up: matcher_pos_up, // mutable for swapping only
    matches: ~[dvec<@arb_depth>],
    sp_lo: uint,
};

fn copy_up(&& mpu: matcher_pos_up) -> matcher_pos {
    alt mpu {
      matcher_pos_up(some(mp)) { copy mp }
      _ { fail }
    }
}

fn count_names(ms: &[matcher]) -> uint {
    vec::foldl(0u, ms, |ct, m| {
        ct + alt m.node {
          mtc_tok(_) { 0u }
          mtc_rep(more_ms, _, _) { count_names(more_ms) }
          mtc_bb(_,_,_) { 1u }
        }})
}

fn new_matcher_pos(ms: ~[matcher], sep: option<token>, lo: uint)
    -> matcher_pos {
    ~{elts: ms, sep: sep, mut idx: 0u, mut up: matcher_pos_up(none),
      matches: copy vec::from_fn(count_names(ms), |_i| dvec::dvec()),
      sp_lo: lo}
}

/* logically, an arb_depth should contain only one kind of nonterminal */
enum arb_depth { leaf(whole_nt), seq(~[@arb_depth], codemap::span) }

type earley_item = matcher_pos;


fn nameize(&&p_s: parse_sess, ms: ~[matcher], &&res: ~[@arb_depth])
    -> hashmap<ident,@arb_depth> {
    fn n_rec(&&p_s: parse_sess, &&m: matcher, &&res: ~[@arb_depth],
             &&ret_val: hashmap<ident, @arb_depth>) {
        alt m {
          {node: mtc_tok(_), span: _} { }
          {node: mtc_rep(more_ms, _, _), span: _} {
            for more_ms.each() |next_m| { n_rec(p_s, next_m, res, ret_val) };
          }
          {node: mtc_bb(bind_name, _, idx), span: sp} {
            if ret_val.contains_key(bind_name) {
                p_s.span_diagnostic.span_fatal(sp, "Duplicated bind name: "
                                               + *bind_name)
            }
            ret_val.insert(bind_name, res[idx]);
          }
        }
    }
    let ret_val = box_str_hash::<@arb_depth>();
    for ms.each() |m| { n_rec(p_s, m, res, ret_val) }
    ret ret_val;
}

enum parse_result {
    success(hashmap<ident, @arb_depth>),
    failure(codemap::span, str)
}

fn parse(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader, ms: ~[matcher])
    -> parse_result {
    let mut cur_eis = ~[];
    vec::push(cur_eis, new_matcher_pos(ms, none, rdr.peek().sp.lo));

    loop {
        let mut bb_eis = ~[]; // black-box parsed by parser.rs
        let mut next_eis = ~[]; // or proceed normally
        let mut eof_eis = ~[];

        let {tok: tok, sp: sp} = rdr.peek();

        /* we append new items to this while we go */
        while cur_eis.len() > 0u { /* for each Earley Item */
            let mut ei = vec::pop(cur_eis);

            let idx = ei.idx;
            let len = ei.elts.len();

            /* at end of sequence */
            if idx >= len {
                // can't move out of `alt`s, so:
                if is_some(ei.up) {
                    // hack: a matcher sequence is repeating iff it has a
                    // parent (the top level is just a container)


                    // disregard separator, try to go up
                    // (remove this condition to make trailing seps ok)
                    if idx == len {
                        // pop from the matcher position

                        let new_pos = copy_up(ei.up);

                        // update matches (the MBE "parse tree") by appending
                        // each tree as a subtree.

                        // I bet this is a perf problem: we're preemptively
                        // doing a lot of array work that will get thrown away
                        // most of the time.
                        for ei.matches.eachi() |idx, elt| {
                            new_pos.matches[idx]
                                .push(@seq(elt.get(), mk_sp(ei.sp_lo,sp.hi)));
                        }

                        new_pos.idx += 1u;
                        vec::push(cur_eis, new_pos);
                    }

                    // can we go around again?

                    // the *_t vars are workarounds for the lack of unary move
                    alt copy ei.sep {
                      some(t) if idx == len { // we need a separator
                        if tok == t { //pass the separator
                            let ei_t <- ei;
                            ei_t.idx += 1u;
                            vec::push(next_eis, ei_t);
                        }
                      }
                      _ { // we don't need a separator
                        let ei_t <- ei;
                        ei_t.idx = 0u;
                        vec::push(cur_eis, ei_t);
                      }
                    }
                } else {
                    vec::push(eof_eis, ei);
                }
            } else {
                alt copy ei.elts[idx].node {
                  /* need to descend into sequence */
                  mtc_rep(matchers, sep, zero_ok) {
                    if zero_ok {
                        let new_ei = copy ei;
                        new_ei.idx += 1u;
                        vec::push(cur_eis, new_ei);
                    }

                    let matches = vec::map(ei.matches, // fresh, same size:
                                           |_m| dvec::<@arb_depth>());
                    let ei_t <- ei;
                    vec::push(cur_eis, ~{
                        elts: matchers, sep: sep, mut idx: 0u,
                        mut up: matcher_pos_up(some(ei_t)),
                        matches: matches, sp_lo: sp.lo
                    });
                  }
                  mtc_bb(_,_,_) { vec::push(bb_eis, ei) }
                  mtc_tok(t) {
                    let ei_t <- ei;
                    if t == tok { ei_t.idx += 1u; vec::push(next_eis, ei_t)}
                  }
                }
            }
        }

        /* error messages here could be improved with links to orig. rules */
        if tok == EOF {
            if eof_eis.len() == 1u {
                ret success(
                    nameize(sess, ms,
                            vec::map(eof_eis[0u].matches, |dv| dv.pop())));
            } else if eof_eis.len() > 1u {
                ret failure(sp, "Ambiguity: multiple successful parses");
            } else {
                ret failure(sp, "Unexpected end of macro invocation");
            }
        } else {
            if (bb_eis.len() > 0u && next_eis.len() > 0u)
                || bb_eis.len() > 1u {
                let nts = str::connect(vec::map(bb_eis, |ei| {
                    alt ei.elts[ei.idx].node
                        { mtc_bb(_,name,_) { *name } _ { fail; } }
                }), " or ");
                ret failure(sp, #fmt[
                    "Local ambiguity: multiple parsing options: \
                     built-in NTs %s or %u other options.",
                    nts, next_eis.len()]);
            } else if (bb_eis.len() == 0u && next_eis.len() == 0u) {
                failure(sp, "No rules expected the token "
                        + to_str(*rdr.interner(), tok));
            } else if (next_eis.len() > 0u) {
                /* Now process the next token */
                while(next_eis.len() > 0u) {
                    vec::push(cur_eis, vec::pop(next_eis));
                }
                rdr.next_token();
            } else /* bb_eis.len() == 1 */ {
                let rust_parser = parser(sess, cfg, rdr.dup(), SOURCE_FILE);

                let ei = vec::pop(bb_eis);
                alt ei.elts[ei.idx].node {
                  mtc_bb(_, name, idx) {
                    ei.matches[idx].push(@leaf(
                        parse_nt(rust_parser, *name)));
                    ei.idx += 1u;
                  }
                  _ { fail; }
                }
                vec::push(cur_eis,ei);

                /* this would fail if zero-length tokens existed */
                while rdr.peek().sp.lo < rust_parser.span.lo {
                    rdr.next_token();
                }
            }
        }

        assert cur_eis.len() > 0u;
    }
}

fn parse_nt(p: parser, name: str) -> whole_nt {
    alt name {
      "item" { alt p.parse_item(~[], ast::public) {
        some(i) { token::w_item(i) }
        none { p.fatal("expected an item keyword") }
      }}
      "block" { token::w_block(p.parse_block()) }
      "stmt" { token::w_stmt(p.parse_stmt(~[])) }
      "pat" { token::w_pat(p.parse_pat()) }
      "expr" { token::w_expr(p.parse_expr()) }
      "ty" { token::w_ty(p.parse_ty(false /* no need to disambiguate*/)) }
      // this could be handled like a token, since it is one
      "ident" { alt copy p.token {
          token::IDENT(sn,b) { p.bump(); token::w_ident(sn,b) }
          _ { p.fatal("expected ident, found "
                      + token::to_str(*p.reader.interner(), copy p.token)) }
      } }
      "path" { token::w_path(p.parse_path_with_tps(false)) }
      "tt" { token::w_tt(p.parse_token_tree()) }
      _ { p.fatal("Unsupported builtin nonterminal parser: " + name)}
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
