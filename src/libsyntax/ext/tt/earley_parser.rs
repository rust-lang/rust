// Earley-like parser for macros.
import parse::token;
import parse::token::{token, EOF, to_str, nonterminal};
import parse::lexer::*; //resolve bug?
//import parse::lexer::{reader, tt_reader, tt_reader_as_reader};
import parse::parser::{parser,SOURCE_FILE};
//import parse::common::parser_common;
import parse::common::*; //resolve bug?
import parse::parse_sess;
import dvec::dvec;
import ast::{matcher, match_tok, match_seq, match_nonterminal, ident};
import ast_util::mk_sp;
import std::map::{hashmap, box_str_hash};

/* This is an Earley-like parser, without support for in-grammar nonterminals,
onlyl calling out to the main rust parser for named nonterminals (which it
commits to fully when it hits one in a grammar). This means that there are no
completer or predictor rules, and therefore no need to store one column per
token: instead, there's a set of current Earley items and a set of next
ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
pathological cases, is worse than traditional Earley parsing, but it's an
easier fit for Macro-by-Example-style rules, and I think the overhead is
lower. */


/* to avoid costly uniqueness checks, we require that `match_seq` always has a
nonempty body. */

enum matcher_pos_up { /* to break a circularity */
    matcher_pos_up(option<matcher_pos>)
}

fn is_some(&&mpu: matcher_pos_up) -> bool {
    match mpu {
      matcher_pos_up(none) => false,
      _ => true
    }
}

type matcher_pos = ~{
    elts: ~[ast::matcher], // maybe should be /&? Need to understand regions.
    sep: option<token>,
    mut idx: uint,
    mut up: matcher_pos_up, // mutable for swapping only
    matches: ~[dvec<@named_match>],
    match_lo: uint, match_hi: uint,
    sp_lo: uint,
};

fn copy_up(&& mpu: matcher_pos_up) -> matcher_pos {
    match mpu {
      matcher_pos_up(some(mp)) => copy mp,
      _ => fail
    }
}

fn count_names(ms: &[matcher]) -> uint {
    vec::foldl(0u, ms, |ct, m| {
        ct + match m.node {
          match_tok(_) => 0u,
          match_seq(more_ms, _, _, _, _) => count_names(more_ms),
          match_nonterminal(_,_,_) => 1u
        }})
}

#[allow(non_implicitly_copyable_typarams)]
fn initial_matcher_pos(ms: ~[matcher], sep: option<token>, lo: uint)
    -> matcher_pos {
    let mut match_idx_hi = 0u;
    for ms.each() |elt| {
        match elt.node {
          match_tok(_) => (),
          match_seq(_,_,_,_,hi) => {
            match_idx_hi = hi;       // it is monotonic...
          }
          match_nonterminal(_,_,pos) => {
            match_idx_hi = pos+1u;  // ...so latest is highest
          }
        }
    }
    ~{elts: ms, sep: sep, mut idx: 0u, mut up: matcher_pos_up(none),
      matches: copy vec::from_fn(count_names(ms), |_i| dvec::dvec()),
      match_lo: 0u, match_hi: match_idx_hi, sp_lo: lo}
}

// named_match is a pattern-match result for a single ast::match_nonterminal:
// so it is associated with a single ident in a parse, and all
// matched_nonterminals in the named_match have the same nonterminal type
// (expr, item, etc). All the leaves in a single named_match correspond to a
// single matcher_nonterminal in the ast::matcher that produced it.
//
// It should probably be renamed, it has more or less exact correspondence to
// ast::match nodes, and the in-memory structure of a particular named_match
// represents the match that occurred when a particular subset of an
// ast::match -- those ast::matcher nodes leading to a single
// match_nonterminal -- was applied to a particular token tree.
//
// The width of each matched_seq in the named_match, and the identity of the
// matched_nonterminals, will depend on the token tree it was applied to: each
// matched_seq corresponds to a single match_seq in the originating
// ast::matcher. The depth of the named_match structure will therefore depend
// only on the nesting depth of ast::match_seqs in the originating
// ast::matcher it was derived from.

enum named_match {
    matched_seq(~[@named_match], codemap::span),
    matched_nonterminal(nonterminal)
}

type earley_item = matcher_pos;

fn nameize(p_s: parse_sess, ms: ~[matcher], res: ~[@named_match])
    -> hashmap<ident,@named_match> {
    fn n_rec(p_s: parse_sess, m: matcher, res: ~[@named_match],
             ret_val: hashmap<ident, @named_match>) {
        match m {
          {node: match_tok(_), span: _} => (),
          {node: match_seq(more_ms, _, _, _, _), span: _} => {
            for more_ms.each() |next_m| { n_rec(p_s, next_m, res, ret_val) };
          }
          {node: match_nonterminal(bind_name, _, idx), span: sp} => {
            if ret_val.contains_key(bind_name) {
                p_s.span_diagnostic.span_fatal(sp, ~"Duplicated bind name: "
                                               + *bind_name)
            }
            ret_val.insert(bind_name, res[idx]);
          }
        }
    }
    let ret_val = box_str_hash::<@named_match>();
    for ms.each() |m| { n_rec(p_s, m, res, ret_val) }
    return ret_val;
}

enum parse_result {
    success(hashmap<ident, @named_match>),
    failure(codemap::span, ~str)
}

fn parse_or_else(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader,
                 ms: ~[matcher]) -> hashmap<ident, @named_match> {
    match parse(sess, cfg, rdr, ms) {
      success(m) => m,
      failure(sp, str) => sess.span_diagnostic.span_fatal(sp, str)
    }
}

fn parse(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader, ms: ~[matcher])
    -> parse_result {
    let mut cur_eis = ~[];
    vec::push(cur_eis, initial_matcher_pos(ms, none, rdr.peek().sp.lo));

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

                        // Only touch the binders we have actually bound
                        for uint::range(ei.match_lo, ei.match_hi) |idx| {
                            let sub = ei.matches[idx].get();
                            new_pos.matches[idx]
                                .push(@matched_seq(sub,
                                                   mk_sp(ei.sp_lo,
                                                         sp.hi)));
                        }

                        new_pos.idx += 1u;
                        vec::push(cur_eis, new_pos);
                    }

                    // can we go around again?

                    // the *_t vars are workarounds for the lack of unary move
                    match copy ei.sep {
                      some(t) if idx == len => { // we need a separator
                        if tok == t { //pass the separator
                            let ei_t <- ei;
                            ei_t.idx += 1u;
                            vec::push(next_eis, ei_t);
                        }
                      }
                      _ => { // we don't need a separator
                        let ei_t <- ei;
                        ei_t.idx = 0u;
                        vec::push(cur_eis, ei_t);
                      }
                    }
                } else {
                    vec::push(eof_eis, ei);
                }
            } else {
                match copy ei.elts[idx].node {
                  /* need to descend into sequence */
                  match_seq(matchers, sep, zero_ok,
                            match_idx_lo, match_idx_hi) => {
                    if zero_ok {
                        let new_ei = copy ei;
                        new_ei.idx += 1u;
                        //we specifically matched zero repeats.
                        for uint::range(match_idx_lo, match_idx_hi) |idx| {
                            new_ei.matches[idx].push(@matched_seq(~[], sp));
                        }

                        vec::push(cur_eis, new_ei);
                    }

                    let matches = vec::map(ei.matches, // fresh, same size:
                                           |_m| dvec::<@named_match>());
                    let ei_t <- ei;
                    vec::push(cur_eis, ~{
                        elts: matchers, sep: sep, mut idx: 0u,
                        mut up: matcher_pos_up(some(ei_t)),
                        matches: matches,
                        match_lo: match_idx_lo, match_hi: match_idx_hi,
                        sp_lo: sp.lo
                    });
                  }
                  match_nonterminal(_,_,_) => { vec::push(bb_eis, ei) }
                  match_tok(t) => {
                    let ei_t <- ei;
                    if t == tok { ei_t.idx += 1u; vec::push(next_eis, ei_t)}
                  }
                }
            }
        }

        /* error messages here could be improved with links to orig. rules */
        if tok == EOF {
            if eof_eis.len() == 1u {
                return success(
                    nameize(sess, ms,
                            vec::map(eof_eis[0u].matches, |dv| dv.pop())));
            } else if eof_eis.len() > 1u {
                return failure(sp, ~"Ambiguity: multiple successful parses");
            } else {
                return failure(sp, ~"Unexpected end of macro invocation");
            }
        } else {
            if (bb_eis.len() > 0u && next_eis.len() > 0u)
                || bb_eis.len() > 1u {
                let nts = str::connect(vec::map(bb_eis, |ei| {
                    match ei.elts[ei.idx].node {
                      match_nonterminal(bind,name,_) => {
                        fmt!{"%s ('%s')", *name, *bind}
                      }
                      _ => fail
                    } }), ~" or ");
                return failure(sp, fmt!{
                    "Local ambiguity: multiple parsing options: \
                     built-in NTs %s or %u other options.",
                    nts, next_eis.len()});
            } else if (bb_eis.len() == 0u && next_eis.len() == 0u) {
                return failure(sp, ~"No rules expected the token "
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
                match ei.elts[ei.idx].node {
                  match_nonterminal(_, name, idx) => {
                    ei.matches[idx].push(@matched_nonterminal(
                        parse_nt(rust_parser, *name)));
                    ei.idx += 1u;
                  }
                  _ => fail
                }
                vec::push(cur_eis,ei);

                /* this would fail if zero-length tokens existed */
                while rdr.peek().sp.lo < rust_parser.span.lo {
                    rdr.next_token();
                } /* except for EOF... */
                while rust_parser.token == EOF && rdr.peek().tok != EOF {
                    rdr.next_token();
                }
            }
        }

        assert cur_eis.len() > 0u;
    }
}

fn parse_nt(p: parser, name: ~str) -> nonterminal {
    match name {
      ~"item" => match p.parse_item(~[]) {
        some(i) => token::nt_item(i),
        none => p.fatal(~"expected an item keyword")
      },
      ~"block" => token::nt_block(p.parse_block()),
      ~"stmt" => token::nt_stmt(p.parse_stmt(~[])),
      ~"pat" => token::nt_pat(p.parse_pat(true)),
      ~"expr" => token::nt_expr(p.parse_expr()),
      ~"ty" => token::nt_ty(p.parse_ty(false /* no need to disambiguate*/)),
      // this could be handled like a token, since it is one
      ~"ident" => match copy p.token {
        token::IDENT(sn,b) => { p.bump(); token::nt_ident(sn,b) }
        _ => p.fatal(~"expected ident, found "
                     + token::to_str(*p.reader.interner(), copy p.token))
      },
      ~"path" => token::nt_path(p.parse_path_with_tps(false)),
      ~"tt" => {
        p.quote_depth += 1u; //but in theory, non-quoted tts might be useful
        let res = token::nt_tt(@p.parse_token_tree());
        p.quote_depth -= 1u;
        res
      }
      ~"matchers" => token::nt_matchers(p.parse_matchers()),
      _ => p.fatal(~"Unsupported builtin nonterminal parser: " + name)
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
