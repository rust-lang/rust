// Earley-like parser for macros.
use parse::token;
use parse::token::{Token, EOF, to_str, nonterminal};
use parse::lexer::*; //resolve bug?
//import parse::lexer::{reader, tt_reader, tt_reader_as_reader};
use parse::parser::{Parser, SOURCE_FILE};
//import parse::common::parser_common;
use parse::common::*; //resolve bug?
use parse::parse_sess;
use dvec::DVec;
use ast::{matcher, match_tok, match_seq, match_nonterminal, ident};
use ast_util::mk_sp;
use std::map::HashMap;

/* This is an Earley-like parser, without support for in-grammar nonterminals,
only by calling out to the main rust parser for named nonterminals (which it
commits to fully when it hits one in a grammar). This means that there are no
completer or predictor rules, and therefore no need to store one column per
token: instead, there's a set of current Earley items and a set of next
ones. Instead of NTs, we have a special case for Kleene star. The big-O, in
pathological cases, is worse than traditional Earley parsing, but it's an
easier fit for Macro-by-Example-style rules, and I think the overhead is
lower. (In order to prevent the pathological case, we'd need to lazily
construct the resulting `named_match`es at the very end. It'd be a pain,
and require more memory to keep around old items, but it would also save
overhead)*/

/* Quick intro to how the parser works:

A 'position' is a dot in the middle of a matcher, usually represented as a
dot. For example `· a $( a )* a b` is a position, as is `a $( · a )* a b`.

The parser walks through the input a character at a time, maintaining a list
of items consistent with the current position in the input string: `cur_eis`.

As it processes them, it fills up `eof_eis` with items that would be valid if
the macro invocation is now over, `bb_eis` with items that are waiting on
a Rust nonterminal like `$e:expr`, and `next_eis` with items that are waiting
on the a particular token. Most of the logic concerns moving the · through the
repetitions indicated by Kleene stars. It only advances or calls out to the
real Rust parser when no `cur_eis` items remain

Example: Start parsing `a a a a b` against [· a $( a )* a b].

Remaining input: `a a a a b`
next_eis: [· a $( a )* a b]

- - - Advance over an `a`. - - -

Remaining input: `a a a b`
cur: [a · $( a )* a b]
Descend/Skip (first item).
next: [a $( · a )* a b]  [a $( a )* · a b].

- - - Advance over an `a`. - - -

Remaining input: `a a b`
cur: [a $( a · )* a b]  next: [a $( a )* a · b]
Finish/Repeat (first item)
next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]

- - - Advance over an `a`. - - - (this looks exactly like the last step)

Remaining input: `a b`
cur: [a $( a · )* a b]  next: [a $( a )* a · b]
Finish/Repeat (first item)
next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]

- - - Advance over an `a`. - - - (this looks exactly like the last step)

Remaining input: `b`
cur: [a $( a · )* a b]  next: [a $( a )* a · b]
Finish/Repeat (first item)
next: [a $( a )* · a b]  [a $( · a )* a b]

- - - Advance over a `b`. - - -

Remaining input: ``
eof: [a $( a )* a b ·]

 */


/* to avoid costly uniqueness checks, we require that `match_seq` always has a
nonempty body. */

enum matcher_pos_up { /* to break a circularity */
    matcher_pos_up(Option<matcher_pos>)
}

fn is_some(&&mpu: matcher_pos_up) -> bool {
    match mpu {
      matcher_pos_up(None) => false,
      _ => true
    }
}

type matcher_pos = ~{
    elts: ~[ast::matcher], // maybe should be /&? Need to understand regions.
    sep: Option<Token>,
    mut idx: uint,
    mut up: matcher_pos_up, // mutable for swapping only
    matches: ~[DVec<@named_match>],
    match_lo: uint, match_hi: uint,
    sp_lo: uint,
};

fn copy_up(&& mpu: matcher_pos_up) -> matcher_pos {
    match mpu {
      matcher_pos_up(Some(mp)) => copy mp,
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
fn initial_matcher_pos(ms: ~[matcher], sep: Option<Token>, lo: uint)
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
    ~{elts: ms, sep: sep, mut idx: 0u, mut up: matcher_pos_up(None),
      matches: copy vec::from_fn(count_names(ms), |_i| dvec::DVec()),
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
    -> HashMap<ident,@named_match> {
    fn n_rec(p_s: parse_sess, m: matcher, res: ~[@named_match],
             ret_val: HashMap<ident, @named_match>) {
        match m {
          {node: match_tok(_), span: _} => (),
          {node: match_seq(more_ms, _, _, _, _), span: _} => {
            for more_ms.each() |next_m| { n_rec(p_s, *next_m, res, ret_val) };
          }
          {node: match_nonterminal(bind_name, _, idx), span: sp} => {
            if ret_val.contains_key(bind_name) {
                p_s.span_diagnostic.span_fatal(sp, ~"Duplicated bind name: "+
                                               *p_s.interner.get(bind_name))
            }
            ret_val.insert(bind_name, res[idx]);
          }
        }
    }
    let ret_val = HashMap();
    for ms.each() |m| { n_rec(p_s, *m, res, ret_val) }
    return ret_val;
}

enum parse_result {
    success(HashMap<ident, @named_match>),
    failure(codemap::span, ~str),
    error(codemap::span, ~str)
}

fn parse_or_else(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader,
                 ms: ~[matcher]) -> HashMap<ident, @named_match> {
    match parse(sess, cfg, rdr, ms) {
      success(m) => m,
      failure(sp, str) => sess.span_diagnostic.span_fatal(sp, str),
      error(sp, str) => sess.span_diagnostic.span_fatal(sp, str)
    }
}

fn parse(sess: parse_sess, cfg: ast::crate_cfg, rdr: reader, ms: ~[matcher])
    -> parse_result {
    let mut cur_eis = ~[];
    cur_eis.push(initial_matcher_pos(ms, None, rdr.peek().sp.lo));

    loop {
        let mut bb_eis = ~[]; // black-box parsed by parser.rs
        let mut next_eis = ~[]; // or proceed normally
        let mut eof_eis = ~[];

        let {tok: tok, sp: sp} = rdr.peek();

        /* we append new items to this while we go */
        while cur_eis.len() > 0u { /* for each Earley Item */
            let mut ei = cur_eis.pop();

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

                        new_pos.idx += 1;
                        cur_eis.push(move new_pos);
                    }

                    // can we go around again?

                    // the *_t vars are workarounds for the lack of unary move
                    match copy ei.sep {
                      Some(t) if idx == len => { // we need a separator
                        if tok == t { //pass the separator
                            let ei_t <- ei;
                            ei_t.idx += 1;
                            next_eis.push(move ei_t);
                        }
                      }
                      _ => { // we don't need a separator
                        let ei_t <- ei;
                        ei_t.idx = 0;
                        cur_eis.push(move ei_t);
                      }
                    }
                } else {
                    eof_eis.push(move ei);
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

                        cur_eis.push(move new_ei);
                    }

                    let matches = vec::map(ei.matches, // fresh, same size:
                                           |_m| DVec::<@named_match>());
                    let ei_t <- ei;
                    cur_eis.push(~{
                        elts: matchers, sep: sep, mut idx: 0u,
                        mut up: matcher_pos_up(Some(move ei_t)),
                        matches: move matches,
                        match_lo: match_idx_lo, match_hi: match_idx_hi,
                        sp_lo: sp.lo
                    });
                  }
                  match_nonterminal(_,_,_) => { bb_eis.push(move ei) }
                  match_tok(t) => {
                    let ei_t <- ei;
                    if t == tok {
                        ei_t.idx += 1;
                        next_eis.push(move ei_t);
                    }
                  }
                }
            }
        }

        /* error messages here could be improved with links to orig. rules */
        if tok == EOF {
            if eof_eis.len() == 1u {
                return success(
                    nameize(sess, ms,
                            eof_eis[0u].matches.map(|dv| dv.pop())));
            } else if eof_eis.len() > 1u {
                return error(sp, ~"Ambiguity: multiple successful parses");
            } else {
                return failure(sp, ~"Unexpected end of macro invocation");
            }
        } else {
            if (bb_eis.len() > 0u && next_eis.len() > 0u)
                || bb_eis.len() > 1u {
                let nts = str::connect(vec::map(bb_eis, |ei| {
                    match ei.elts[ei.idx].node {
                      match_nonterminal(bind,name,_) => {
                        fmt!("%s ('%s')", *sess.interner.get(name),
                             *sess.interner.get(bind))
                      }
                      _ => fail
                    } }), ~" or ");
                return error(sp, fmt!(
                    "Local ambiguity: multiple parsing options: \
                     built-in NTs %s or %u other options.",
                    nts, next_eis.len()));
            } else if (bb_eis.len() == 0u && next_eis.len() == 0u) {
                return failure(sp, ~"No rules expected the token: "
                            + to_str(rdr.interner(), tok));
            } else if (next_eis.len() > 0u) {
                /* Now process the next token */
                while(next_eis.len() > 0u) {
                    cur_eis.push(next_eis.pop());
                }
                rdr.next_token();
            } else /* bb_eis.len() == 1 */ {
                let rust_parser = Parser(sess, cfg, rdr.dup(), SOURCE_FILE);

                let ei = bb_eis.pop();
                match ei.elts[ei.idx].node {
                  match_nonterminal(_, name, idx) => {
                    ei.matches[idx].push(@matched_nonterminal(
                        parse_nt(rust_parser, *sess.interner.get(name))));
                    ei.idx += 1u;
                  }
                  _ => fail
                }
                cur_eis.push(move ei);

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

fn parse_nt(p: Parser, name: ~str) -> nonterminal {
    match name {
      ~"item" => match p.parse_item(~[]) {
        Some(i) => token::nt_item(i),
        None => p.fatal(~"expected an item keyword")
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
                     + token::to_str(p.reader.interner(), copy p.token))
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
