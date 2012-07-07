import util::interner::interner;
import diagnostic::span_handler;
import ast::{token_tree,tt_delim,tt_flat,tt_dotdotdot,tt_interpolate,ident};
import earley_parser::{arb_depth,seq,leaf};
import codemap::span;
import parse::token::{EOF,ACTUALLY,IDENT,token,w_ident};
import std::map::{hashmap,box_str_hash};

export tt_reader,  new_tt_reader, dup_tt_reader, tt_next_token;

enum tt_frame_up { /* to break a circularity */
    tt_frame_up(option<tt_frame>)
}

/* TODO: figure out how to have a uniquely linked stack, and change to `~` */
///an unzipping of `token_tree`s
type tt_frame = @{
    readme: ~[ast::token_tree],
    mut idx: uint,
    dotdotdoted: bool,
    sep: option<token>,
    up: tt_frame_up,
};

type tt_reader = @{
    sp_diag: span_handler,
    interner: @interner<@str>,
    mut cur: tt_frame,
    /* for MBE-style macro transcription */
    interpolations: std::map::hashmap<ident, @arb_depth>,
    mut repeat_idx: ~[mut uint],
    mut repeat_len: ~[uint],
    /* cached: */
    mut cur_tok: token,
    mut cur_span: span
};

/** This can do Macro-By-Example transcription. On the other hand, if
 *  `src` contains no `tt_dotdotdot`s and `tt_interpolate`s, `interp` can (and
 *  should) be none. */
fn new_tt_reader(sp_diag: span_handler, itr: @interner<@str>,
                 interp: option<std::map::hashmap<ident,@arb_depth>>,
                 src: ~[ast::token_tree])
    -> tt_reader {
    let r = @{sp_diag: sp_diag, interner: itr,
              mut cur: @{readme: src, mut idx: 0u, dotdotdoted: false,
                         sep: none, up: tt_frame_up(option::none)},
              interpolations: alt interp { /* just a convienience */
                none { std::map::box_str_hash::<@arb_depth>() }
                some(x) { x }
              },
              mut repeat_idx: ~[mut], mut repeat_len: ~[],
              /* dummy values, never read: */
              mut cur_tok: EOF,
              mut cur_span: ast_util::mk_sp(0u,0u)
             };
    tt_next_token(r); /* get cur_tok and cur_span set up */
    ret r;
}

pure fn dup_tt_frame(&&f: tt_frame) -> tt_frame {
    @{readme: f.readme, mut idx: f.idx, dotdotdoted: f.dotdotdoted,
      sep: f.sep, up: alt f.up {
        tt_frame_up(some(up_frame)) {
          tt_frame_up(some(dup_tt_frame(up_frame)))
        }
        tt_frame_up(none) { tt_frame_up(none) }
      }
     }
}

pure fn dup_tt_reader(&&r: tt_reader) -> tt_reader {
    @{sp_diag: r.sp_diag, interner: r.interner,
      mut cur: dup_tt_frame(r.cur),
      interpolations: r.interpolations,
      mut repeat_idx: copy r.repeat_idx, mut repeat_len: copy r.repeat_len,
      mut cur_tok: r.cur_tok, mut cur_span: r.cur_span}
}


pure fn lookup_cur_ad_by_ad(r: tt_reader, start: @arb_depth) -> @arb_depth {
    pure fn red(&&ad: @arb_depth, &&idx: uint) -> @arb_depth {
        alt *ad {
          leaf(_) { ad /* end of the line; duplicate henceforth */ }
          seq(ads, _) { ads[idx] }
        }
    }
    vec::foldl(start, r.repeat_idx, red)
}

fn lookup_cur_ad(r: tt_reader, name: ident) -> @arb_depth {
    lookup_cur_ad_by_ad(r, r.interpolations.get(name))
}
enum lis {
    lis_unconstrained, lis_constraint(uint, ident), lis_contradiction(str)
}

fn lockstep_iter_size(&&t: token_tree, &&r: tt_reader) -> lis {
    fn lis_merge(lhs: lis, rhs: lis) -> lis {
        alt lhs {
          lis_unconstrained { rhs }
          lis_contradiction(_) { lhs }
          lis_constraint(l_len, l_id) {
            alt rhs {
              lis_unconstrained { lhs }
              lis_contradiction(_) { rhs }
              lis_constraint(r_len, _) if l_len == r_len { lhs }
              lis_constraint(r_len, r_id) {
                lis_contradiction(#fmt["Inconsistent lockstep iteration: \
                                        '%s' has %u items, but '%s' has %u",
                                       *l_id, l_len, *r_id, r_len])
              }
            }
          }
        }
    }
    alt t {
      tt_delim(tts) | tt_dotdotdot(_, tts, _, _) {
        vec::foldl(lis_unconstrained, tts, {|lis, tt|
            lis_merge(lis, lockstep_iter_size(tt, r)) })
      }
      tt_flat(*) { lis_unconstrained }
      tt_interpolate(_, name) {
        alt *lookup_cur_ad(r, name) {
          leaf(_) { lis_unconstrained }
          seq(ads, _) { lis_constraint(ads.len(), name) }
        }
      }
    }
}


fn tt_next_token(&&r: tt_reader) -> {tok: token, sp: span} {
    let ret_val = { tok: r.cur_tok, sp: r.cur_span };
    while r.cur.idx >= vec::len(r.cur.readme) {
        /* done with this set; pop or repeat? */
        if ! r.cur.dotdotdoted
            || r.repeat_idx.last() == r.repeat_len.last() - 1 {

            alt r.cur.up {
              tt_frame_up(none) {
                r.cur_tok = EOF;
                ret ret_val;
              }
              tt_frame_up(some(tt_f)) {
                if r.cur.dotdotdoted {
                    vec::pop(r.repeat_idx); vec::pop(r.repeat_len);
                }

                r.cur = tt_f;
                r.cur.idx += 1u;
              }
            }

        } else { /* repeat */
            r.cur.idx = 0u;
            r.repeat_idx[r.repeat_idx.len() - 1u] += 1u;
            alt r.cur.sep {
              some(tk) {
                r.cur_tok = tk; /* repeat same span, I guess */
                ret ret_val;
              }
              none {}
            }
        }
    }
    loop { /* because it's easiest, this handles `tt_delim` not starting
    with a `tt_flat`, even though it won't happen */
        alt r.cur.readme[r.cur.idx] {
          tt_delim(tts) {
            r.cur = @{readme: tts, mut idx: 0u, dotdotdoted: false,
                      sep: none, up: tt_frame_up(option::some(r.cur)) };
            // if this could be 0-length, we'd need to potentially recur here
          }
          tt_flat(sp, tok) {
            r.cur_span = sp; r.cur_tok = tok;
            r.cur.idx += 1u;
            ret ret_val;
          }
          tt_dotdotdot(sp, tts, sep, zerok) {
            alt lockstep_iter_size(tt_dotdotdot(sp, tts, sep, zerok), r) {
              lis_unconstrained {
                r.sp_diag.span_fatal(
                    sp, /* blame macro writer */
                    "attempted to repeat an expression containing no syntax \
                     variables matched as repeating at this depth");
              }
              lis_contradiction(msg) { /* TODO blame macro invoker instead*/
                r.sp_diag.span_fatal(sp, msg);
              }
              lis_constraint(len, _) {
                vec::push(r.repeat_len, len);
                vec::push(r.repeat_idx, 0u);
                r.cur = @{readme: tts, mut idx: 0u, dotdotdoted: true,
                          sep: sep, up: tt_frame_up(option::some(r.cur)) };

                if len == 0 {
                    if !zerok {
                        r.sp_diag.span_fatal(sp, /* TODO blame invoker */
                                             "this must repeat at least \
                                              once");
                    }
                    /* we need to pop before we proceed, so recur */
                    ret tt_next_token(r);
                }
              }
            }
          }
          // TODO: think about span stuff here
          tt_interpolate(sp, ident) {
            alt *lookup_cur_ad(r, ident) {
              /* sidestep the interpolation tricks for ident because
              (a) idents can be in lots of places, so it'd be a pain
              (b) we actually can, since it's a token. */
              leaf(w_ident(sn,b)) {
                r.cur_span = sp; r.cur_tok = IDENT(sn,b);
                r.cur.idx += 1u;
                ret ret_val;
              }
              leaf(w_nt) {
                r.cur_span = sp; r.cur_tok = ACTUALLY(w_nt);
                r.cur.idx += 1u;
                ret ret_val;
              }
              seq(*) {
                r.sp_diag.span_fatal(
                    copy r.cur_span, /* blame the macro writer */
                    #fmt["variable '%s' is still repeating at this depth",
                         *ident]);
              }
            }
          }
        }
    }

}