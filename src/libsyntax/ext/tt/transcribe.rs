import util::interner::interner;
import diagnostic::span_handler;
import ast::{tt_delim,tt_flat,tt_dotdotdot,tt_interpolate,ident};
import earley_parser::arb_depth;
import codemap::span;
import parse::token::{EOF,token};

export tt_reader,  new_tt_reader, dup_tt_reader, tt_next_token;

enum tt_frame_up { /* to break a circularity */
    tt_frame_up(option<tt_frame>)
}

/* TODO: figure out how to have a uniquely linked stack, and change to `~` */
///an unzipping of `token_tree`s
type tt_frame = @{
    readme: [ast::token_tree]/~,
    mut idx: uint,
    up: tt_frame_up
};

type tt_reader = @{
    span_diagnostic: span_handler,
    interner: @interner<@str>,
    mut cur: tt_frame,
    /* for MBE-style macro transcription */
    interpolations: std::map::hashmap<ident, @arb_depth>,
    /* cached: */
    mut cur_tok: token,
    mut cur_span: span
};

/** This can do Macro-By-Example transcription. On the other hand, if
 *  `doc` contains no `tt_dotdotdot`s and `tt_interpolate`s, `interp` can (and
 *  should) be none. */
fn new_tt_reader(span_diagnostic: span_handler, itr: @interner<@str>,
                 interp: option<std::map::hashmap<ident,@arb_depth>>,
                 src: [ast::token_tree]/~)
    -> tt_reader {
    let r = @{span_diagnostic: span_diagnostic, interner: itr,
              mut cur: @{readme: src, mut idx: 0u,
                         up: tt_frame_up(option::none)},
              interpolations: alt interp { /* just a convienience */
                none { std::map::box_str_hash::<@arb_depth>() }
                some(x) { x }
              },
              /* dummy values, never read: */
              mut cur_tok: EOF,
              mut cur_span: ast_util::mk_sp(0u,0u)
             };
    tt_next_token(r); /* get cur_tok and cur_span set up */
    ret r;
}

pure fn dup_tt_frame(&&f: tt_frame) -> tt_frame {
    @{readme: f.readme, mut idx: f.idx,
      up: alt f.up {
        tt_frame_up(some(up_frame)) {
          tt_frame_up(some(dup_tt_frame(up_frame)))
        }
        tt_frame_up(none) { tt_frame_up(none) }
      }
     }
}

pure fn dup_tt_reader(&&r: tt_reader) -> tt_reader {
    @{span_diagnostic: r.span_diagnostic, interner: r.interner,
      mut cur: dup_tt_frame(r.cur),
      interpolations: r.interpolations,
      mut cur_tok: r.cur_tok, mut cur_span: r.cur_span}
}


fn tt_next_token(&&r: tt_reader) -> {tok: token, sp: span} {
    let ret_val = { tok: r.cur_tok, sp: r.cur_span };
    if r.cur.idx >= vec::len(r.cur.readme) {
        /* done with this set; pop */
        alt r.cur.up {
          tt_frame_up(none) {
            r.cur_tok = EOF;
            ret ret_val;
          }
          tt_frame_up(some(tt_f)) {
            r.cur = tt_f;
            /* the above `if` would need to be a `while` if we didn't know
            that the last thing in a `tt_delim` is always a `tt_flat` */
            r.cur.idx += 1u;
          }
        }
    }
    /* if `tt_delim`s could be 0-length, we'd need to be able to switch
    between popping and pushing until we got to an actual `tt_flat` */
    loop { /* because it's easiest, this handles `tt_delim` not starting
    with a `tt_flat`, even though it won't happen */
        alt copy r.cur.readme[r.cur.idx] {
          tt_delim(tts) {
            r.cur = @{readme: tts, mut idx: 0u,
                      up: tt_frame_up(option::some(r.cur)) };
          }
          tt_flat(sp, tok) {
            r.cur_span = sp; r.cur_tok = tok;
            r.cur.idx += 1u;
            ret ret_val;
          }
          tt_dotdotdot(tts) {
            fail;
          }
          tt_interpolate(ident) {
            fail;
          }
        }
    }

}