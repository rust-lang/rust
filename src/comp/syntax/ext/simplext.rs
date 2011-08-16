use std;

import codemap::span;
import std::ivec;
import std::option;
import std::map::hashmap;
import std::map::new_str_hash;
import option::some;
import option::none;

import base::syntax_extension;
import base::ext_ctxt;
import base::normal;
import base::expr_to_str;
import base::expr_to_ident;

import fold::*;
import ast::node_id;
import ast::respan;
import ast::ident;
import ast::path;
import ast::ty;
import ast::blk;
import ast::blk_;
import ast::expr;
import ast::expr_;
import ast::path_;
import ast::expr_path;
import ast::expr_vec;
import ast::expr_mac;
import ast::mac_invoc;

export add_new_extension;

fn path_to_ident(pth: &path) -> option::t[ident] {
    if ivec::len(pth.node.idents) == 1u && ivec::len(pth.node.types) == 0u {
        ret some(pth.node.idents.(0u));
    }
    ret none;
}

//an ivec of binders might be a little big.
type clause = {params: binders, body: @expr};

/* logically, an arb_depth should contain only one kind of matchable */
tag arb_depth[T] { leaf(T); seq(@[arb_depth[T]], span); }


tag matchable {
    match_expr(@expr);
    match_path(path);
    match_ident(ast::spanned[ident]);
    match_ty(@ty);
    match_block(ast::blk);
    match_exact; /* don't bind anything, just verify the AST traversal */
}

/* for when given an incompatible bit of AST */
fn match_error(cx: &ext_ctxt, m: &matchable, expected: &str) -> ! {
    alt m {
      match_expr(x) {
        cx.span_fatal(x.span,
                      "this argument is an expr, expected " + expected);
      }
      match_path(x) {
        cx.span_fatal(x.span,
                      "this argument is a path, expected " + expected);
      }
      match_ident(x) {
        cx.span_fatal(x.span,
                      "this argument is an ident, expected " + expected);
      }
      match_ty(x) {
        cx.span_fatal(x.span,
                      "this argument is a type, expected " + expected);
      }
      match_block(x) {
        cx.span_fatal(x.span,
                      "this argument is a block, expected " + expected);
      }
      match_exact. { cx.bug("what is a match_exact doing in a bindings?"); }
    }
}

// We can't make all the matchables in a match_result the same type because
// idents can be paths, which can be exprs.

// If we want better match failure error messages (like in Fortifying Syntax),
// we'll want to return something indicating amount of progress and location
// of failure instead of `none`.
type match_result = option::t[arb_depth[matchable]];
type selector = fn(&matchable) -> match_result ;

fn elts_to_ell(cx: &ext_ctxt, elts: &[@expr])
    -> {pre: [@expr], rep: option::t[@expr], post: [@expr]} {
    let idx: uint = 0u;
    let res = none;
    for elt: @expr in elts {
        alt elt.node {
          expr_mac(m) {
            alt m.node {
              ast::mac_ellipsis. {
                if res != none {
                    cx.span_fatal(m.span, "only one ellipsis allowed");
                }
                res = some({pre: ivec::slice(elts, 0u, idx - 1u),
                            rep: some(elts.(idx - 1u)),
                            post: ivec::slice(elts, idx + 1u,
                                              ivec::len(elts))});
              }
              _ { }
            }
          }
          _ { }
        }
        idx += 1u;
    }
    ret alt res {
      some(val) { val }
      none. { {pre: elts, rep: none, post: ~[]} }
    }
}

fn option_flatten_map[T, U](f: &fn(&T) -> option::t[U] , v: &[T]) ->
   option::t[[U]] {
    let res = ~[];
    for elem: T in v {
        alt f(elem) { none. { ret none; } some(fv) { res += ~[fv]; } }
    }
    ret some(res);
}

fn a_d_map(ad: &arb_depth[matchable], f: &selector) -> match_result {
    alt ad {
      leaf(x) { ret f(x); }
      seq(ads, span) {
        alt option_flatten_map(bind a_d_map(_, f), *ads) {
          none. { ret none; }
          some(ts) { ret some(seq(@ts, span)); }
        }
      }
    }
}

fn compose_sels(s1: selector, s2: selector) -> selector {
    fn scomp(s1: selector, s2: selector, m: &matchable) -> match_result {
        ret alt s1(m) {
              none. { none }
              some(matches) { a_d_map(matches, s2) }
            }
    }
    ret bind scomp(s1, s2, _);
}



type binders =
    {real_binders: hashmap[ident, selector],
     mutable literal_ast_matchers: [selector]};
type bindings = hashmap[ident, arb_depth[matchable]];

fn acumm_bindings(cx: &ext_ctxt, b_dest: &bindings, b_src: &bindings) { }

/* these three functions are the big moving parts */

/* create the selectors needed to bind and verify the pattern */

fn pattern_to_selectors(cx: &ext_ctxt, e: @expr) -> binders {
    let res: binders =
        {real_binders: new_str_hash[selector](),
         mutable literal_ast_matchers: ~[]};
    //this oughta return binders instead, but macro args are a sequence of
    //expressions, rather than a single expression
    fn trivial_selector(m: &matchable) -> match_result { ret some(leaf(m)); }
    p_t_s_rec(cx, match_expr(e), trivial_selector, res);
    ret res;
}



/* use the selectors on the actual arguments to the macro to extract
bindings. Most of the work is done in p_t_s, which generates the
selectors. */

fn use_selectors_to_bind(b: &binders, e: @expr) -> option::t[bindings] {
    let res = new_str_hash[arb_depth[matchable]]();
    //need to do this first, to check vec lengths.
    for sel: selector in b.literal_ast_matchers {
        alt sel(match_expr(e)) { none. { ret none; } _ { } }
    }
    let never_mind: bool = false;
    for each pair: @{key: ident, val: selector} in b.real_binders.items() {
        alt pair.val(match_expr(e)) {
          none. { never_mind = true; }
          some(mtc) { res.insert(pair.key, mtc); }
        }
    }
    //HACK: `ret` doesn't work in `for each`
    if never_mind { ret none; }
    ret some(res);
}

/* use the bindings on the body to generate the expanded code */

fn transcribe(cx: &ext_ctxt, b: &bindings, body: @expr) -> @expr {
    let idx_path: @mutable [uint] = @mutable ~[];
    fn new_id(old: node_id, cx: &ext_ctxt) -> node_id { ret cx.next_id(); }
    fn new_span(cx: &ext_ctxt, sp: &span) -> span {
        /* this discards information in the case of macro-defining macros */
        ret {lo: sp.lo, hi: sp.hi, expanded_from: cx.backtrace()};
    }
    let afp = default_ast_fold();
    let f_pre =
        {fold_ident: bind transcribe_ident(cx, b, idx_path, _, _),
         fold_path: bind transcribe_path(cx, b, idx_path, _, _),
         fold_expr:
             bind transcribe_expr(cx, b, idx_path, _, _, afp.fold_expr),
         fold_ty: bind transcribe_type(cx, b, idx_path, _, _, afp.fold_ty),
         fold_block:
             bind transcribe_block(cx, b, idx_path, _, _, afp.fold_block),
         map_exprs: bind transcribe_exprs(cx, b, idx_path, _, _),
         new_id: bind new_id(_, cx),
         new_span: bind new_span(cx, _) with *afp};
    let f = make_fold(f_pre);
    let result = f.fold_expr(body);
    dummy_out(f); //temporary: kill circular reference
    ret result;
}



/* helper: descend into a matcher */
fn follow(m: &arb_depth[matchable], idx_path: @mutable [uint]) ->
   arb_depth[matchable] {
    let res: arb_depth[matchable] = m;
    for idx: uint in *idx_path {
        alt res {
          leaf(_) { ret res;/* end of the line */ }
          seq(new_ms, _) { res = new_ms.(idx); }
        }
    }
    ret res;
}

fn follow_for_trans(cx: &ext_ctxt, mmaybe: &option::t[arb_depth[matchable]],
                    idx_path: @mutable [uint]) -> option::t[matchable] {
    alt mmaybe {
      none. { ret none }
      some(m) {
        ret alt follow(m, idx_path) {
              seq(_, sp) {
                cx.span_fatal(sp,
                              "syntax matched under ... but not " +
                                  "used that way.")
              }
              leaf(m) { ret some(m) }
            }
      }
    }

}

/* helper for transcribe_exprs: what vars from `b` occur in `e`? */
iter free_vars(b: &bindings, e: @expr) -> ident {
    let idents: hashmap[ident, ()] = new_str_hash[()]();
    fn mark_ident(i: &ident, fld: ast_fold, b: &bindings,
                  idents: &hashmap[ident, ()]) -> ident {
        if b.contains_key(i) { idents.insert(i, ()); }
        ret i;
    }
    // using fold is a hack: we want visit, but it doesn't hit idents ) :
    // solve this with macros
    let f_pre =
        {fold_ident: bind mark_ident(_, _, b, idents)
            with *default_ast_fold()};
    let f = make_fold(f_pre);
    f.fold_expr(e); // ignore result
    dummy_out(f);
    for each id: ident in idents.keys() { put id; }
}


/* handle sequences (anywhere in the AST) of exprs, either real or ...ed */
fn transcribe_exprs(cx: &ext_ctxt, b: &bindings, idx_path: @mutable [uint],
                    recur: fn(&@expr) -> @expr , exprs: [@expr])
    -> [@expr] {
    alt elts_to_ell(cx, exprs) {
      {pre: pre, rep: repeat_me_maybe, post: post} {
        let res = ivec::map(recur, pre);
        alt repeat_me_maybe {
          none. {}
          some(repeat_me) {
            let repeat: option::t[{rep_count: uint, name: ident}] = none;
            /* we need to walk over all the free vars in lockstep, except for
            the leaves, which are just duplicated */
            for each fv: ident in free_vars(b, repeat_me) {
                let cur_pos = follow(b.get(fv), idx_path);
                alt cur_pos {
                  leaf(_) { }
                  seq(ms, _) {
                    alt repeat {
                      none. {
                        repeat = some({rep_count: ivec::len(*ms), name: fv});
                      }
                      some({rep_count: old_len, name: old_name}) {
                        let len = ivec::len(*ms);
                        if old_len != len {
                            let msg = #fmt("'%s' occurs %u times, but ", fv,
                                           len) + #fmt("'%s' occurs %u times",
                                                       old_name, old_len);
                            cx.span_fatal(repeat_me.span, msg);
                        }
                      }
                    }
                  }
                }
            }
            alt repeat {
              none. {
                cx.span_fatal(repeat_me.span,
                              "'...' surrounds an expression without any" +
                              " repeating syntax variables");
              }
              some({rep_count: rc, _}) {
                /* Whew, we now know how how many times to repeat */
                let idx: uint = 0u;
                while idx < rc {
                    *idx_path += ~[idx];
                    res += ~[recur(repeat_me)]; // whew!
                    ivec::pop(*idx_path);
                    idx += 1u;
                }
              }
            }
          }
        }
        res += ivec::map(recur, post);
        ret res;
      }
    }
}



// substitute, in a position that's required to be an ident
fn transcribe_ident(cx: &ext_ctxt, b: &bindings, idx_path: @mutable [uint],
                    i: &ident, fld: ast_fold) -> ident {
    ret alt follow_for_trans(cx, b.find(i), idx_path) {
          some(match_ident(a_id)) { a_id.node }
          some(m) { match_error(cx, m, "an identifier") }
          none. { i }
        }
}


fn transcribe_path(cx: &ext_ctxt, b: &bindings, idx_path: @mutable [uint],
                   p: &path_, fld: ast_fold) -> path_ {
    // Don't substitute into qualified names.
    if ivec::len(p.types) > 0u || ivec::len(p.idents) != 1u { ret p; }
    ret alt follow_for_trans(cx, b.find(p.idents.(0)), idx_path) {
          some(match_ident(id)) {
            {global: false, idents: ~[id.node], types: ~[]}
          }
          some(match_path(a_pth)) { a_pth.node }
          some(m) { match_error(cx, m, "a path") }
          none. { p }
        }
}


fn transcribe_expr(cx: &ext_ctxt, b: &bindings, idx_path: @mutable [uint],
                   e: &ast::expr_, fld: ast_fold,
                   orig: fn(&ast::expr_, ast_fold) -> ast::expr_ ) ->
   ast::expr_ {
    ret alt e {
          expr_path(p) {
            // Don't substitute into qualified names.
            if ivec::len(p.node.types) > 0u || ivec::len(p.node.idents) != 1u
               {
                e
            }
            alt follow_for_trans(cx, b.find(p.node.idents.(0)), idx_path) {
              some(match_ident(id)) {
                expr_path(respan(id.span,
                                 {global: false,
                                  idents: ~[id.node],
                                  types: ~[]}))
              }
              some(match_path(a_pth)) { expr_path(a_pth) }
              some(match_expr(a_exp)) { a_exp.node }
              some(m) { match_error(cx, m, "an expression") }
              none. { orig(e, fld) }
            }
          }
          _ { orig(e, fld) }
        }
}

fn transcribe_type(cx: &ext_ctxt, b: &bindings, idx_path: @mutable [uint],
                   t: &ast::ty_, fld: ast_fold,
                   orig: fn(&ast::ty_, ast_fold) -> ast::ty_ ) -> ast::ty_ {
    ret alt t {
          ast::ty_path(pth, _) {
            alt path_to_ident(pth) {
              some(id) {
                alt follow_for_trans(cx, b.find(id), idx_path) {
                  some(match_ty(ty)) { ty.node }
                  some(m) { match_error(cx, m, "a type") }
                  none. { orig(t, fld) }
                }
              }
              none. { orig(t, fld) }
            }
          }
          _ { orig(t, fld) }
        }
}


/* for parsing reasons, syntax variables bound to blocks must be used like
`{v}` */

fn transcribe_block(cx: &ext_ctxt, b: &bindings, idx_path: @mutable [uint],
                    blk: &blk_, fld: ast_fold,
                    orig: fn(&blk_, ast_fold) -> blk_ ) -> blk_ {
    ret alt block_to_ident(blk) {
          some(id) {
            alt follow_for_trans(cx, b.find(id), idx_path) {
              some(match_block(new_blk)) { new_blk.node }

              // possibly allow promotion of ident/path/expr to blocks?
              some(m) {
                match_error(cx, m, "a block")
              }
              none. { orig(blk, fld) }
            }
          }
          none. { orig(blk, fld) }
        }
}


/* traverse the pattern, building instructions on how to bind the actual
argument. ps accumulates instructions on navigating the tree.*/
fn p_t_s_rec(cx: &ext_ctxt, m: &matchable, s: &selector, b: &binders) {

    //it might be possible to traverse only exprs, not matchables
    alt m {
      match_expr(e) {
        alt e.node {
          expr_path(p_pth) { p_t_s_r_path(cx, p_pth, s, b); }
          expr_vec(p_elts, _, _) {
            alt elts_to_ell(cx, p_elts) {
              {pre: pre, rep: some(repeat_me), post: post} {
                p_t_s_r_length(cx, ivec::len(pre) + ivec::len(post),
                               true, s, b);
                if(ivec::len(pre) > 0u) {
                    p_t_s_r_actual_vector(cx, pre, true, s, b);
                }
                p_t_s_r_ellipses(cx, repeat_me, ivec::len(pre), s, b);

                if(ivec::len(post) > 0u) {
                    cx.span_unimpl(e.span,
                                   "matching after `...` not yet supported");
                }
              }
              {pre: pre, rep: none., post: post} {
                if post != ~[] {
                    cx.bug("elts_to_ell provided an invalid result");
                }
                p_t_s_r_length(cx, ivec::len(pre), false, s, b);
                p_t_s_r_actual_vector(cx, pre, false, s, b);
              }
            }
          }

          /* TODO: handle embedded types and blocks, at least */
          expr_mac(mac) {
            p_t_s_r_mac(cx, mac, s, b);
          }
          _ {
            fn select(cx: &ext_ctxt, m: &matchable, pat: @expr) ->
               match_result {
                ret alt m {
                      match_expr(e) {
                        if e == pat { some(leaf(match_exact)) } else { none }
                      }
                      _ { cx.bug("broken traversal in p_t_s_r") }
                    }
            }
            b.literal_ast_matchers += ~[bind select(cx, _, e)];
          }
        }
      }
    }
}


/* make a match more precise */
fn specialize_match(m: &matchable) -> matchable {
    ret alt m {
          match_expr(e) {
            alt e.node {
              expr_path(pth) {
                alt path_to_ident(pth) {
                  some(id) { match_ident(respan(pth.span, id)) }
                  none. { match_path(pth) }
                }
              }
              _ { m }
            }
          }
          _ { m }
        }
}

/* pattern_to_selectors helper functions */
fn p_t_s_r_path(cx: &ext_ctxt, p: &path, s: &selector, b: &binders) {
    alt path_to_ident(p) {
      some(p_id) {
        fn select(cx: &ext_ctxt, m: &matchable) -> match_result {
            ret alt m {
                  match_expr(e) { some(leaf(specialize_match(m))) }
                  _ { cx.bug("broken traversal in p_t_s_r") }
                }
        }
        if b.real_binders.contains_key(p_id) {
            cx.span_fatal(p.span, "duplicate binding identifier");
        }
        b.real_binders.insert(p_id, compose_sels(s, bind select(cx, _)));
      }
      none. { }
    }
}

fn block_to_ident(blk: &blk_) -> option::t[ident] {
    if ivec::len(blk.stmts) != 0u { ret none; }
    ret alt blk.expr {
          some(expr) {
            alt expr.node { expr_path(pth) { path_to_ident(pth) } _ { none } }
          }
          none. { none }
        }
}

fn p_t_s_r_mac(cx: &ext_ctxt, mac: &ast::mac, s: &selector, b: &binders) {
    fn select_pt_1(cx: &ext_ctxt, m: &matchable,
                   fn_m: fn(&ast::mac) -> match_result ) -> match_result {
        ret alt m {
              match_expr(e) {
                alt e.node { expr_mac(mac) { fn_m(mac) } _ { none } }
              }
              _ { cx.bug("broken traversal in p_t_s_r") }
            }
    }
    fn no_des(cx: &ext_ctxt, sp: &span, syn: &str) -> ! {
        cx.span_fatal(sp, "destructuring " + syn + " is not yet supported");
    }
    alt mac.node {
      ast::mac_ellipsis. { cx.span_fatal(mac.span, "misused `...`"); }
      ast::mac_invoc(_, _, _) { no_des(cx, mac.span, "macro calls"); }
      ast::mac_embed_type(ty) {
        alt ty.node {
          ast::ty_path(pth, _) {
            alt path_to_ident(pth) {
              some(id) {
                /* look for an embedded type */
                fn select_pt_2(m: &ast::mac) -> match_result {
                    ret alt m.node {
                          ast::mac_embed_type(t) { some(leaf(match_ty(t))) }
                          _ { none }
                        }
                }
                let final_step = bind select_pt_1(cx, _, select_pt_2);
                b.real_binders.insert(id, compose_sels(s, final_step));
              }
              none. { no_des(cx, pth.span, "under `#<>`"); }
            }
          }
          _ { no_des(cx, ty.span, "under `#<>`"); }
        }
      }
      ast::mac_embed_block(blk) {
        alt block_to_ident(blk.node) {
          some(id) {
            fn select_pt_2(m: &ast::mac) -> match_result {
                ret alt m.node {
                      ast::mac_embed_block(blk) {
                        some(leaf(match_block(blk)))
                      }
                      _ { none }
                    }
            }
            let final_step = bind select_pt_1(cx, _, select_pt_2);
            b.real_binders.insert(id, compose_sels(s, final_step));
          }
          none. { no_des(cx, blk.span, "under `#{}`"); }
        }
      }
    }
}

fn p_t_s_r_ellipses(cx: &ext_ctxt, repeat_me: @expr, offset: uint,
                    s: &selector, b: &binders) {
    fn select(cx: &ext_ctxt, repeat_me: @expr, offset: uint, m: &matchable) ->
        match_result {
        ret alt m {
              match_expr(e) {
                alt e.node {
                  expr_vec(arg_elts, _, _) {
                    let elts = ~[];
                    let idx = offset;
                    while idx < ivec::len(arg_elts) {
                        elts += ~[leaf(match_expr(arg_elts.(idx)))];
                        idx += 1u;
                    }
                    // using repeat_me.span is a little wacky, but the
                    // error we want to report is one in the macro def
                    some(seq(@elts, repeat_me.span))
                  }
                  _ { none }
                }
              }
              _ { cx.bug("broken traversal in p_t_s_r") }
            }
    }
    p_t_s_rec(cx, match_expr(repeat_me),
              compose_sels(s, bind select(cx, repeat_me, offset, _)), b);
}


fn p_t_s_r_length(cx: &ext_ctxt, len: uint, at_least: bool, s: selector,
                  b: &binders) {
    fn len_select(cx: &ext_ctxt, m: &matchable, at_least: bool, len: uint)
        -> match_result {
        ret alt m {
              match_expr(e) {
                alt e.node {
                  expr_vec(arg_elts, _, _) {
                    let actual_len = ivec::len(arg_elts);
                    if (at_least && actual_len >= len) || actual_len == len {
                        some(leaf(match_exact))
                    } else { none }
                  }
                  _ { none }
                }
              }
              _ { none }
            }
    }
    b.literal_ast_matchers +=
        ~[compose_sels(s, bind len_select(cx, _, at_least, len))];
}

fn p_t_s_r_actual_vector(cx: &ext_ctxt, elts: [@expr], repeat_after: bool,
                         s: &selector, b: &binders) {
    let idx: uint = 0u;
    while idx < ivec::len(elts) {
        fn select(cx: &ext_ctxt, m: &matchable, idx: uint) -> match_result {
            ret alt m {
                  match_expr(e) {
                    alt e.node {
                      expr_vec(arg_elts, _, _) {
                        some(leaf(match_expr(arg_elts.(idx))))
                      }
                      _ { none }
                    }
                  }
                  _ { cx.bug("broken traversal in p_t_s_r") }
                }
        }
        p_t_s_rec(cx, match_expr(elts.(idx)),
                  compose_sels(s, bind select(cx, _, idx)), b);
        idx += 1u;
    }
}

fn add_new_extension(cx: &ext_ctxt, sp: span, arg: @expr,
                     body: option::t[str]) -> base::macro_def {
    let args: [@ast::expr] = alt arg.node {
      ast::expr_vec(elts, _, _) { elts }
      _ {
        cx.span_fatal(sp, "#macro requires arguments of the form `[...]`.")
      }
    };

    let macro_name: option::t[str] = none;
    let clauses: [@clause] = ~[];
    for arg: @expr in args {
        alt arg.node {
          expr_vec(elts, mut, seq_kind) {
            if ivec::len(elts) != 2u {
                cx.span_fatal((*arg).span,
                              "extension clause must consist of [" +
                                  "macro invocation, expansion body]");
            }


            alt elts.(0u).node {
              expr_mac(mac) {
                alt mac.node {
                  mac_invoc(pth, invoc_arg, body) {
                    alt path_to_ident(pth) {
                      some(id) {
                        alt macro_name {
                          none. { macro_name = some(id); }
                          some(other_id) {
                            if id != other_id {
                                cx.span_fatal(pth.span, "macro name must be "
                                              + "consistent");
                            }
                          }
                        }
                      }
                      none. {
                        cx.span_fatal(pth.span,
                                      "macro name must not be a path");
                      }
                    }
                    clauses +=
                        ~[@{params: pattern_to_selectors(cx, invoc_arg),
                            body: elts.(1u)}];
                    // FIXME: check duplicates (or just simplify
                    // the macro arg situation)
                  }
                }
              }
              _ {
                cx.span_fatal(elts.(0u).span,
                              "extension clause must" +
                                  " start with a macro invocation.");
              }
            }
          }
          _ {
            cx.span_fatal((*arg).span,
                          "extension must be [clause, " + " ...]");
          }
        }
    }

    let ext = bind generic_extension(_, _, _, _, clauses);

    ret {ident:
             alt macro_name {
               some(id) { id }
               none. {
                 cx.span_fatal(sp,
                               "macro definition must have " +
                                   "at least one clause")
               }
             },
         ext: normal(ext)};

    fn generic_extension(cx: &ext_ctxt, sp: span, arg: @expr,
                         body: option::t[str], clauses: [@clause]) -> @expr {
        for c: @clause in clauses {
            alt use_selectors_to_bind(c.params, arg) {
              some(bindings) {
                ret transcribe(cx, bindings, c.body)
              }
              none. { cont; }
            }
        }
        cx.span_fatal(sp, "no clauses match macro invocation");
    }
}



//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
