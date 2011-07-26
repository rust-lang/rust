use std;

import codemap::span;
import std::ivec;
import std::vec;
import std::option;
import vec::map;
import vec::len;
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

fn path_to_ident(&path pth) -> option::t[ident] {
    if (ivec::len(pth.node.idents) == 1u
        && ivec::len(pth.node.types) == 0u) {
        ret some(pth.node.idents.(0u));
    }
    ret none;
}

//an ivec of binders might be a little big.
type clause = rec((binders)[] params, @expr body);

/* logically, an arb_depth should contain only one kind of matchable */
tag arb_depth[T] {
    leaf(T);
    seq(vec[arb_depth[T]], span);
}


tag matchable {
    match_expr(@expr);
    match_path(path);
    match_ident(ast::spanned[ident]);
    match_ty(@ty);
    match_block(ast::blk);
    match_exact; /* don't bind anything, just verify the AST traversal */
}

/* for when given an incompatible bit of AST */
fn match_error(&ext_ctxt cx, &matchable m, &str expected) -> ! {
    alt(m) {
      case (match_expr(?x)) {
        cx.span_fatal(x.span, "this argument is an expr, expected "
                      + expected);
      }
      case (match_path(?x)) {
        cx.span_fatal(x.span, "this argument is a path, expected "
                      + expected);
      }
      case (match_ident(?x)) {
        cx.span_fatal(x.span, "this argument is an ident, expected "
                      + expected);
      }
      case (match_ty(?x)) {
        cx.span_fatal(x.span, "this argument is a type, expected "
                      + expected);
      }
      case (match_block(?x)) {
        cx.span_fatal(x.span, "this argument is a block, expected "
                      + expected);
      }
      case (match_exact) {
        cx.bug("what is a match_exact doing in a bindings?");
      }
    }
}

// We can't make all the matchables in a match_result the same type because
// idents can be paths, which can be exprs.

// If we want better match failure error messages (like in Fortifying Syntax),
// we'll want to return something indicating amount of progress and location
// of failure instead of `none`.
type match_result = option::t[arb_depth[matchable]];
type selector = fn(&matchable) -> match_result;

fn elts_to_ell(&ext_ctxt cx, &(@expr)[] elts) -> option::t[@expr] {
    let uint idx = 0u;
    for (@expr elt in elts) {
        alt (elt.node) {
          case (expr_mac(?m)) {
            alt (m.node) {
              case (ast::mac_ellipsis) {
                if (idx != 1u || ivec::len(elts) != 2u) {
                    cx.span_fatal(m.span,
                                  "Ellpisis may only appear"
                                  +" after exactly 1 item.");
                }
                ret some(elts.(0));
              }
            }
          }
          case (_) { }
        }
        idx += 1u;
    }
    ret none;
}

fn option_flatten_map[T,U](&fn(&T)->option::t[U] f, &vec[T] v)
    -> option::t[vec[U]] {
    auto res = vec::alloc[U](vec::len(v));
    for (T elem in v) {
        alt (f(elem)) {
          case (none) { ret none; }
          case (some(?fv)) { res += [fv]; }
        }
    }
    ret some(res);
}

fn a_d_map(&arb_depth[matchable] ad, &selector f)
    -> match_result {
    alt (ad) {
      case (leaf(?x)) { ret f(x); }
      case (seq(?ads,?span)) {
        alt (option_flatten_map(bind a_d_map(_, f), ads)) {
          case (none) { ret none; }
          case (some(?ts)) { ret some(seq(ts,span)); }
        }
      }
    }
}

fn compose_sels(selector s1, selector s2) -> selector {
    fn scomp(selector s1, selector s2, &matchable m) ->
        match_result {
        ret alt (s1(m)) {
          case (none) { none }
          case (some(?matches)) { a_d_map(matches, s2) }
        }
    }
    ret bind scomp(s1, s2, _);
}



type binders = rec(hashmap[ident,selector] real_binders,
                   mutable (selector)[] literal_ast_matchers);
type bindings = hashmap[ident, arb_depth[matchable]];

fn acumm_bindings(&ext_ctxt cx, &bindings b_dest, &bindings b_src) {
}

/* these three functions are the big moving parts */

/* create the selectors needed to bind and verify the pattern */

fn pattern_to_selectors(&ext_ctxt cx, @expr e) -> binders {
    let binders res = rec(real_binders=new_str_hash[selector](),
                          mutable literal_ast_matchers=~[]);
    //this oughta return binders instead, but macro args are a sequence of
    //expressions, rather than a single expression
    fn trivial_selector(&matchable m) -> match_result {
        ret some(leaf(m));
    }
    p_t_s_rec(cx, match_expr(e), trivial_selector, res);
    ret res;
}



/* use the selectors on the actual arguments to the macro to extract
bindings. Most of the work is done in p_t_s, which generates the
selectors. */

fn use_selectors_to_bind(&binders b, @expr e) -> option::t[bindings] {
    auto res = new_str_hash[arb_depth[matchable]]();
    let bool never_mind = false;
    for each(@rec(ident key, selector val) pair
             in b.real_binders.items()) {
        alt (pair.val(match_expr(e))) {
          case (none) { never_mind = true; }
          case (some(?mtc)) { res.insert(pair.key, mtc); }
        }
    }
    if (never_mind) { ret none; } //HACK: `ret` doesn't work in `for each`
    for (selector sel in b.literal_ast_matchers) {
        alt (sel(match_expr(e))) {
          case (none) { ret none; }
          case (_) { }
        }
    }
    ret some(res);
}

/* use the bindings on the body to generate the expanded code */

fn transcribe(&ext_ctxt cx, &bindings b, @expr body) -> @expr {
    let @mutable vec[uint] idx_path = @mutable [];
    auto afp = default_ast_fold();
    auto f_pre =
        rec(fold_ident = bind transcribe_ident(cx, b, idx_path, _, _),
            fold_path = bind transcribe_path(cx, b, idx_path, _, _),
            fold_expr = bind transcribe_expr(cx, b, idx_path, _, _,
                                             afp.fold_expr),
            fold_ty = bind transcribe_type(cx, b, idx_path, _, _,
                                           afp.fold_ty),
            fold_block = bind transcribe_block(cx, b, idx_path, _, _,
                                               afp.fold_block),
            map_exprs = bind transcribe_exprs(cx, b, idx_path, _, _)
            with *afp);
    auto f = make_fold(f_pre);
    auto result = f.fold_expr(body);
    dummy_out(f);  //temporary: kill circular reference
    ret result;
}



/* helper: descend into a matcher */
fn follow(&arb_depth[matchable] m, @mutable vec[uint] idx_path)
    -> arb_depth[matchable] {
    let arb_depth[matchable] res = m;
    for (uint idx in *idx_path) {
        alt(res) {
          case (leaf(_)) { ret res; /* end of the line */ }
          case (seq(?new_ms,_)) { res = new_ms.(idx); }
        }
    }
    ret res;
}

fn follow_for_trans(&ext_ctxt cx, &option::t[arb_depth[matchable]] mmaybe,
                    @mutable vec[uint] idx_path) -> option::t[matchable] {
    alt(mmaybe) {
      case (none) { ret none }
      case (some(?m)) {
        ret alt(follow(m, idx_path)) {
          case (seq(_,?sp)) {
            cx.span_fatal(sp, "syntax matched under ... but not "
                          + "used that way.")
          }
          case (leaf(?m)) {
            ret some(m)
          }
        }
      }
    }

}

/* helper for transcribe_exprs: what vars from `b` occur in `e`? */
iter free_vars(&bindings b, @expr e) -> ident {
    let hashmap[ident,()] idents = new_str_hash[()]();
    fn mark_ident(&ident i, ast_fold fld, &bindings b,
                  &hashmap[ident,()] idents) -> ident {
        if(b.contains_key(i)) { idents.insert(i,()); }
        ret i;
    }
    // using fold is a hack: we want visit, but it doesn't hit idents ) :
    // solve this with macros
    auto f_pre = rec(fold_ident=bind mark_ident(_, _, b, idents)
                     with *default_ast_fold());
    auto f = make_fold(f_pre);
    f.fold_expr(e); // ignore result
    dummy_out(f);
    for each(ident id in idents.keys()) { put id; }
}


/* handle sequences (anywhere in the AST) of exprs, either real or ...ed */
fn transcribe_exprs(&ext_ctxt cx, &bindings b, @mutable vec[uint] idx_path,
                    fn(&@expr)->@expr recur, (@expr)[] exprs) -> (@expr)[] {
    alt (elts_to_ell(cx, exprs)) {
      case (some(?repeat_me)) {
        let option::t[rec(uint rep_count, ident name)] repeat = none;
        /* we need to walk over all the free vars in lockstep, except for
        the leaves, which are just duplicated */
        for each (ident fv in free_vars(b, repeat_me)) {
            auto cur_pos = follow(b.get(fv), idx_path);
            alt (cur_pos) {
              case (leaf(_)) { }
              case (seq(?ms,_)) {
                alt (repeat) {
                  case (none) {
                    repeat = some
                        (rec(rep_count=vec::len(ms), name=fv));
                  }
                  case (some({rep_count: ?old_len,
                              name: ?old_name})) {
                    auto len = vec::len(ms);
                    if (old_len != len) {
                        cx.span_fatal
                            (repeat_me.span,
                             #fmt("'%s' occurs %u times, but ",
                                  fv, len)+
                             #fmt("'%s' occurs %u times",
                                  old_name, old_len));
                    }
                  }
                }
              }
            }
        }
        auto res = ~[];
        alt (repeat) {
          case (none) {
            cx.span_fatal(repeat_me.span,
                          "'...' surrounds an expression without any"
                          + " repeating syntax variables");
          }
          case (some({rep_count: ?rc, _})) {
            /* Whew, we now know how how many times to repeat */
            let uint idx = 0u;
            while (idx < rc) {
                vec::push(*idx_path, idx);
                res += ~[recur(repeat_me)]; // whew!
                vec::pop(*idx_path);
                idx += 1u;
            }
          }
        }
        ret res;
      }
      case (none) { ret ivec::map(recur, exprs); }
    }
}



// substitute, in a position that's required to be an ident
fn transcribe_ident(&ext_ctxt cx, &bindings b, @mutable vec[uint] idx_path,
                    &ident i, ast_fold fld) -> ident {
    ret alt (follow_for_trans(cx, b.find(i), idx_path)) {
      case (some(match_ident(?a_id))) { a_id.node }
      case (some(?m)) { match_error(cx, m, "an identifier") }
      case (none) { i }
    }
}


fn transcribe_path(&ext_ctxt cx, &bindings b, @mutable vec[uint] idx_path,
                   &path_ p, ast_fold fld) -> path_ {
    // Don't substitute into qualified names.
    if (ivec::len(p.types) > 0u || ivec::len(p.idents) != 1u) { ret p; }
    ret alt (follow_for_trans(cx, b.find(p.idents.(0)), idx_path)) {
      case (some(match_ident(?id))) {
        rec(global=false, idents=~[id.node], types=~[])
      }
      case (some(match_path(?a_pth))) { a_pth.node }
      case (some(?m)) { match_error(cx, m, "a path") }
      case (none) { p }
    }
}


fn transcribe_expr(&ext_ctxt cx, &bindings b, @mutable vec[uint] idx_path,
                   &ast::expr_ e, ast_fold fld,
                   fn(&ast::expr_, ast_fold) -> ast::expr_ orig)
    -> ast::expr_ {
    ret alt(e) {
      case (expr_path(?p)){
        // Don't substitute into qualified names.
        if (ivec::len(p.node.types) > 0u ||
            ivec::len(p.node.idents) != 1u) { e }
        alt (follow_for_trans(cx, b.find(p.node.idents.(0)), idx_path)) {
          case (some(match_ident(?id))) {
            expr_path(respan(id.span,
                             rec(global=false,
                                 idents=~[id.node],types=~[])))
          }
          case (some(match_path(?a_pth))) { expr_path(a_pth) }
          case (some(match_expr(?a_exp))) { a_exp.node }
          case (some(?m)) { match_error(cx, m, "an expression")}
          case (none) { orig(e,fld) }
        }
      }
      case (_) { orig(e,fld) }
    }
}

fn transcribe_type(&ext_ctxt cx, &bindings b, @mutable vec[uint] idx_path,
                   &ast::ty_ t, ast_fold fld,
                   fn(&ast::ty_, ast_fold) -> ast::ty_ orig) -> ast::ty_ {
    ret alt(t) {
      case (ast::ty_path(?pth,_)) {
        alt (path_to_ident(pth)) {
          case (some(?id)) {
            alt (follow_for_trans(cx, b.find(id), idx_path)) {
              case (some(match_ty(?ty))) { ty.node }
              case (some(?m)) { match_error(cx, m, "a type") }
              case (none) { orig(t, fld) }
            }
          }
          case (none) { orig(t, fld) }
        }
      }
      case (_) { orig(t, fld) }
    }
}


/* for parsing reasons, syntax variables bound to blocks must be used like
`{v}` */

fn transcribe_block(&ext_ctxt cx, &bindings b, @mutable vec[uint] idx_path,
                    &blk_ blk, ast_fold fld,
                    fn(&blk_, ast_fold) -> blk_ orig) -> blk_ {
    ret alt (block_to_ident(blk)) {
      case (some(?id)) {
        alt (follow_for_trans(cx, b.find(id), idx_path)) {
          case (some(match_block(?new_blk))) { new_blk.node }
          // possibly allow promotion of ident/path/expr to blocks?
          case (some(?m)) { match_error(cx, m, "a block")}
          case (none) { orig(blk, fld) }
        }
      }
      case (none) { orig(blk, fld) }
    }
}


/* traverse the pattern, building instructions on how to bind the actual
argument. ps accumulates instructions on navigating the tree.*/
fn p_t_s_rec(&ext_ctxt cx, &matchable m, &selector s, &binders b) {
    //it might be possible to traverse only exprs, not matchables
    alt (m) {
      case (match_expr(?e)) {
        alt (e.node) {
          case (expr_path(?p_pth)) {
            p_t_s_r_path(cx,p_pth, s, b);
          }
          case (expr_vec(?p_elts, _, _)) {
            alt (elts_to_ell(cx, p_elts)) {
              case (some(?repeat_me)) {
                p_t_s_r_ellipses(cx, repeat_me, s, b);
              }
              case (none) {
                p_t_s_r_actual_vector(cx, p_elts, s, b);
              }
            }
          }
          /* TODO: handle embedded types and blocks, at least */
          case (expr_mac(?mac)) {
            p_t_s_r_mac(cx, mac, s, b);
          }
          case (_) {
            fn select(&ext_ctxt cx, &matchable m, @expr pat)
                -> match_result {
                ret alt(m) {
                  case (match_expr(?e)) {
                    if (e==pat) { some(leaf(match_exact)) } else { none }
                  }
                  case (_) { cx.bug("broken traversal in p_t_s_r"); fail }
                }
            }
            b.literal_ast_matchers += ~[bind select(cx,_,e)];
          }
        }
      }
    }
}


/* make a match more precise */
fn specialize_match(&matchable m) -> matchable {
    ret alt (m) {
      case (match_expr(?e)) {
        alt (e.node) {
          case (expr_path(?pth)) {
            alt (path_to_ident(pth)) {
              case (some(?id)) { match_ident(respan(pth.span,id)) }
              case (none) { match_path(pth) }
            }
          }
          case (_) { m }
        }
      }
      case (_) { m }
    }
}

/* pattern_to_selectors helper functions */
fn p_t_s_r_path(&ext_ctxt cx, &path p, &selector s, &binders b) {
    alt (path_to_ident(p)) {
      case (some(?p_id)) {
        fn select(&ext_ctxt cx, &matchable m) -> match_result {
            ret alt (m) {
              case (match_expr(?e)) { some(leaf(specialize_match(m))) }
              case (_) { cx.bug("broken traversal in p_t_s_r"); fail }
            }
        }
        if (b.real_binders.contains_key(p_id)) {
            cx.span_fatal(p.span, "duplicate binding identifier");
        }
        b.real_binders.insert(p_id, compose_sels(s, bind select(cx,_)));
      }
      case (none) { }
    }
}

fn block_to_ident(&blk_ blk) -> option::t[ident] {
    if(ivec::len(blk.stmts) != 0u) { ret none; }
    ret alt (blk.expr) {
      case (some(?expr)) {
        alt (expr.node) {
          case (expr_path(?pth)) { path_to_ident(pth) }
          case (_) { none }
        }
      }
      case(none) { none }
    }
}

fn p_t_s_r_mac(&ext_ctxt cx, &ast::mac mac, &selector s, &binders b) {
    fn select_pt_1(&ext_ctxt cx, &matchable m, fn(&ast::mac) ->
                   match_result fn_m) -> match_result {
        ret alt(m) {
          case (match_expr(?e)) {
            alt(e.node) {
              case (expr_mac(?mac)) { fn_m(mac) }
              case (_) { none }
            }
          }
          case (_) { cx.bug("broken traversal in p_t_s_r"); fail }
        }
    }
    fn no_des(&ext_ctxt cx, &span sp, &str syn) -> ! {
        cx.span_fatal(sp, "destructuring "+syn+" is not yet supported");
    }
    alt (mac.node) {
      case (ast::mac_ellipsis) { cx.span_fatal(mac.span, "misused `...`"); }
      case (ast::mac_invoc(_,_, _)) { no_des(cx, mac.span, "macro calls"); }
      case (ast::mac_embed_type(?ty)) {
        alt (ty.node) {
          case ast::ty_path(?pth, _) {
            alt (path_to_ident(pth)) {
              case (some(?id)) {
                /* look for an embedded type */
                fn select_pt_2(&ast::mac m) -> match_result {
                    ret alt (m.node) {
                      case (ast::mac_embed_type(?t)) {
                        some(leaf(match_ty(t)))
                      }
                      case (_) { none }
                    }
                }
                b.real_binders.insert(id,
                                      bind select_pt_1(cx, _, select_pt_2));
              }
              case (none) { no_des(cx, pth.span, "under `#<>`"); }
            }
          }
          case (_) { no_des(cx, ty.span, "under `#<>`"); }
        }
      }
      case (ast::mac_embed_block(?blk)) {
        alt (block_to_ident(blk.node)) {
          case (some(?id)) {
            fn select_pt_2(&ast::mac m) -> match_result {
                ret alt (m.node) {
                  case (ast::mac_embed_block(?blk)) {
                    some(leaf(match_block(blk)))
                  }
                  case (_) { none }
                }
            }
            b.real_binders.insert(id, bind select_pt_1(cx, _, select_pt_2));
          }
          case (none) { no_des(cx, blk.span, "under `#{}`"); }
        }
      }
    }
}

/* TODO: move this to vec.rs */

fn ivec_to_vec[T](&(T)[] v) -> vec[T] {
    let vec[T] rs = vec::alloc[T](ivec::len(v));
    for (T ve in v) { rs += [ve]; }
    ret rs;
}

fn p_t_s_r_ellipses(&ext_ctxt cx, @expr repeat_me, &selector s, &binders b) {
    fn select(&ext_ctxt cx, @expr repeat_me, &matchable m) -> match_result {
        ret alt (m) {
          case (match_expr(?e)) {
            alt (e.node) {
              case (expr_vec(?arg_elts, _, _)) {
                auto elts = ivec::map(leaf, ivec::map(match_expr,
                                                      arg_elts));
                // using repeat_me.span is a little wacky, but the
                // error we want to report is one in the macro def
                some(seq(ivec_to_vec(elts), repeat_me.span))
              }
              case (_) { none }
            }
          }
          case (_) { cx.bug("broken traversal in p_t_s_r"); fail }
        }
    }
    p_t_s_rec(cx, match_expr(repeat_me),
              compose_sels(s, bind select(cx, repeat_me, _)), b);
}

fn p_t_s_r_actual_vector(&ext_ctxt cx, (@expr)[] elts, &selector s,
                         &binders b) {
    fn len_select(&ext_ctxt cx, &matchable m, uint len) -> match_result {
        ret alt (m) {
          case (match_expr(?e)) {
            alt (e.node) {
              case (expr_vec(?arg_elts, _, _)) {
                if (ivec::len(arg_elts) == len) { some(leaf(match_exact)) }
                else { none }
              }
              case (_) { none }
            }
          }
          case (_) { none }
        }
    }
    b.literal_ast_matchers +=
        ~[compose_sels(s, bind len_select(cx, _, ivec::len(elts)))];


    let uint idx = 0u;
    while (idx < ivec::len(elts)) {
        fn select(&ext_ctxt cx, &matchable m, uint idx) -> match_result {
            ret alt (m) {
              case (match_expr(?e)) {
                alt (e.node) {
                  case (expr_vec(?arg_elts, _, _)) {
                    some(leaf(match_expr(arg_elts.(idx))))
                  }
                  case (_) { none }
                }
              }
              case (_) { cx.bug("broken traversal in p_t_s_r"); fail}
            }
        }
        p_t_s_rec(cx, match_expr(elts.(idx)),
                  compose_sels(s, bind select(cx, _, idx)), b);
        idx += 1u;
    }
}

fn add_new_extension(&ext_ctxt cx, span sp, &(@expr)[] args,
                     option::t[str] body) -> base::macro_def {
    let option::t[str] macro_name = none;
    let (clause)[] clauses = ~[];
    for (@expr arg in args) {
        alt(arg.node) {
          case(expr_vec(?elts, ?mut, ?seq_kind)) {
            if (ivec::len(elts) != 2u) {
                cx.span_fatal((*arg).span,
                              "extension clause must consist of [" +
                              "macro invocation, expansion body]");
            }

            alt(elts.(0u).node) {
              case(expr_mac(?mac)) {
                alt (mac.node) {
                  case (mac_invoc(?pth, ?invoc_args, ?body)) {
                    alt (path_to_ident(pth)) {
                      case (some(?id)) { macro_name=some(id); }
                      case (none) {
                        cx.span_fatal(pth.span, "macro name "
                                      + "must not be a path");
                      }
                    }
                    auto bdrses = ~[];
                    for(@expr arg in invoc_args) {
                        bdrses +=
                            ~[pattern_to_selectors(cx, arg)];
                    }
                    clauses +=
                        ~[rec(params=bdrses, body=elts.(1u))];
                    // FIXME: check duplicates (or just simplify
                    // the macro arg situation)
                  }
                }
              }
              case(_) {
                cx.span_fatal(elts.(0u).span, "extension clause must"
                              + " start with a macro invocation.");
              }
            }
          }
          case(_) {
            cx.span_fatal((*arg).span, "extension must be [clause, "
                          + " ...]");
          }
        }
    }

    auto ext = bind generic_extension(_,_,_,_,clauses);

    ret rec(ident=alt (macro_name) {
      case (some(?id)) { id }
      case (none) {
        cx.span_fatal(sp, "macro definition must have "
                      + "at least one clause")
      }
    }, ext=normal(ext));


    fn generic_extension(&ext_ctxt cx, span sp, &(@expr)[] args,
                         option::t[str] body, (clause)[] clauses)
        -> @expr {


        for (clause c in clauses) {
            if (ivec::len(args) != ivec::len(c.params)) { cont; }
            let uint i = 0u;
            let bindings bdgs = new_str_hash[arb_depth[matchable]]();
            let bool abort = false;
            while (i < ivec::len(args)) {
                alt (use_selectors_to_bind(c.params.(i), args.(i))) {
                  case (some(?new_bindings)) {
                    /* ick; I wish macros just took one expr */
                    for each (@rec(ident key, arb_depth[matchable] val) it
                              in new_bindings.items()) {
                        bdgs.insert(it.key, it.val);
                    }
                  }
                  case (none) { abort = true; }
                }
                i += 1u;
            }
            if (abort) { cont; }
            ret transcribe(cx, bdgs, c.body);
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
