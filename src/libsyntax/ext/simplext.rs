use codemap::span;
use std::map::{hashmap, str_hash, uint_hash};
use dvec::DVec;

use base::*;

use fold::*;
use ast_util::respan;
use ast::{ident, path, ty, blk_, expr, expr_path,
             expr_vec, expr_mac, mac_invoc, node_id, expr_index};

export add_new_extension;

fn path_to_ident(pth: @path) -> Option<ident> {
    if vec::len(pth.idents) == 1u && vec::len(pth.types) == 0u {
        return Some(pth.idents[0u]);
    }
    return None;
}

//a vec of binders might be a little big.
type clause = {params: binders, body: @expr};

/* logically, an arb_depth should contain only one kind of matchable */
enum arb_depth<T> { leaf(T), seq(@~[arb_depth<T>], span), }


enum matchable {
    match_expr(@expr),
    match_path(@path),
    match_ident(ast::spanned<ident>),
    match_ty(@ty),
    match_block(ast::blk),
    match_exact, /* don't bind anything, just verify the AST traversal */
}

/* for when given an incompatible bit of AST */
fn match_error(cx: ext_ctxt, m: matchable, expected: ~str) -> ! {
    match m {
      match_expr(x) => cx.span_fatal(
          x.span, ~"this argument is an expr, expected " + expected),
      match_path(x) => cx.span_fatal(
          x.span, ~"this argument is a path, expected " + expected),
      match_ident(x) => cx.span_fatal(
          x.span, ~"this argument is an ident, expected " + expected),
      match_ty(x) => cx.span_fatal(
          x.span, ~"this argument is a type, expected " + expected),
      match_block(x) => cx.span_fatal(
          x.span, ~"this argument is a block, expected " + expected),
      match_exact => cx.bug(~"what is a match_exact doing in a bindings?")
    }
}

// We can't make all the matchables in a match_result the same type because
// idents can be paths, which can be exprs.

// If we want better match failure error messages (like in Fortifying Syntax),
// we'll want to return something indicating amount of progress and location
// of failure instead of `none`.
type match_result = Option<arb_depth<matchable>>;
type selector = fn@(matchable) -> match_result;

fn elts_to_ell(cx: ext_ctxt, elts: ~[@expr]) ->
   {pre: ~[@expr], rep: Option<@expr>, post: ~[@expr]} {
    let mut idx: uint = 0u;
    let mut res = None;
    for elts.each |elt| {
        match elt.node {
          expr_mac(m) => match m.node {
            ast::mac_ellipsis => {
                if res.is_some() {
                    cx.span_fatal(m.span, ~"only one ellipsis allowed");
                }
                res =
                    Some({pre: vec::slice(elts, 0u, idx - 1u),
                          rep: Some(elts[idx - 1u]),
                          post: vec::slice(elts, idx + 1u, vec::len(elts))});
            }
            _ => ()
          },
          _ => ()
        }
        idx += 1u;
    }
    return match res {
          Some(val) => val,
          None => {pre: elts, rep: None, post: ~[]}
    }
}

fn option_flatten_map<T: Copy, U: Copy>(f: fn@(T) -> Option<U>, v: ~[T]) ->
   Option<~[U]> {
    let mut res = ~[];
    for v.each |elem| {
        match f(elem) {
          None => return None,
          Some(fv) => vec::push(res, fv)
        }
    }
    return Some(res);
}

fn a_d_map(ad: arb_depth<matchable>, f: selector) -> match_result {
    match ad {
      leaf(x) => return f(x),
      seq(ads, span) => match option_flatten_map(|x| a_d_map(x, f), *ads) {
        None => return None,
        Some(ts) => return Some(seq(@ts, span))
      }
    }
}

fn compose_sels(s1: selector, s2: selector) -> selector {
    fn scomp(s1: selector, s2: selector, m: matchable) -> match_result {
        return match s1(m) {
              None => None,
              Some(matches) => a_d_map(matches, s2)
            }
    }
    return { |x| scomp(s1, s2, x) };
}



type binders =
    {real_binders: hashmap<ident, selector>,
     literal_ast_matchers: DVec<selector>};
type bindings = hashmap<ident, arb_depth<matchable>>;

fn acumm_bindings(_cx: ext_ctxt, _b_dest: bindings, _b_src: bindings) { }

/* these three functions are the big moving parts */

/* create the selectors needed to bind and verify the pattern */

fn pattern_to_selectors(cx: ext_ctxt, e: @expr) -> binders {
    let res: binders =
        {real_binders: uint_hash::<selector>(),
         literal_ast_matchers: DVec()};
    //this oughta return binders instead, but macro args are a sequence of
    //expressions, rather than a single expression
    fn trivial_selector(m: matchable) -> match_result {
        return Some(leaf(m));
    }
    p_t_s_rec(cx, match_expr(e), trivial_selector, res);
    return res;
}



/* use the selectors on the actual arguments to the macro to extract
bindings. Most of the work is done in p_t_s, which generates the
selectors. */

fn use_selectors_to_bind(b: binders, e: @expr) -> Option<bindings> {
    let res = uint_hash::<arb_depth<matchable>>();
    //need to do this first, to check vec lengths.
    for b.literal_ast_matchers.each |sel| {
        match sel(match_expr(e)) { None => return None, _ => () }
    }
    let mut never_mind: bool = false;
    for b.real_binders.each |key, val| {
        match val(match_expr(e)) {
          None => never_mind = true,
          Some(mtc) => { res.insert(key, mtc); }
        }
    };
    //HACK: `ret` doesn't work in `for each`
    if never_mind { return None; }
    return Some(res);
}

/* use the bindings on the body to generate the expanded code */

fn transcribe(cx: ext_ctxt, b: bindings, body: @expr) -> @expr {
    let idx_path: @mut ~[uint] = @mut ~[];
    fn new_id(_old: node_id, cx: ext_ctxt) -> node_id { return cx.next_id(); }
    fn new_span(cx: ext_ctxt, sp: span) -> span {
        /* this discards information in the case of macro-defining macros */
        return {lo: sp.lo, hi: sp.hi, expn_info: cx.backtrace()};
    }
    let afp = default_ast_fold();
    let f_pre =
        @{fold_ident: |x,y|transcribe_ident(cx, b, idx_path, x, y),
          fold_path: |x,y|transcribe_path(cx, b, idx_path, x, y),
          fold_expr: |x,y,z|
              transcribe_expr(cx, b, idx_path, x, y, z, afp.fold_expr)
          ,
          fold_ty: |x,y,z|
              transcribe_type(cx, b, idx_path,
                              x, y, z, afp.fold_ty)
          ,
          fold_block: |x,y,z|
              transcribe_block(cx, b, idx_path, x, y, z, afp.fold_block)
          ,
          map_exprs: |x,y|
              transcribe_exprs(cx, b, idx_path, x, y)
          ,
          new_id: |x|new_id(x, cx),
          .. *afp};
    let f = make_fold(f_pre);
    let result = f.fold_expr(body);
    return result;
}


/* helper: descend into a matcher */
pure fn follow(m: arb_depth<matchable>, idx_path: &[uint]) ->
   arb_depth<matchable> {
    let mut res: arb_depth<matchable> = m;
    for vec::each(idx_path) |idx| {
        res = match res {
          leaf(_) => return res,/* end of the line */
          seq(new_ms, _) => new_ms[idx]
        }
    }
    return res;
}

fn follow_for_trans(cx: ext_ctxt, mmaybe: Option<arb_depth<matchable>>,
                    idx_path: @mut ~[uint]) -> Option<matchable> {
    match mmaybe {
      None => return None,
      Some(m) => {
        return match follow(m, *idx_path) {
              seq(_, sp) => {
                cx.span_fatal(sp,
                              ~"syntax matched under ... but not " +
                                  ~"used that way.")
              }
              leaf(m) => return Some(m)
            }
      }
    }

}

/* helper for transcribe_exprs: what vars from `b` occur in `e`? */
fn free_vars(b: bindings, e: @expr, it: fn(ident)) {
    let idents: hashmap<ident, ()> = uint_hash::<()>();
    fn mark_ident(&&i: ident, _fld: ast_fold, b: bindings,
                  idents: hashmap<ident, ()>) -> ident {
        if b.contains_key(i) { idents.insert(i, ()); }
        return i;
    }
    // using fold is a hack: we want visit, but it doesn't hit idents ) :
    // solve this with macros
    let f_pre =
        @{fold_ident: |x,y|mark_ident(x, y, b, idents),
          .. *default_ast_fold()};
    let f = make_fold(f_pre);
    f.fold_expr(e); // ignore result
    for idents.each_key |x| { it(x); };
}

fn wrong_occurs(cx: ext_ctxt, l: ident, l_c: uint, r: ident, r_c: uint)
    -> ~str {
    fmt!("'%s' occurs %u times, but '%s' occurs %u times",
         *cx.parse_sess().interner.get(l), l_c,
         *cx.parse_sess().interner.get(r), r_c)
}

/* handle sequences (anywhere in the AST) of exprs, either real or ...ed */
fn transcribe_exprs(cx: ext_ctxt, b: bindings, idx_path: @mut ~[uint],
                    recur: fn@(&&@expr) -> @expr,
                    exprs: ~[@expr]) -> ~[@expr] {
    match elts_to_ell(cx, exprs) {
      {pre: pre, rep: repeat_me_maybe, post: post} => {
        let mut res = vec::map(pre, recur);
        match repeat_me_maybe {
          None => (),
          Some(repeat_me) => {
            let mut repeat: Option<{rep_count: uint, name: ident}> = None;
            /* we need to walk over all the free vars in lockstep, except for
            the leaves, which are just duplicated */
            do free_vars(b, repeat_me) |fv| {
                let fv_depth = b.get(fv);
                let cur_pos = follow(fv_depth, *idx_path);
                match cur_pos {
                  leaf(_) => (),
                  seq(ms, _) => {
                    match repeat {
                      None => {
                        repeat = Some({rep_count: vec::len(*ms), name: fv});
                      }
                      Some({rep_count: old_len, name: old_name}) => {
                        let len = vec::len(*ms);
                        if old_len != len {
                            let msg = wrong_occurs(cx, fv, len,
                                                   old_name, old_len);
                            cx.span_fatal(repeat_me.span, msg);
                        }
                      }
                    }
                  }
                }
            };
            match repeat {
              None => {
                cx.span_fatal(repeat_me.span,
                              ~"'...' surrounds an expression without any" +
                                  ~" repeating syntax variables");
              }
              Some({rep_count: rc, _}) => {
                /* Whew, we now know how how many times to repeat */
                let mut idx: uint = 0u;
                while idx < rc {
                    vec::push(*idx_path, idx);
                    vec::push(res, recur(repeat_me)); // whew!
                    vec::pop(*idx_path);
                    idx += 1u;
                }
              }
            }
          }
        }
        res = vec::append(res, vec::map(post, recur));
        return res;
      }
    }
}



// substitute, in a position that's required to be an ident
fn transcribe_ident(cx: ext_ctxt, b: bindings, idx_path: @mut ~[uint],
                    &&i: ident, _fld: ast_fold) -> ident {
    return match follow_for_trans(cx, b.find(i), idx_path) {
          Some(match_ident(a_id)) => a_id.node,
          Some(m) => match_error(cx, m, ~"an identifier"),
          None => i
        }
}


fn transcribe_path(cx: ext_ctxt, b: bindings, idx_path: @mut ~[uint],
                   p: path, _fld: ast_fold) -> path {
    // Don't substitute into qualified names.
    if vec::len(p.types) > 0u || vec::len(p.idents) != 1u { return p; }
    match follow_for_trans(cx, b.find(p.idents[0]), idx_path) {
      Some(match_ident(id)) => {
        {span: id.span, global: false, idents: ~[id.node],
         rp: None, types: ~[]}
      }
      Some(match_path(a_pth)) => *a_pth,
      Some(m) => match_error(cx, m, ~"a path"),
      None => p
    }
}


fn transcribe_expr(cx: ext_ctxt, b: bindings, idx_path: @mut ~[uint],
                   e: ast::expr_, s: span, fld: ast_fold,
                   orig: fn@(ast::expr_, span, ast_fold)->(ast::expr_, span))
    -> (ast::expr_, span)
{
    return match e {
          expr_path(p) => {
            // Don't substitute into qualified names.
            if vec::len(p.types) > 0u || vec::len(p.idents) != 1u {
                (e, s);
            }
            match follow_for_trans(cx, b.find(p.idents[0]), idx_path) {
              Some(match_ident(id)) => {
                (expr_path(@{span: id.span,
                             global: false,
                             idents: ~[id.node],
                             rp: None,
                             types: ~[]}), id.span)
              }
              Some(match_path(a_pth)) => (expr_path(a_pth), s),
              Some(match_expr(a_exp)) => (a_exp.node, a_exp.span),
              Some(m) => match_error(cx, m, ~"an expression"),
              None => orig(e, s, fld)
            }
          }
          _ => orig(e, s, fld)
        }
}

fn transcribe_type(cx: ext_ctxt, b: bindings, idx_path: @mut ~[uint],
                   t: ast::ty_, s: span, fld: ast_fold,
                   orig: fn@(ast::ty_, span, ast_fold) -> (ast::ty_, span))
    -> (ast::ty_, span)
{
    return match t {
          ast::ty_path(pth, _) => {
            match path_to_ident(pth) {
              Some(id) => {
                match follow_for_trans(cx, b.find(id), idx_path) {
                  Some(match_ty(ty)) => (ty.node, ty.span),
                  Some(m) => match_error(cx, m, ~"a type"),
                  None => orig(t, s, fld)
                }
              }
              None => orig(t, s, fld)
            }
          }
          _ => orig(t, s, fld)
        }
}


/* for parsing reasons, syntax variables bound to blocks must be used like
`{v}` */

fn transcribe_block(cx: ext_ctxt, b: bindings, idx_path: @mut ~[uint],
                    blk: blk_, s: span, fld: ast_fold,
                    orig: fn@(blk_, span, ast_fold) -> (blk_, span))
    -> (blk_, span)
{
    return match block_to_ident(blk) {
          Some(id) => {
            match follow_for_trans(cx, b.find(id), idx_path) {
              Some(match_block(new_blk)) => (new_blk.node, new_blk.span),

              // possibly allow promotion of ident/path/expr to blocks?
              Some(m) => match_error(cx, m, ~"a block"),
              None => orig(blk, s, fld)
            }
          }
          None => orig(blk, s, fld)
        }
}


/* traverse the pattern, building instructions on how to bind the actual
argument. ps accumulates instructions on navigating the tree.*/
fn p_t_s_rec(cx: ext_ctxt, m: matchable, s: selector, b: binders) {

    //it might be possible to traverse only exprs, not matchables
    match m {
      match_expr(e) => {
        match e.node {
          expr_path(p_pth) => p_t_s_r_path(cx, p_pth, s, b),
          expr_vec(p_elts, _) => {
            match elts_to_ell(cx, p_elts) {
              {pre: pre, rep: Some(repeat_me), post: post} => {
                p_t_s_r_length(cx, vec::len(pre) + vec::len(post), true, s,
                               b);
                if vec::len(pre) > 0u {
                    p_t_s_r_actual_vector(cx, pre, true, s, b);
                }
                p_t_s_r_ellipses(cx, repeat_me, vec::len(pre), s, b);

                if vec::len(post) > 0u {
                    cx.span_unimpl(e.span,
                                   ~"matching after `...` not yet supported");
                }
              }
              {pre: pre, rep: None, post: post} => {
                if post.len() > 0 {
                    cx.bug(~"elts_to_ell provided an invalid result");
                }
                p_t_s_r_length(cx, vec::len(pre), false, s, b);
                p_t_s_r_actual_vector(cx, pre, false, s, b);
              }
            }
          }
          /* FIXME (#2251): handle embedded types and blocks, at least */
          expr_mac(mac) => {
            p_t_s_r_mac(cx, mac, s, b);
          }
          _ => {
            fn select(cx: ext_ctxt, m: matchable, pat: @expr) ->
               match_result {
                return match m {
                      match_expr(e) => {
                        if box::ptr_eq(e, pat) {
                            // XXX: Is this right?
                            Some(leaf(match_exact))
                        } else {
                            None
                        }
                      }
                      _ => cx.bug(~"broken traversal in p_t_s_r")
                    }
            }
            b.literal_ast_matchers.push(|x| select(cx, x, e));
          }
        }
      }
      _ => cx.bug(~"undocumented invariant in p_t_s_rec")
    }
}


/* make a match more precise */
fn specialize_match(m: matchable) -> matchable {
    return match m {
          match_expr(e) => {
            match e.node {
              expr_path(pth) => {
                match path_to_ident(pth) {
                  Some(id) => match_ident(respan(pth.span, id)),
                  None => match_path(pth)
                }
              }
              _ => m
            }
          }
          _ => m
        }
}

/* pattern_to_selectors helper functions */
fn p_t_s_r_path(cx: ext_ctxt, p: @path, s: selector, b: binders) {
    match path_to_ident(p) {
      Some(p_id) => {
        fn select(cx: ext_ctxt, m: matchable) -> match_result {
            return match m {
                  match_expr(*) => Some(leaf(specialize_match(m))),
                  _ => cx.bug(~"broken traversal in p_t_s_r")
                }
        }
        if b.real_binders.contains_key(p_id) {
            cx.span_fatal(p.span, ~"duplicate binding identifier");
        }
        b.real_binders.insert(p_id, compose_sels(s, |x| select(cx, x)));
      }
      None => ()
    }
}

fn block_to_ident(blk: blk_) -> Option<ident> {
    if vec::len(blk.stmts) != 0u { return None; }
    return match blk.expr {
          Some(expr) => match expr.node {
            expr_path(pth) => path_to_ident(pth),
            _ => None
          },
          None => None
        }
}

fn p_t_s_r_mac(cx: ext_ctxt, mac: ast::mac, _s: selector, _b: binders) {
    fn select_pt_1(cx: ext_ctxt, m: matchable,
                   fn_m: fn(ast::mac) -> match_result) -> match_result {
        return match m {
              match_expr(e) => match e.node {
                expr_mac(mac) => fn_m(mac),
                _ => None
              },
              _ => cx.bug(~"broken traversal in p_t_s_r")
            }
    }
    fn no_des(cx: ext_ctxt, sp: span, syn: ~str) -> ! {
        cx.span_fatal(sp, ~"destructuring " + syn + ~" is not yet supported");
    }
    match mac.node {
      ast::mac_ellipsis => cx.span_fatal(mac.span, ~"misused `...`"),
      ast::mac_invoc(_, _, _) => no_des(cx, mac.span, ~"macro calls"),
      ast::mac_invoc_tt(_, _) => no_des(cx, mac.span, ~"macro calls"),
      ast::mac_aq(_,_) => no_des(cx, mac.span, ~"antiquotes"),
      ast::mac_var(_) => no_des(cx, mac.span, ~"antiquote variables")
    }
}

fn p_t_s_r_ellipses(cx: ext_ctxt, repeat_me: @expr, offset: uint, s: selector,
                    b: binders) {
    fn select(cx: ext_ctxt, repeat_me: @expr, offset: uint, m: matchable) ->
       match_result {
        return match m {
              match_expr(e) => {
                match e.node {
                  expr_vec(arg_elts, _) => {
                    let mut elts = ~[];
                    let mut idx = offset;
                    while idx < vec::len(arg_elts) {
                        vec::push(elts, leaf(match_expr(arg_elts[idx])));
                        idx += 1u;
                    }

                    // using repeat_me.span is a little wacky, but the
                    // error we want to report is one in the macro def
                    Some(seq(@elts, repeat_me.span))
                  }
                  _ => None
                }
              }
              _ => cx.bug(~"broken traversal in p_t_s_r")
            }
    }
    p_t_s_rec(cx, match_expr(repeat_me),
              compose_sels(s, |x| select(cx, repeat_me, offset, x)), b);
}


fn p_t_s_r_length(cx: ext_ctxt, len: uint, at_least: bool, s: selector,
                  b: binders) {
    fn len_select(_cx: ext_ctxt, m: matchable, at_least: bool, len: uint) ->
       match_result {
        return match m {
              match_expr(e) => {
                match e.node {
                  expr_vec(arg_elts, _) => {
                    let actual_len = vec::len(arg_elts);
                    if at_least && actual_len >= len || actual_len == len {
                        Some(leaf(match_exact))
                    } else { None }
                  }
                  _ => None
                }
              }
              _ => None
            }
    }
    b.literal_ast_matchers.push(
        compose_sels(s, |x| len_select(cx, x, at_least, len)));
}

fn p_t_s_r_actual_vector(cx: ext_ctxt, elts: ~[@expr], _repeat_after: bool,
                         s: selector, b: binders) {
    let mut idx: uint = 0u;
    while idx < vec::len(elts) {
        fn select(cx: ext_ctxt, m: matchable, idx: uint) -> match_result {
            return match m {
                  match_expr(e) => {
                    match e.node {
                      expr_vec(arg_elts, _) => {
                        Some(leaf(match_expr(arg_elts[idx])))
                      }
                      _ => None
                    }
                  }
                  _ => cx.bug(~"broken traversal in p_t_s_r")
                }
        }
        p_t_s_rec(cx, match_expr(elts[idx]),
                  compose_sels(s, |x, copy idx| select(cx, x, idx)), b);
        idx += 1u;
    }
}

fn add_new_extension(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> base::macro_def {
    let args = get_mac_args_no_max(cx, sp, arg, 0u, ~"macro");

    let mut macro_name: Option<~str> = None;
    let mut clauses: ~[@clause] = ~[];
    for args.each |arg| {
        match arg.node {
          expr_vec(elts, _) => {
            if vec::len(elts) != 2u {
                cx.span_fatal((*arg).span,
                              ~"extension clause must consist of ~[" +
                                  ~"macro invocation, expansion body]");
            }


            match elts[0u].node {
              expr_mac(mac) => {
                match mac.node {
                  mac_invoc(pth, invoc_arg, _) => {
                    match path_to_ident(pth) {
                      Some(id) => {
                        let id_str = cx.str_of(id);
                        match macro_name {
                          None => macro_name = Some(id_str),
                          Some(other_id) => if id_str != other_id {
                            cx.span_fatal(pth.span,
                                          ~"macro name must be " +
                                          ~"consistent");
                          }
                        }
                      },
                      None => cx.span_fatal(pth.span,
                                            ~"macro name must not be a path")
                    }
                    let arg = match invoc_arg {
                      Some(arg) => arg,
                      None => cx.span_fatal(mac.span,
                                           ~"macro must have arguments")
                    };
                    vec::push(clauses,
                              @{params: pattern_to_selectors(cx, arg),
                                body: elts[1u]});

                    // FIXME (#2251): check duplicates (or just simplify
                    // the macro arg situation)
                  }
                  _ => {
                      cx.span_bug(mac.span, ~"undocumented invariant in \
                         add_extension");
                  }
                }
              }
              _ => {
                cx.span_fatal(elts[0u].span,
                              ~"extension clause must" +
                                  ~" start with a macro invocation.");
              }
            }
          }
          _ => {
            cx.span_fatal((*arg).span,
                          ~"extension must be ~[clause, " + ~" ...]");
          }
        }
    }

    let ext = |a,b,c,d, move clauses| generic_extension(a,b,c,d,clauses);

    return {name:
             match macro_name {
               Some(id) => id,
               None => cx.span_fatal(sp, ~"macro definition must have " +
                                     ~"at least one clause")
             },
         ext: normal({expander: ext, span: Some(option::get(arg).span)})};

    fn generic_extension(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                         _body: ast::mac_body,
                         clauses: ~[@clause]) -> @expr {
        let arg = match arg {
          Some(arg) => arg,
          None => cx.span_fatal(sp, ~"macro must have arguments")
        };
        for clauses.each |c| {
            match use_selectors_to_bind(c.params, arg) {
              Some(bindings) => return transcribe(cx, bindings, c.body),
              None => loop
            }
        }
        cx.span_fatal(sp, ~"no clauses match macro invocation");
    }
}



//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
