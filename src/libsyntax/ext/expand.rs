// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, decl_local, empty_ctxt, expr_, expr_mac, Local, mac_invoc_tt};
use ast::{item_mac, Mrk, stmt_, stmt_decl, stmt_mac, stmt_expr, stmt_semi, SyntaxContext};
use ast::{SCTable, token_tree, illegal_ctxt};
use ast;
use ast_util::{new_rename, new_mark, mtwt_resolve};
use attr;
use attr::AttrMetaMethods;
use codemap;
use codemap::{span, spanned, ExpnInfo, NameAndSpan};
use ext::base::*;
use fold::*;
use parse;
use parse::{parse_item_from_source_str};
use parse::token::{fresh_mark, fresh_name, ident_to_str, intern};
use visit;
use visit::Visitor;

use std::vec;

pub fn expand_expr(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   e: &expr_,
                   s: span,
                   fld: @ast_fold,
                   orig: @fn(&expr_, span, @ast_fold) -> (expr_, span))
                -> (expr_, span) {
    match *e {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        expr_mac(ref mac) => {
            match (*mac).node {
                // it would almost certainly be cleaner to pass the whole
                // macro invocation in, rather than pulling it apart and
                // marking the tts and the ctxt separately. This also goes
                // for the other three macro invocation chunks of code
                // in this file.
                // Token-tree macros:
                mac_invoc_tt(ref pth, ref tts, ctxt) => {
                    if (pth.idents.len() > 1u) {
                        cx.span_fatal(
                            pth.span,
                            fmt!("expected macro name without module \
                                  separators"));
                    }
                    let extname = &pth.idents[0];
                    let extnamestr = ident_to_str(extname);
                    // leaving explicit deref here to highlight unbox op:
                    match (*extsbox).find(&extname.name) {
                        None => {
                            cx.span_fatal(
                                pth.span,
                                fmt!("macro undefined: '%s'", extnamestr))
                        }
                        Some(@SE(NormalTT(expandfun, exp_span))) => {
                            cx.bt_push(ExpnInfo {
                                call_site: s,
                                callee: NameAndSpan {
                                    name: extnamestr,
                                    span: exp_span,
                                },
                            });
                            let fm = fresh_mark();
                            // mark before:
                            let marked_before = mark_tts(*tts,fm);
                            let marked_ctxt = new_mark(fm, ctxt);
                            let expanded =
                                match expandfun(cx, mac.span, marked_before, marked_ctxt) {
                                    MRExpr(e) => e,
                                    MRAny(expr_maker,_,_) => expr_maker(),
                                    _ => {
                                        cx.span_fatal(
                                            pth.span,
                                            fmt!(
                                                "non-expr macro in expr pos: %s",
                                                extnamestr
                                            )
                                        )
                                    }
                                };
                            // mark after:
                            let marked_after = mark_expr(expanded,fm);

                            //keep going, outside-in
                            let fully_expanded =
                                fld.fold_expr(marked_after).node.clone();
                            cx.bt_pop();

                            (fully_expanded, s)
                        }
                        _ => {
                            cx.span_fatal(
                                pth.span,
                                fmt!("'%s' is not a tt-style macro", extnamestr)
                            )
                        }
                    }
                }
            }
        }
        _ => orig(e, s, fld)
    }
}

// This is a secondary mechanism for invoking syntax extensions on items:
// "decorator" attributes, such as #[auto_encode]. These are invoked by an
// attribute prefixing an item, and are interpreted by feeding the item
// through the named attribute _as a syntax extension_ and splicing in the
// resulting item vec into place in favour of the decorator. Note that
// these do _not_ work for macro extensions, just ItemDecorator ones.
//
// NB: there is some redundancy between this and expand_item, below, and
// they might benefit from some amount of semantic and language-UI merger.
pub fn expand_mod_items(extsbox: @mut SyntaxEnv,
                        cx: @ExtCtxt,
                        module_: &ast::_mod,
                        fld: @ast_fold,
                        orig: @fn(&ast::_mod, @ast_fold) -> ast::_mod)
                     -> ast::_mod {

    // Fold the contents first:
    let module_ = orig(module_, fld);

    // For each item, look through the attributes.  If any of them are
    // decorated with "item decorators", then use that function to transform
    // the item into a new set of items.
    let new_items = do vec::flat_map(module_.items) |item| {
        do item.attrs.rev_iter().fold(~[*item]) |items, attr| {
            let mname = attr.name();

            match (*extsbox).find(&intern(mname)) {
              Some(@SE(ItemDecorator(dec_fn))) => {
                  cx.bt_push(ExpnInfo {
                      call_site: attr.span,
                      callee: NameAndSpan {
                          name: mname,
                          span: None
                      }
                  });
                  let r = dec_fn(cx, attr.span, attr.node.value, items);
                  cx.bt_pop();
                  r
              },
              _ => items,
            }
        }
    };

    ast::_mod { items: new_items, ..module_ }
}

// eval $e with a new exts frame:
macro_rules! with_exts_frame (
    ($extsboxexpr:expr,$macros_escape:expr,$e:expr) =>
    ({let extsbox = $extsboxexpr;
      let oldexts = *extsbox;
      *extsbox = oldexts.push_frame();
      extsbox.insert(intern(special_block_name),
                     @BlockInfo(BlockInfo{macros_escape:$macros_escape,pending_renames:@mut ~[]}));
      let result = $e;
      *extsbox = oldexts;
      result
     })
)

static special_block_name : &'static str = " block";

// When we enter a module, record it, for the sake of `module!`
pub fn expand_item(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   it: @ast::item,
                   fld: @ast_fold,
                   orig: @fn(@ast::item, @ast_fold) -> Option<@ast::item>)
                -> Option<@ast::item> {
    // need to do expansion first... it might turn out to be a module.
    let maybe_it = match it.node {
      ast::item_mac(*) => expand_item_mac(extsbox, cx, it, fld),
      _ => Some(it)
    };
    match maybe_it {
      Some(it) => {
          match it.node {
              ast::item_mod(_) | ast::item_foreign_mod(_) => {
                  cx.mod_push(it.ident);
                  let macro_escape = contains_macro_escape(it.attrs);
                  let result = with_exts_frame!(extsbox,macro_escape,orig(it,fld));
                  cx.mod_pop();
                  result
              }
              _ => orig(it,fld)
          }
      }
      None => None
    }
}

// does this attribute list contain "macro_escape" ?
pub fn contains_macro_escape(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "macro_escape")
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
pub fn expand_item_mac(extsbox: @mut SyntaxEnv,
                       cx: @ExtCtxt, it: @ast::item,
                       fld: @ast_fold)
                    -> Option<@ast::item> {
    let (pth, tts, ctxt) = match it.node {
        item_mac(codemap::spanned { node: mac_invoc_tt(ref pth, ref tts, ctxt), _}) => {
            (pth, (*tts).clone(), ctxt)
        }
        _ => cx.span_bug(it.span, "invalid item macro invocation")
    };

    let extname = &pth.idents[0];
    let extnamestr = ident_to_str(extname);
    let fm = fresh_mark();
    let expanded = match (*extsbox).find(&extname.name) {
        None => cx.span_fatal(pth.span,
                              fmt!("macro undefined: '%s!'", extnamestr)),

        Some(@SE(NormalTT(expander, span))) => {
            if it.ident.name != parse::token::special_idents::invalid.name {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects no ident argument, \
                                    given '%s'", extnamestr,
                                   ident_to_str(&it.ident)));
            }
            cx.bt_push(ExpnInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: span
                }
            });
            // mark before expansion:
            let marked_before = mark_tts(tts,fm);
            let marked_ctxt = new_mark(fm,ctxt);
            expander(cx, it.span, marked_before, marked_ctxt)
        }
        Some(@SE(IdentTT(expander, span))) => {
            if it.ident.name == parse::token::special_idents::invalid.name {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects an ident argument",
                                   extnamestr));
            }
            cx.bt_push(ExpnInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: span
                }
            });
            let fm = fresh_mark();
            // mark before expansion:
            let marked_tts = mark_tts(tts,fm);
            let marked_ctxt = new_mark(fm,ctxt);
            expander(cx, it.span, it.ident, marked_tts, marked_ctxt)
        }
        _ => cx.span_fatal(
            it.span, fmt!("%s! is not legal in item position", extnamestr))
    };

    let maybe_it = match expanded {
        MRItem(it) => mark_item(it,fm).chain(|i| {fld.fold_item(i)}),
        MRExpr(_) => cx.span_fatal(pth.span,
                                   fmt!("expr macro in item position: %s", extnamestr)),
        MRAny(_, item_maker, _) => item_maker().chain(|i| {mark_item(i,fm)})
                                      .chain(|i| {fld.fold_item(i)}),
        MRDef(ref mdef) => {
            // yikes... no idea how to apply the mark to this. I'm afraid
            // we're going to have to wait-and-see on this one.
            insert_macro(*extsbox,intern(mdef.name), @SE((*mdef).ext));
            None
        }
    };
    cx.bt_pop();
    return maybe_it;
}


// insert a macro into the innermost frame that doesn't have the
// macro_escape tag.
fn insert_macro(exts: SyntaxEnv, name: ast::Name, transformer: @Transformer) {
    let is_non_escaping_block =
        |t : &@Transformer| -> bool{
        match t {
            &@BlockInfo(BlockInfo {macros_escape:false,_}) => true,
            &@BlockInfo(BlockInfo {_}) => false,
            _ => fail!(fmt!("special identifier %? was bound to a non-BlockInfo",
                            special_block_name))
        }
    };
    exts.insert_into_frame(name,transformer,intern(special_block_name),
                           is_non_escaping_block)
}

// expand a stmt
pub fn expand_stmt(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   s: &stmt_,
                   sp: span,
                   fld: @ast_fold,
                   orig: @fn(&stmt_, span, @ast_fold)
                             -> (Option<stmt_>, span))
                -> (Option<stmt_>, span) {
    // why the copying here and not in expand_expr?
    // looks like classic changed-in-only-one-place
    let (mac, pth, tts, semi, ctxt) = match *s {
        stmt_mac(ref mac, semi) => {
            match mac.node {
                mac_invoc_tt(ref pth, ref tts, ctxt) => {
                    ((*mac).clone(), pth, (*tts).clone(), semi, ctxt)
                }
            }
        }
        _ => return expand_non_macro_stmt(*extsbox,s,sp,fld,orig)
    };
    if (pth.idents.len() > 1u) {
        cx.span_fatal(
            pth.span,
            fmt!("expected macro name without module \
                  separators"));
    }
    let extname = &pth.idents[0];
    let extnamestr = ident_to_str(extname);
    let (fully_expanded, sp) = match (*extsbox).find(&extname.name) {
        None =>
            cx.span_fatal(pth.span, fmt!("macro undefined: '%s'", extnamestr)),

        Some(@SE(NormalTT(expandfun, exp_span))) => {
            cx.bt_push(ExpnInfo {
                call_site: sp,
                callee: NameAndSpan { name: extnamestr, span: exp_span }
            });
            let fm = fresh_mark();
            // mark before expansion:
            let marked_tts = mark_tts(tts,fm);
            let marked_ctxt = new_mark(fm,ctxt);
            let expanded = match expandfun(cx, mac.span, marked_tts, marked_ctxt) {
                MRExpr(e) =>
                    @codemap::spanned { node: stmt_expr(e, cx.next_id()),
                                    span: e.span},
                MRAny(_,_,stmt_mkr) => stmt_mkr(),
                _ => cx.span_fatal(
                    pth.span,
                    fmt!("non-stmt macro in stmt pos: %s", extnamestr))
            };
            let marked_after = mark_stmt(expanded,fm);

            //keep going, outside-in
            let fully_expanded = match fld.fold_stmt(marked_after) {
                Some(stmt) => {
                    let fully_expanded = &stmt.node;
                    cx.bt_pop();
                    (*fully_expanded).clone()
                }
                None => {
                    cx.span_fatal(pth.span,
                                  "macro didn't expand to a statement")
                }
            };

            (fully_expanded, sp)
        }

        _ => {
            cx.span_fatal(pth.span,
                          fmt!("'%s' is not a tt-style macro", extnamestr))
        }
    };

    (match fully_expanded {
        stmt_expr(e, stmt_id) if semi => Some(stmt_semi(e, stmt_id)),
        _ => { Some(fully_expanded) } /* might already have a semi */
    }, sp)

}


// expand a non-macro stmt. this is essentially the fallthrough for
// expand_stmt, above.
fn expand_non_macro_stmt (exts: SyntaxEnv,
                          s: &stmt_,
                          sp: span,
                          fld: @ast_fold,
                          orig: @fn(&stmt_, span, @ast_fold) -> (Option<stmt_>, span))
    -> (Option<stmt_>,span) {
    // is it a let?
    match *s {
        stmt_decl(@spanned{node: decl_local(ref local), span: stmt_span}, node_id) => {
            let block_info = get_block_info(exts);
            let pending_renames = block_info.pending_renames;
            // there's no need to have a lot of these, we could
            // just pass one of them around...
            let name_finder = new_name_finder();

            // take it apart:
            let @Local{is_mutbl:is_mutbl,
                       ty:_,
                       pat:pat,
                       init:init,
                       id:id,
                       span:span
                      } = *local;
            // types can't be copied automatically because of the owned ptr in ty_tup...
            let ty = local.ty.clone();
            // expand the pat (it might contain exprs... #:(o)>
            let expanded_pat = fld.fold_pat(pat);
            // find the pat_idents in the pattern:
            // oh dear heaven... this is going to include the enum names, as well....
            // ... but that should be okay, as long as the new names are gensyms
            // for the old ones.
            let idents = @mut ~[];
            // is this really the right way to call a visitor? it's so clunky!
            ((*name_finder).visit_pat) (expanded_pat,
                                        (idents,
                                         visit::mk_vt(name_finder)));
            // generate fresh names, push them to a new pending list
            let new_pending_renames = @mut ~[];
            for idents.iter().advance |ident| {
                let new_name = fresh_name(ident);
                new_pending_renames.push((*ident,new_name));
            }
            let mut rename_fld = renames_to_fold(new_pending_renames);
            // rewrite the pattern using the new names (the old ones
            // have already been applied):
            let rewritten_pat = rename_fld.fold_pat(expanded_pat);
            // add them to the existing pending renames:
            for new_pending_renames.iter().advance |pr| {pending_renames.push(*pr)}
            // also, don't forget to expand the init:
            let new_init_opt = init.map(|e| fld.fold_expr(*e));
            let rewritten_local =
                @Local{is_mutbl:is_mutbl,
                       ty:ty,
                       pat:rewritten_pat,
                       init:new_init_opt,
                       id:id,
                       span:span};
            (Some(stmt_decl(@spanned{node:decl_local(rewritten_local),
                                     span: stmt_span},node_id)),
             sp)
        },
        _ => {
            orig(s, sp, fld)
        }
    }
}


// return a visitor that extracts the pat_ident paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
pub fn new_name_finder() -> @Visitor<@mut ~[ast::ident]> {
    let default_visitor = visit::default_visitor();
    @Visitor{
        visit_pat : |p:@ast::pat,
                     (ident_accum, v): (@mut ~[ast::ident], visit::vt<@mut ~[ast::ident]>)| {
            match *p {
                // we found a pat_ident!
                ast::pat{id:_, node: ast::pat_ident(_,ref path,ref inner), span:_} => {
                    match path {
                        // a path of length one:
                        &ast::Path{global: false,idents: [id], span:_,rp:_,types:_} =>
                        ident_accum.push(id),
                        // I believe these must be enums...
                        _ => ()
                    }
                    // visit optional subpattern of pat_ident:
                    for inner.iter().advance |subpat: &@ast::pat| {
                        (v.visit_pat)(*subpat, (ident_accum, v))
                    }
                }
                // use the default traversal for non-pat_idents
                _ => visit::visit_pat(p,(ident_accum,v))
            }
        },
        .. *default_visitor
    }
}

// return a visitor that extracts the paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
pub fn new_path_finder() -> @Visitor<@mut ~[@ast::Path]> {
    let default_visitor = visit::default_visitor();
    @Visitor{
        visit_expr :
            |e:@ast::expr,
             (path_accum, v):(@mut ~[@ast::Path], visit::vt<@mut ~[@ast::Path]>)| {
            match *e {
                ast::expr{id:_,span:_,node:ast::expr_path(ref p)} => {
                    path_accum.push(@p.clone());
                    // not calling visit_path, should be fine.
                }
                _ => visit::visit_expr(e,(path_accum,v))
            }
        },
        .. *default_visitor
    }
}

// return a visitor that extracts the exprs
// from a given thingy and puts them in a mutable
// array. Maybe it's time for some abstraction, here....
pub fn new_expr_finder() -> @Visitor<@mut ~[@ast::expr]> {
    let default_visitor = visit::default_visitor();
    @Visitor{
        visit_expr :
            |e:@ast::expr,
             (path_accum, v):(@mut ~[@ast::expr], visit::vt<@mut ~[@ast::expr]>)| {
            path_accum.push(e);
            // call the built-in one as well to recur:
            visit::visit_expr(e,(path_accum,v))
        },
        .. *default_visitor
    }
}

// expand a block. pushes a new exts_frame, then calls expand_block_elts
pub fn expand_block(extsbox: @mut SyntaxEnv,
                    _cx: @ExtCtxt,
                    blk: &Block,
                    fld: @ast_fold,
                    orig: @fn(&Block, @ast_fold) -> Block)
                 -> Block {
    // see note below about treatment of exts table
    with_exts_frame!(extsbox,false,
                     expand_block_elts(*extsbox, blk, fld))
}

// expand the elements of a block.
pub fn expand_block_elts(exts: SyntaxEnv, b: &Block, fld: @ast_fold) -> Block {
    let block_info = get_block_info(exts);
    let pending_renames = block_info.pending_renames;
    let mut rename_fld = renames_to_fold(pending_renames);
    let new_view_items = b.view_items.map(|x| fld.fold_view_item(x));
    let mut new_stmts = ~[];
    for b.stmts.iter().advance |x| {
        match fld.fold_stmt(mustbesome(rename_fld.fold_stmt(*x))) {
            Some(s) => new_stmts.push(s),
            None => ()
        }
    }
    let new_expr = b.expr.map(|x| fld.fold_expr(rename_fld.fold_expr(*x)));
    Block{
        view_items: new_view_items,
        stmts: new_stmts,
        expr: new_expr,
        id: fld.new_id(b.id),
        rules: b.rules,
        span: b.span,
    }
}

// rename_fold should never return "None".
// (basically, just .get() with a better message...)
fn mustbesome<T>(val : Option<T>) -> T {
    match val {
        Some(v) => v,
        None => fail!("rename_fold returned None")
    }
}

// get the (innermost) BlockInfo from an exts stack
fn get_block_info(exts : SyntaxEnv) -> BlockInfo {
    match exts.find_in_topmost_frame(&intern(special_block_name)) {
        Some(@BlockInfo(bi)) => bi,
        _ => fail!(fmt!("special identifier %? was bound to a non-BlockInfo",
                       @" block"))
    }
}

pub fn new_span(cx: @ExtCtxt, sp: span) -> span {
    /* this discards information in the case of macro-defining macros */
    return span {lo: sp.lo, hi: sp.hi, expn_info: cx.backtrace()};
}

// FIXME (#2247): this is a moderately bad kludge to inject some macros into
// the default compilation environment in that it injects strings, rather than
// syntax elements.

pub fn std_macros() -> @str {
    return
@"mod __std_macros {
    #[macro_escape];
    #[doc(hidden)];

    macro_rules! ignore (($($x:tt)*) => (()))

    macro_rules! error (
        ($arg:expr) => (
            __log(1u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(1u32, fmt!( $($arg),+ ))
        )
    )

    macro_rules! warn (
        ($arg:expr) => (
            __log(2u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(2u32, fmt!( $($arg),+ ))
        )
    )

    macro_rules! info (
        ($arg:expr) => (
            __log(3u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(3u32, fmt!( $($arg),+ ))
        )
    )

    // conditionally define debug!, but keep it type checking even
    // in non-debug builds.
    macro_rules! __debug (
        ($arg:expr) => (
            __log(4u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(4u32, fmt!( $($arg),+ ))
        )
    )

    #[cfg(debug)]
    #[macro_escape]
    mod debug_macro {
        macro_rules! debug (($($arg:expr),*) => {
            __debug!($($arg),*)
        })
    }

    #[cfg(not(debug))]
    #[macro_escape]
    mod debug_macro {
        macro_rules! debug (($($arg:expr),*) => {
            if false { __debug!($($arg),*) }
        })
    }

    macro_rules! fail(
        () => (
            fail!(\"explicit failure\")
        );
        ($msg:expr) => (
            ::std::sys::FailWithCause::fail_with($msg, file!(), line!())
        );
        ($( $arg:expr ),+) => (
            ::std::sys::FailWithCause::fail_with(fmt!( $($arg),+ ), file!(), line!())
        )
    )

    macro_rules! assert(
        ($cond:expr) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with(
                    ~\"assertion failed: \" + stringify!($cond), file!(), line!())
            }
        };
        ($cond:expr, $msg:expr) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with($msg, file!(), line!())
            }
        };
        ($cond:expr, $( $arg:expr ),+) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with(fmt!( $($arg),+ ), file!(), line!())
            }
        }
    )

    macro_rules! assert_eq (
        ($given:expr , $expected:expr) => (
            {
                let given_val = $given;
                let expected_val = $expected;
                // check both directions of equality....
                if !((given_val == expected_val) && (expected_val == given_val)) {
                    fail!(\"assertion failed: `(left == right) && (right == \
                    left)` (left: `%?`, right: `%?`)\", given_val, expected_val);
                }
            }
        )
    )

    macro_rules! assert_approx_eq (
        ($given:expr , $expected:expr) => (
            {
                use std::cmp::ApproxEq;

                let given_val = $given;
                let expected_val = $expected;
                // check both directions of equality....
                if !(
                    given_val.approx_eq(&expected_val) &&
                    expected_val.approx_eq(&given_val)
                ) {
                    fail!(\"left: %? does not approximately equal right: %?\",
                          given_val, expected_val);
                }
            }
        );
        ($given:expr , $expected:expr , $epsilon:expr) => (
            {
                use std::cmp::ApproxEq;

                let given_val = $given;
                let expected_val = $expected;
                let epsilon_val = $epsilon;
                // check both directions of equality....
                if !(
                    given_val.approx_eq_eps(&expected_val, &epsilon_val) &&
                    expected_val.approx_eq_eps(&given_val, &epsilon_val)
                ) {
                    fail!(\"left: %? does not approximately equal right: %? with epsilon: %?\",
                          given_val, expected_val, epsilon_val);
                }
            }
        )
    )

    macro_rules! condition (

        { pub $c:ident: $in:ty -> $out:ty; } => {

            pub mod $c {
                #[allow(non_uppercase_statics)];
                static key: ::std::local_data::Key<
                    @::std::condition::Handler<$in, $out>> =
                    &::std::local_data::Key;

                pub static cond :
                    ::std::condition::Condition<$in,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        };

        { $c:ident: $in:ty -> $out:ty; } => {

            // FIXME (#6009): remove mod's `pub` below once variant above lands.
            pub mod $c {
                #[allow(non_uppercase_statics)];
                static key: ::std::local_data::Key<
                    @::std::condition::Handler<$in, $out>> =
                    &::std::local_data::Key;

                pub static cond :
                    ::std::condition::Condition<$in,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        }
    )

    //
    // A scheme-style conditional that helps to improve code clarity in some instances when
    // the `if`, `else if`, and `else` keywords obscure predicates undesirably.
    //
    // # Example
    //
    // ~~~
    // let clamped =
    //     if x > mx { mx }
    //     else if x < mn { mn }
    //     else { x };
    // ~~~
    //
    // Using `cond!`, the above could be written as:
    //
    // ~~~
    // let clamped = cond!(
    //     (x > mx) { mx }
    //     (x < mn) { mn }
    //     _        { x  }
    // );
    // ~~~
    //
    // The optional default case is denoted by `_`.
    //
    macro_rules! cond (
        ( $(($pred:expr) $body:block)+ _ $default:block ) => (
            $(if $pred $body else)+
            $default
        );
        // for if the default case was ommitted
        ( $(($pred:expr) $body:block)+ ) => (
            $(if $pred $body)else+
        );
    )

    macro_rules! printf (
        ($arg:expr) => (
            print(fmt!(\"%?\", $arg))
        );
        ($( $arg:expr ),+) => (
            print(fmt!($($arg),+))
        )
    )

    macro_rules! printfln (
        ($arg:expr) => (
            println(fmt!(\"%?\", $arg))
        );
        ($( $arg:expr ),+) => (
            println(fmt!($($arg),+))
        )
    )
}";
}

// add a bunch of macros as though they were placed at the head of the
// program (ick). This should run before cfg stripping.
pub fn inject_std_macros(parse_sess: @mut parse::ParseSess,
                         cfg: ast::CrateConfig, c: &Crate) -> @Crate {
    let sm = match parse_item_from_source_str(@"<std-macros>",
                                              std_macros(),
                                              cfg.clone(),
                                              ~[],
                                              parse_sess) {
        Some(item) => item,
        None => fail!("expected core macros to parse correctly")
    };

    let injecter = @AstFoldFns {
        fold_mod: |modd, _| {
            // just inject the std macros at the start of the first
            // module in the crate (i.e the crate file itself.)
            let items = vec::append(~[sm], modd.items);
            ast::_mod {
                items: items,
                // FIXME #2543: Bad copy.
                .. (*modd).clone()
            }
        },
        .. *default_ast_fold()
    };
    @make_fold(injecter).fold_crate(c)
}

pub fn expand_crate(parse_sess: @mut parse::ParseSess,
                    cfg: ast::CrateConfig, c: &Crate) -> @Crate {
    // adding *another* layer of indirection here so that the block
    // visitor can swap out one exts table for another for the duration
    // of the block.  The cleaner alternative would be to thread the
    // exts table through the fold, but that would require updating
    // every method/element of AstFoldFns in fold.rs.
    let extsbox = @mut syntax_expander_table();
    let afp = default_ast_fold();
    let cx = ExtCtxt::new(parse_sess, cfg.clone());
    let f_pre = @AstFoldFns {
        fold_expr: |expr,span,recur|
            expand_expr(extsbox, cx, expr, span, recur, afp.fold_expr),
        fold_mod: |modd,recur|
            expand_mod_items(extsbox, cx, modd, recur, afp.fold_mod),
        fold_item: |item,recur|
            expand_item(extsbox, cx, item, recur, afp.fold_item),
        fold_stmt: |stmt,span,recur|
            expand_stmt(extsbox, cx, stmt, span, recur, afp.fold_stmt),
        fold_block: |blk,recur|
            expand_block(extsbox, cx, blk, recur, afp.fold_block),
        new_span: |a| new_span(cx, a),
        .. *afp};
    let f = make_fold(f_pre);

    @f.fold_crate(c)
}

// a function in SyntaxContext -> SyntaxContext
pub trait CtxtFn{
    pub fn f(&self, ast::SyntaxContext) -> ast::SyntaxContext;
}

// a renamer adds a rename to the syntax context
pub struct Renamer {
    from : ast::ident,
    to : ast::Name
}

impl CtxtFn for Renamer {
    pub fn f(&self, ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        new_rename(self.from,self.to,ctxt)
    }
}

// a renamer that performs a whole bunch of renames
pub struct MultiRenamer {
    renames : @mut ~[(ast::ident,ast::Name)]
}

impl CtxtFn for MultiRenamer {
    pub fn f(&self, starting_ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        // the individual elements are memoized... it would
        // also be possible to memoize on the whole list at once.
        self.renames.iter().fold(starting_ctxt,|ctxt,&(from,to)| {
            new_rename(from,to,ctxt)
        })
    }
}

// a marker adds the given mark to the syntax context
pub struct Marker { mark : Mrk }

impl CtxtFn for Marker {
    pub fn f(&self, ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        new_mark(self.mark,ctxt)
    }
}

// a repainter just replaces the given context with the one it's closed over
pub struct Repainter { ctxt : SyntaxContext }

impl CtxtFn for Repainter {
    pub fn f(&self, ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        self.ctxt
    }
}

// given a function from ctxts to ctxts, produce
// an ast_fold that applies that function to all ctxts:
pub fn fun_to_ctxt_folder<T : 'static + CtxtFn>(cf: @T) -> @AstFoldFns {
    let afp = default_ast_fold();
    let fi : @fn(ast::ident, @ast_fold) -> ast::ident =
        |ast::ident{name, ctxt}, _| {
        ast::ident{name:name,ctxt:cf.f(ctxt)}
    };
    let fm : @fn(&ast::mac_, span, @ast_fold) -> (ast::mac_,span) =
        |m, sp, fld| {
        match *m {
            mac_invoc_tt(ref path, ref tts, ctxt) =>
            (mac_invoc_tt(fld.fold_path(path),
                         fold_tts(*tts,fld),
                         cf.f(ctxt)),
            sp)
        }

    };
    @AstFoldFns{
        fold_ident : fi,
        fold_mac : fm,
        .. *afp
    }
}



// given a mutable list of renames, return a tree-folder that applies those
// renames.
// FIXME #4536: currently pub to allow testing
pub fn renames_to_fold(renames : @mut ~[(ast::ident,ast::Name)]) -> @AstFoldFns {
    fun_to_ctxt_folder(@MultiRenamer{renames : renames})
}

// just a convenience:
pub fn new_mark_folder(m : Mrk) -> @AstFoldFns { fun_to_ctxt_folder(@Marker{mark:m}) }
pub fn new_rename_folder(from : ast::ident, to : ast::Name) -> @AstFoldFns {
    fun_to_ctxt_folder(@Renamer{from:from,to:to})
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts : &[token_tree], m : Mrk) -> ~[token_tree] {
    fold_tts(tts,new_mark_folder(m) as @ast_fold)
}

// apply a given mark to the given expr. Used following the expansion of a macro.
fn mark_expr(expr : @ast::expr, m : Mrk) -> @ast::expr {
    new_mark_folder(m).fold_expr(expr)
}

// apply a given mark to the given stmt. Used following the expansion of a macro.
fn mark_stmt(expr : &ast::stmt, m : Mrk) -> @ast::stmt {
    new_mark_folder(m).fold_stmt(expr).get()
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_item(expr : @ast::item, m : Mrk) -> Option<@ast::item> {
    new_mark_folder(m).fold_item(expr)
}

// replace all contexts in a given expr with the given mark. Used
// for capturing macros
pub fn replace_ctxts(expr : @ast::expr, ctxt : SyntaxContext) -> @ast::expr {
    fun_to_ctxt_folder(@Repainter{ctxt:ctxt}).fold_expr(expr)
}

#[cfg(test)]
mod test {
    use super::*;
    use ast;
    use ast::{Attribute_, AttrOuter, MetaWord, empty_ctxt};
    use ast_util::{get_sctable, mtwt_resolve, new_ident, new_rename};
    use codemap;
    use codemap::spanned;
    use parse;
    use parse::token::{gensym, intern, get_ident_interner, ident_to_str};
    use print::pprust;
    use std;
    use std::vec;
    use util::parser_testing::{string_to_crate, string_to_crate_and_sess, string_to_item}
    use util::parser_testing::{string_to_pat, strs_to_idents};
    use visit::{mk_vt};
    use visit;

    // make sure that fail! is present
    #[test] fn fail_exists_test () {
        let src = @"fn main() { fail!(\"something appropriately gloomy\");}";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
        let crate_ast = inject_std_macros(sess, ~[], crate_ast);
        // don't bother with striping, doesn't affect fail!.
        expand_crate(sess,~[],crate_ast);
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    // make sure that macros can leave scope
    #[should_fail]
    #[test] fn macros_cant_escape_fns_test () {
        let src = @"fn bogus() {macro_rules! z (() => (3+4))}\
                    fn inty() -> int { z!() }";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
        // should fail:
        expand_crate(sess,~[],crate_ast);
    }

    // make sure that macros can leave scope for modules
    #[should_fail]
    #[test] fn macros_cant_escape_mods_test () {
        let src = @"mod foo {macro_rules! z (() => (3+4))}\
                    fn inty() -> int { z!() }";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
        // should fail:
        expand_crate(sess,~[],crate_ast);
    }

    // macro_escape modules shouldn't cause macros to leave scope
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = @"#[macro_escape] mod foo {macro_rules! z (() => (3+4))}\
                    fn inty() -> int { z!() }";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[], sess);
        // should fail:
        expand_crate(sess,~[],crate_ast);
    }

    #[test] fn std_macros_must_parse () {
        let src = super::std_macros();
        let sess = parse::new_parse_sess(None);
        let cfg = ~[];
        let item_ast = parse::parse_item_from_source_str(
            @"<test>",
            src,
            cfg,~[],sess);
        match item_ast {
            Some(_) => (), // success
            None => fail!("expected this to parse")
        }
    }

    #[test] fn test_contains_flatten (){
        let attr1 = make_dummy_attr (@"foo");
        let attr2 = make_dummy_attr (@"bar");
        let escape_attr = make_dummy_attr (@"macro_escape");
        let attrs1 = ~[attr1, escape_attr, attr2];
        assert_eq!(contains_macro_escape (attrs1),true);
        let attrs2 = ~[attr1,attr2];
        assert_eq!(contains_macro_escape (attrs2),false);
    }

    // make a MetaWord outer attribute with the given name
    fn make_dummy_attr(s: @str) -> ast::Attribute {
        spanned {
            span:codemap::dummy_sp(),
            node: Attribute_ {
                style: AttrOuter,
                value: @spanned {
                    node: MetaWord(s),
                    span: codemap::dummy_sp(),
                },
                is_sugared_doc: false,
            }
        }
    }

    #[test]
    fn renaming () {
        let item_ast = string_to_crate(@"fn f() -> int { a }");
        let a_name = intern("a");
        let a2_name = gensym("a2");
        let renamer = new_rename_folder(ast::ident{name:a_name,ctxt:empty_ctxt},
                                        a2_name);
        let renamed_ast = renamer.fold_crate(item_ast);
        let pv = new_path_finder();
        let varrefs = @mut ~[];
        visit::visit_crate(&renamed_ast, (varrefs, mk_vt(pv)));
        match varrefs {
            @[@Path{idents:[id],_}] => assert_eq!(mtwt_resolve(id),a2_name),
            _ => assert_eq!(0,1)
        }

        // try a double-rename, with pending_renames.
        let a3_name = gensym("a3");
        // a context that renames from ("a",empty) to "a2" :
        let ctxt2 = new_rename(new_ident(a_name),a2_name,empty_ctxt);
        let pending_renames = @mut ~[(new_ident(a_name),a2_name),
                                     (ast::ident{name:a_name,ctxt:ctxt2},a3_name)];
        let double_renamed = renames_to_fold(pending_renames).fold_crate(item_ast);
        let varrefs = @mut ~[];
        visit::visit_crate(&double_renamed, (varrefs, mk_vt(pv)));
        match varrefs {
            @[@Path{idents:[id],_}] => assert_eq!(mtwt_resolve(id),a3_name),
            _ => assert_eq!(0,1)
        }
    }

    fn fake_print_crate(crate: @ast::Crate) {
        let s = pprust::rust_printer(std::io::stderr(),get_ident_interner());
        pprust::print_crate_(s, crate);
    }

    fn expand_crate_str(crate_str: @str) -> @ast::Crate {
        let (crate_ast,ps) = string_to_crate_and_sess(crate_str);
        // the cfg argument actually does matter, here...
        expand_crate(ps,~[],crate_ast)
    }

    //fn expand_and_resolve(crate_str: @str) -> ast::crate {
        //let expanded_ast = expand_crate_str(crate_str);
        // std::io::println(fmt!("expanded: %?\n",expanded_ast));
        //mtwt_resolve_crate(expanded_ast)
    //}
    //fn expand_and_resolve_and_pretty_print (crate_str : @str) -> ~str {
        //let resolved_ast = expand_and_resolve(crate_str);
        //pprust::to_str(&resolved_ast,fake_print_crate,get_ident_interner())
    //}

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(@"macro_rules! m((a)=>(13)) fn main(){m!(a);}");
    }

    // renaming tests expand a crate and then check that the bindings match
    // the right varrefs. The specification of the test case includes the
    // text of the crate, and also an array of arrays.  Each element in the
    // outer array corresponds to a binding in the traversal of the AST
    // induced by visit.  Each of these arrays contains a list of indexes,
    // interpreted as the varrefs in the varref traversal that this binding
    // should match.  So, for instance, in a program with two bindings and
    // three varrefs, the array ~[~[1,2],~[0]] would indicate that the first
    // binding should match the second two varrefs, and the second binding
    // should match the first varref.
    //
    // The comparisons are done post-mtwt-resolve, so we're comparing renamed
    // names; differences in marks don't matter any more.
    type renaming_test = (&'static str, ~[~[uint]]);

    #[test]
    fn automatic_renaming () {
        // need some other way to test these...
        let tests : ~[renaming_test] =
            ~[// b & c should get new names throughout, in the expr too:
                ("fn a() -> int { let b = 13; let c = b; b+c }",
                 ~[~[0,1],~[2]]),
                // both x's should be renamed (how is this causing a bug?)
                ("fn main () {let x : int = 13;x;}",
                 ~[~[0]]),
                // the use of b after the + should be renamed, the other one not:
                ("macro_rules! f (($x:ident) => (b + $x)) fn a() -> int { let b = 13; f!(b)}",
                 ~[~[1]]),
                // the b before the plus should not be renamed (requires marks)
                ("macro_rules! f (($x:ident) => ({let b=9; ($x + b)})) fn a() -> int { f!(b)}",
                 ~[~[1]]),
                // the marks going in and out of letty should cancel, allowing that $x to
                // capture the one following the semicolon.
                // this was an awesome test case, and caught a *lot* of bugs.
                ("macro_rules! letty(($x:ident) => (let $x = 15;))
                  macro_rules! user(($x:ident) => ({letty!($x); $x}))
                  fn main() -> int {user!(z)}",
                 ~[~[0]])
                // FIXME #6994: the next string exposes the bug referred to in issue 6994, so I'm
                // commenting it out.
                // the z flows into and out of two macros (g & f) along one path, and one
                // (just g) along the other, so the result of the whole thing should
                // be "let z_123 = 3; z_123"
                //"macro_rules! g (($x:ident) =>
                //   ({macro_rules! f(($y:ident)=>({let $y=3;$x}));f!($x)}))
                //   fn a(){g!(z)}"
                // create a really evil test case where a $x appears inside a binding of $x
                // but *shouldnt* bind because it was inserted by a different macro....
            ];
        for tests.iter().advance |s| {
            run_renaming_test(s);
        }
    }


    fn run_renaming_test(t : &renaming_test) {
        let nv = new_name_finder();
        let pv = new_path_finder();
        let (teststr, bound_connections) = match *t {
            (ref str,ref conns) => (str.to_managed(), conns.clone())
        };
        let cr = expand_crate_str(teststr.to_managed());
        let bindings = @mut ~[];
        visit::visit_crate(cr, (bindings, mk_vt(nv)));
        let varrefs = @mut ~[];
        visit::visit_crate(cr, (varrefs, mk_vt(pv)));
        // must be one check clause for each binding:
        assert_eq!(bindings.len(),bound_connections.len());
        for bound_connections.iter().enumerate().advance |(binding_idx,shouldmatch)| {
            let binding_name = mtwt_resolve(bindings[binding_idx]);
            // shouldmatch can't name varrefs that don't exist:
            assert!((shouldmatch.len() == 0) ||
                    (varrefs.len() > *shouldmatch.iter().max().get()));
            for varrefs.iter().enumerate().advance |(idx,varref)| {
                if shouldmatch.contains(&idx) {
                    // it should be a path of length 1, and it should
                    // be free-identifier=? to the given binding
                    assert_eq!(varref.idents.len(),1);
                    let varref_name = mtwt_resolve(varref.idents[0]);
                    if (!(varref_name==binding_name)){
                        std::io::println("uh oh, should match but doesn't:");
                        std::io::println(fmt!("varref: %?",varref));
                        std::io::println(fmt!("binding: %?", bindings[binding_idx]));
                        let table = get_sctable();
                        std::io::println("SC table:");
                        for table.table.iter().enumerate().advance |(idx,val)| {
                            std::io::println(fmt!("%4u : %?",idx,val));
                        }
                    }
                    assert_eq!(varref_name,binding_name);
                } else {
                    let fail = (varref.idents.len() == 1)
                        && (mtwt_resolve(varref.idents[0]) == binding_name);
                    // temp debugging:
                    if (fail) {
                        std::io::println("uh oh, matches but shouldn't:");
                        std::io::println(fmt!("varref: %?",varref));
                        std::io::println(fmt!("binding: %?", bindings[binding_idx]));
                        std::io::println(fmt!("sc_table: %?",get_sctable()));
                    }
                    assert!(!fail);
                }
            }
        }
    }

    #[test] fn quote_expr_test() {
        quote_ext_cx_test(@"fn main(){let ext_cx = 13; quote_expr!(dontcare);}");
    }
    #[test] fn quote_item_test() {
        quote_ext_cx_test(@"fn main(){let ext_cx = 13; quote_item!(dontcare);}");
    }
    #[test] fn quote_pat_test() {
        quote_ext_cx_test(@"fn main(){let ext_cx = 13; quote_pat!(dontcare);}");
    }
    #[test] fn quote_ty_test() {
        quote_ext_cx_test(@"fn main(){let ext_cx = 13; quote_ty!(dontcare);}");
    }
    #[test] fn quote_tokens_test() {
        quote_ext_cx_test(@"fn main(){let ext_cx = 13; quote_tokens!(dontcare);}");
    }

    fn quote_ext_cx_test(crate_str : @str) {
        let crate = expand_crate_str(crate_str);
        let nv = new_name_finder();
        let pv = new_path_finder();
        // find the ext_cx binding
        let bindings = @mut ~[];
        visit::visit_crate(crate, (bindings, mk_vt(nv)));
        let cxbinds : ~[&ast::ident] =
            bindings.iter().filter(|b|{@"ext_cx" == (ident_to_str(*b))}).collect();
        let cxbind = match cxbinds {
            [b] => b,
            _ => fail!("expected just one binding for ext_cx")
        };
        let resolved_binding = mtwt_resolve(*cxbind);
        // find all the ext_cx varrefs:
        let varrefs = @mut ~[];
        visit::visit_crate(crate, (varrefs, mk_vt(pv)));
        // the ext_cx binding should bind all of the ext_cx varrefs:
        for varrefs.iter().filter(|p|{ p.idents.len() == 1
                                          && (@"ext_cx" == (ident_to_str(&p.idents[0])))
                                     }).enumerate().advance |(idx,v)| {
            if (mtwt_resolve(v.idents[0]) != resolved_binding) {
                std::io::println("uh oh, ext_cx binding didn't match ext_cx varref:");
                std::io::println(fmt!("this is varref # %?",idx));
                std::io::println(fmt!("binding: %?",cxbind));
                std::io::println(fmt!("resolves to: %?",resolved_binding));
                std::io::println(fmt!("varref: %?",v.idents[0]));
                std::io::println(fmt!("resolves to: %?",mtwt_resolve(v.idents[0])));
                let table = get_sctable();
                std::io::println("SC table:");
                for table.table.iter().enumerate().advance |(idx,val)| {
                    std::io::println(fmt!("%4u : %?",idx,val));
                }
            }
            assert_eq!(mtwt_resolve(v.idents[0]),resolved_binding);
        };
    }

    #[test]
    fn pat_idents(){
        let pat = string_to_pat(@"(a,Foo{x:c @ (b,9),y:Bar(4,d)})");
        let pat_idents = new_name_finder();
        let idents = @mut ~[];
        ((*pat_idents).visit_pat)(pat, (idents, mk_vt(pat_idents)));
        assert_eq!(idents,@mut strs_to_idents(~["a","c","b","d"]));
    }

}
