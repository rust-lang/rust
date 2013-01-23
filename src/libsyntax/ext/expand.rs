// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast::{crate, expr_, expr_mac, mac_invoc_tt};
use ast::{tt_delim, tt_tok, item_mac, stmt_, stmt_mac, stmt_expr, stmt_semi};
use ast;
use codemap::{span, ExpandedFrom};
use ext::base::*;
use fold::*;
use parse::{parser, parse_expr_from_source_str, new_parser_from_tts};

use core::option;
use core::vec;
use std::map::HashMap;

fn expand_expr(exts: HashMap<~str, SyntaxExtension>, cx: ext_ctxt,
               e: expr_, s: span, fld: ast_fold,
               orig: fn@(expr_, span, ast_fold) -> (expr_, span))
    -> (expr_, span)
{
    return match e {
      // expr_mac should really be expr_ext or something; it's the
      // entry-point for all syntax extensions.
          expr_mac(ref mac) => {

            match (*mac).node {

              // Token-tree macros, these will be the only case when we're
              // finished transitioning.
              mac_invoc_tt(pth, ref tts) => {
                assert (vec::len(pth.idents) == 1u);
                /* using idents and token::special_idents would make the
                the macro names be hygienic */
                let extname = cx.parse_sess().interner.get(pth.idents[0]);
                match exts.find(*extname) {
                  None => {
                    cx.span_fatal(pth.span,
                                  fmt!("macro undefined: '%s'", *extname))
                  }
                  Some(NormalTT(SyntaxExpanderTT{expander: exp,
                                                 span: exp_sp})) => {
                    cx.bt_push(ExpandedFrom({call_site: s,
                                callie: {name: *extname, span: exp_sp}}));

                    let expanded = match exp(cx, (*mac).span, (*tts)) {
                      MRExpr(e) => e,
                      MRAny(expr_maker,_,_) => expr_maker(),
                      _ => cx.span_fatal(
                          pth.span, fmt!("non-expr macro in expr pos: %s",
                                         *extname))
                    };

                    //keep going, outside-in
                    let fully_expanded = fld.fold_expr(expanded).node;
                    cx.bt_pop();

                    (fully_expanded, s)
                  }
                  _ => {
                    cx.span_fatal(pth.span,
                                  fmt!("'%s' is not a tt-style macro",
                                       *extname))
                  }

                }
              }
            }
          }
          _ => orig(e, s, fld)
        };
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
fn expand_mod_items(exts: HashMap<~str, SyntaxExtension>, cx: ext_ctxt,
                    module_: ast::_mod, fld: ast_fold,
                    orig: fn@(ast::_mod, ast_fold) -> ast::_mod)
    -> ast::_mod
{
    // Fold the contents first:
    let module_ = orig(module_, fld);

    // For each item, look through the attributes.  If any of them are
    // decorated with "item decorators", then use that function to transform
    // the item into a new set of items.
    let new_items = do vec::flat_map(module_.items) |item| {
        do vec::foldr(item.attrs, ~[*item]) |attr, items| {
            let mname = match attr.node.value.node {
              ast::meta_word(ref n) => (*n),
              ast::meta_name_value(ref n, _) => (*n),
              ast::meta_list(ref n, _) => (*n)
            };
            match exts.find(mname) {
              None | Some(NormalTT(_)) | Some(ItemTT(*)) => items,
              Some(ItemDecorator(dec_fn)) => {
                  cx.bt_push(ExpandedFrom({call_site: attr.span,
                                           callie: {name: copy mname,
                                                    span: None}}));
                  let r = dec_fn(cx, attr.span, attr.node.value, items);
                  cx.bt_pop();
                  r
              }
            }
        }
    };

    ast::_mod { items: new_items, ..module_ }
}


// When we enter a module, record it, for the sake of `module!`
fn expand_item(exts: HashMap<~str, SyntaxExtension>,
               cx: ext_ctxt, &&it: @ast::item, fld: ast_fold,
               orig: fn@(&&v: @ast::item, ast_fold) -> Option<@ast::item>)
    -> Option<@ast::item>
{
    let is_mod = match it.node {
      ast::item_mod(_) | ast::item_foreign_mod(_) => true,
      _ => false
    };
    let maybe_it = match it.node {
      ast::item_mac(*) => expand_item_mac(exts, cx, it, fld),
      _ => Some(it)
    };

    match maybe_it {
      Some(it) => {
        if is_mod { cx.mod_push(it.ident); }
        let ret_val = orig(it, fld);
        if is_mod { cx.mod_pop(); }
        return ret_val;
      }
      None => return None
    }
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
fn expand_item_mac(exts: HashMap<~str, SyntaxExtension>,
                   cx: ext_ctxt, &&it: @ast::item,
                   fld: ast_fold) -> Option<@ast::item> {

    let (pth, tts) = match it.node {
        item_mac(ast::spanned { node: mac_invoc_tt(pth, ref tts), _}) => {
            (pth, (*tts))
        }
        _ => cx.span_bug(it.span, ~"invalid item macro invocation")
    };

    let extname = cx.parse_sess().interner.get(pth.idents[0]);
    let expanded = match exts.find(*extname) {
        None => cx.span_fatal(pth.span,
                              fmt!("macro undefined: '%s!'", *extname)),

        Some(NormalTT(ref expand)) => {
            if it.ident != parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects no ident argument, \
                                    given '%s'", *extname,
                                   *cx.parse_sess().interner.get(it.ident)));
            }
            cx.bt_push(ExpandedFrom({call_site: it.span,
                                     callie: {name: *extname,
                                              span: (*expand).span}}));
            ((*expand).expander)(cx, it.span, tts)
        }
        Some(ItemTT(ref expand)) => {
            if it.ident == parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects an ident argument",
                                   *extname));
            }
            cx.bt_push(ExpandedFrom({call_site: it.span,
                                     callie: {name: *extname,
                                              span: (*expand).span}}));
            ((*expand).expander)(cx, it.span, it.ident, tts)
        }
        _ => cx.span_fatal(
            it.span, fmt!("%s! is not legal in item position", *extname))
    };

    let maybe_it = match expanded {
        MRItem(it) => fld.fold_item(it),
        MRExpr(_) => cx.span_fatal(pth.span,
                                    ~"expr macro in item position: "
                                    + *extname),
        MRAny(_, item_maker, _) =>
            option::chain(item_maker(), |i| {fld.fold_item(i)}),
        MRDef(ref mdef) => {
            exts.insert((*mdef).name, (*mdef).ext);
            None
        }
    };
    cx.bt_pop();
    return maybe_it;
}

fn expand_stmt(exts: HashMap<~str, SyntaxExtension>, cx: ext_ctxt,
               && s: stmt_, sp: span, fld: ast_fold,
               orig: fn@(&&s: stmt_, span, ast_fold) -> (stmt_, span))
    -> (stmt_, span)
{

    let (mac, pth, tts, semi) = match s {
        stmt_mac(ref mac, semi) => {
            match (*mac).node {
                mac_invoc_tt(pth, ref tts) => ((*mac), pth, (*tts), semi)
            }
        }
        _ => return orig(s, sp, fld)
    };

    assert(vec::len(pth.idents) == 1u);
    let extname = cx.parse_sess().interner.get(pth.idents[0]);
    let (fully_expanded, sp) = match exts.find(*extname) {
        None =>
            cx.span_fatal(pth.span, fmt!("macro undefined: '%s'", *extname)),

        Some(NormalTT(
            SyntaxExpanderTT{expander: exp, span: exp_sp})) => {
            cx.bt_push(ExpandedFrom(
                {call_site: sp, callie: {name: *extname, span: exp_sp}}));
            let expanded = match exp(cx, mac.span, tts) {
                MRExpr(e) =>
                    @ast::spanned { node: stmt_expr(e, cx.next_id()),
                                    span: e.span},
                MRAny(_,_,stmt_mkr) => stmt_mkr(),
                _ => cx.span_fatal(
                    pth.span,
                    fmt!("non-stmt macro in stmt pos: %s", *extname))
            };

            //keep going, outside-in
            let fully_expanded = fld.fold_stmt(expanded).node;
            cx.bt_pop();

            (fully_expanded, sp)
        }

        _ => {
            cx.span_fatal(pth.span,
                          fmt!("'%s' is not a tt-style macro", *extname))
        }
    };

    return (match fully_expanded {
        stmt_expr(e, stmt_id) if semi => stmt_semi(e, stmt_id),
        _ => { fully_expanded } /* might already have a semi */
    }, sp)

}


fn new_span(cx: ext_ctxt, sp: span) -> span {
    /* this discards information in the case of macro-defining macros */
    return span {lo: sp.lo, hi: sp.hi, expn_info: cx.backtrace()};
}

// FIXME (#2247): this is a terrible kludge to inject some macros into
// the default compilation environment. When the macro-definition system
// is substantially more mature, these should move from here, into a
// compiled part of libcore at very least.

fn core_macros() -> ~str {
    return
~"{
    macro_rules! ignore (($($x:tt)*) => (()))

    macro_rules! error ( ($( $arg:expr ),+) => (
        log(::core::error, fmt!( $($arg),+ )) ))
    macro_rules! warn ( ($( $arg:expr ),+) => (
        log(::core::warn, fmt!( $($arg),+ )) ))
    macro_rules! info ( ($( $arg:expr ),+) => (
        log(::core::info, fmt!( $($arg),+ )) ))
    macro_rules! debug ( ($( $arg:expr ),+) => (
        log(::core::debug, fmt!( $($arg),+ )) ))

    macro_rules! die(
        ($msg: expr) => (
            ::core::sys::begin_unwind($msg, file!().to_owned(), line!())
        );
        () => (
            die!(~\"explicit failure\")
        )
    )

    macro_rules! fail_unless(
        ($cond:expr) => {
            if !$cond {
                die!(~\"assertion failed: \" + stringify!($cond))
            }
        }
    )

    macro_rules! condition (

        { $c:ident: $in:ty -> $out:ty; } => {

            mod $c {
                fn key(_x: @::core::condition::Handler<$in,$out>) { }

                pub const cond : ::core::condition::Condition<$in,$out> =
                    ::core::condition::Condition {
                    name: stringify!($c),
                    key: key
                };
            }
        }
    )

}";
}

fn expand_crate(parse_sess: parse::parse_sess,
                cfg: ast::crate_cfg, c: @crate) -> @crate {
    let exts = syntax_expander_table();
    let afp = default_ast_fold();
    let cx: ext_ctxt = mk_ctxt(parse_sess, cfg);
    let f_pre = @AstFoldFns {
        fold_expr: |a,b,c| expand_expr(exts, cx, a, b, c, afp.fold_expr),
        fold_mod: |a,b| expand_mod_items(exts, cx, a, b, afp.fold_mod),
        fold_item: |a,b| expand_item(exts, cx, a, b, afp.fold_item),
        fold_stmt: |a,b,c| expand_stmt(exts, cx, a, b, c, afp.fold_stmt),
        new_span: |a| new_span(cx, a),
        .. *afp};
    let f = make_fold(f_pre);
    let cm = parse_expr_from_source_str(~"<core-macros>",
                                        @core_macros(),
                                        cfg,
                                        parse_sess);

    // This is run for its side-effects on the expander env,
    // as it registers all the core macros as expanders.
    f.fold_expr(cm);

    let res = @f.fold_crate(*c);
    return res;
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
