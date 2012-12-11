// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::map::HashMap;
use parse::parser;
use diagnostic::span_handler;
use codemap::{CodeMap, span, ExpnInfo, ExpandedFrom};
use ast_util::dummy_sp;
use parse::token;

// new-style macro! tt code:
//
//    syntax_expander_tt, syntax_expander_tt_item, mac_result,
//    normal_tt, item_tt
//
// also note that ast::mac used to have a bunch of extraneous cases and
// is now probably a redundant AST node, can be merged with
// ast::mac_invoc_tt.

type macro_def = {name: ~str, ext: syntax_extension};

type item_decorator =
    fn@(ext_ctxt, span, ast::meta_item, ~[@ast::item]) -> ~[@ast::item];

type syntax_expander_tt = {expander: syntax_expander_tt_, span: Option<span>};
type syntax_expander_tt_ = fn@(ext_ctxt, span, ~[ast::token_tree])
    -> mac_result;

type syntax_expander_tt_item
    = {expander: syntax_expander_tt_item_, span: Option<span>};
type syntax_expander_tt_item_
    = fn@(ext_ctxt, span, ast::ident, ~[ast::token_tree]) -> mac_result;

enum mac_result {
    mr_expr(@ast::expr),
    mr_item(@ast::item),
    mr_any(fn@()-> @ast::expr, fn@()-> Option<@ast::item>, fn@()->@ast::stmt),
    mr_def(macro_def)
}

enum syntax_extension {

    // #[auto_encode] and such
    item_decorator(item_decorator),

    // Token-tree expanders
    normal_tt(syntax_expander_tt),

    // perhaps macro_rules! will lose its odd special identifier argument,
    // and this can go away also
    item_tt(syntax_expander_tt_item),
}

// A temporary hard-coded map of methods for expanding syntax extension
// AST nodes into full ASTs
fn syntax_expander_table() -> HashMap<~str, syntax_extension> {
    fn builtin_normal_tt(f: syntax_expander_tt_) -> syntax_extension {
        normal_tt({expander: f, span: None})
    }
    fn builtin_item_tt(f: syntax_expander_tt_item_) -> syntax_extension {
        item_tt({expander: f, span: None})
    }
    let syntax_expanders = HashMap();
    syntax_expanders.insert(~"macro_rules",
                            builtin_item_tt(
                                ext::tt::macro_rules::add_new_extension));
    syntax_expanders.insert(~"fmt",
                            builtin_normal_tt(ext::fmt::expand_syntax_ext));
    syntax_expanders.insert(
        ~"auto_serialize",
        item_decorator(ext::auto_serialize::expand_auto_serialize));
    syntax_expanders.insert(
        ~"auto_deserialize",
        item_decorator(ext::auto_serialize::expand_auto_deserialize));
    syntax_expanders.insert(
        ~"auto_encode",
        item_decorator(ext::auto_encode::expand_auto_encode));
    syntax_expanders.insert(
        ~"auto_decode",
        item_decorator(ext::auto_encode::expand_auto_decode));
    syntax_expanders.insert(~"env",
                            builtin_normal_tt(ext::env::expand_syntax_ext));
    syntax_expanders.insert(~"concat_idents",
                            builtin_normal_tt(
                                ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert(~"log_syntax",
                            builtin_normal_tt(
                                ext::log_syntax::expand_syntax_ext));
    syntax_expanders.insert(~"deriving_eq",
                            item_decorator(
                                ext::deriving::expand_deriving_eq));
    syntax_expanders.insert(~"deriving_iter_bytes",
                            item_decorator(
                                ext::deriving::expand_deriving_iter_bytes));

    // Quasi-quoting expanders
    syntax_expanders.insert(
        ~"quote_tokens", builtin_normal_tt(ext::quote::expand_quote_tokens));
    syntax_expanders.insert(~"quote_expr",
                            builtin_normal_tt(ext::quote::expand_quote_expr));
    syntax_expanders.insert(~"quote_ty",
                            builtin_normal_tt(ext::quote::expand_quote_ty));
    syntax_expanders.insert(~"quote_item",
                            builtin_normal_tt(ext::quote::expand_quote_item));
    syntax_expanders.insert(~"quote_pat",
                            builtin_normal_tt(ext::quote::expand_quote_pat));
    syntax_expanders.insert(~"quote_stmt",
                            builtin_normal_tt(ext::quote::expand_quote_stmt));

    syntax_expanders.insert(~"line",
                            builtin_normal_tt(
                                ext::source_util::expand_line));
    syntax_expanders.insert(~"col",
                            builtin_normal_tt(
                                ext::source_util::expand_col));
    syntax_expanders.insert(~"file",
                            builtin_normal_tt(
                                ext::source_util::expand_file));
    syntax_expanders.insert(~"stringify",
                            builtin_normal_tt(
                                ext::source_util::expand_stringify));
    syntax_expanders.insert(~"include",
                            builtin_normal_tt(
                                ext::source_util::expand_include));
    syntax_expanders.insert(~"include_str",
                            builtin_normal_tt(
                                ext::source_util::expand_include_str));
    syntax_expanders.insert(~"include_bin",
                            builtin_normal_tt(
                                ext::source_util::expand_include_bin));
    syntax_expanders.insert(~"module_path",
                            builtin_normal_tt(
                                ext::source_util::expand_mod));
    syntax_expanders.insert(~"proto",
                            builtin_item_tt(ext::pipes::expand_proto));
    syntax_expanders.insert(
        ~"trace_macros",
        builtin_normal_tt(ext::trace_macros::expand_trace_macros));
    return syntax_expanders;
}

// One of these is made during expansion and incrementally updated as we go;
// when a macro expansion occurs, the resulting nodes have the backtrace()
// -> expn_info of their expansion context stored into their span.
trait ext_ctxt {
    fn codemap() -> @CodeMap;
    fn parse_sess() -> parse::parse_sess;
    fn cfg() -> ast::crate_cfg;
    fn call_site() -> span;
    fn print_backtrace();
    fn backtrace() -> Option<@ExpnInfo>;
    fn mod_push(mod_name: ast::ident);
    fn mod_pop();
    fn mod_path() -> ~[ast::ident];
    fn bt_push(ei: codemap::ExpnInfo);
    fn bt_pop();
    fn span_fatal(sp: span, msg: &str) -> !;
    fn span_err(sp: span, msg: &str);
    fn span_warn(sp: span, msg: &str);
    fn span_unimpl(sp: span, msg: &str) -> !;
    fn span_bug(sp: span, msg: &str) -> !;
    fn bug(msg: &str) -> !;
    fn next_id() -> ast::node_id;
    pure fn trace_macros() -> bool;
    fn set_trace_macros(x: bool);
    /* for unhygienic identifier transformation */
    fn str_of(id: ast::ident) -> ~str;
    fn ident_of(st: ~str) -> ast::ident;
}

fn mk_ctxt(parse_sess: parse::parse_sess,
           cfg: ast::crate_cfg) -> ext_ctxt {
    type ctxt_repr = {parse_sess: parse::parse_sess,
                      cfg: ast::crate_cfg,
                      mut backtrace: Option<@ExpnInfo>,
                      mut mod_path: ~[ast::ident],
                      mut trace_mac: bool};
    impl ctxt_repr: ext_ctxt {
        fn codemap() -> @CodeMap { self.parse_sess.cm }
        fn parse_sess() -> parse::parse_sess { self.parse_sess }
        fn cfg() -> ast::crate_cfg { self.cfg }
        fn call_site() -> span {
            match self.backtrace {
                Some(@ExpandedFrom({call_site: cs, _})) => cs,
                None => self.bug(~"missing top span")
            }
        }
        fn print_backtrace() { }
        fn backtrace() -> Option<@ExpnInfo> { self.backtrace }
        fn mod_push(i: ast::ident) { self.mod_path.push(i); }
        fn mod_pop() { self.mod_path.pop(); }
        fn mod_path() -> ~[ast::ident] { return self.mod_path; }
        fn bt_push(ei: codemap::ExpnInfo) {
            match ei {
              ExpandedFrom({call_site: cs, callie: ref callie}) => {
                self.backtrace =
                    Some(@ExpandedFrom({
                        call_site: span {lo: cs.lo, hi: cs.hi,
                                         expn_info: self.backtrace},
                        callie: (*callie)}));
              }
            }
        }
        fn bt_pop() {
            match self.backtrace {
              Some(@ExpandedFrom({
                  call_site: span {expn_info: prev, _}, _
              })) => {
                self.backtrace = prev
              }
              _ => self.bug(~"tried to pop without a push")
            }
        }
        fn span_fatal(sp: span, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_fatal(sp, msg);
        }
        fn span_err(sp: span, msg: &str) {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_err(sp, msg);
        }
        fn span_warn(sp: span, msg: &str) {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_warn(sp, msg);
        }
        fn span_unimpl(sp: span, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_unimpl(sp, msg);
        }
        fn span_bug(sp: span, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_bug(sp, msg);
        }
        fn bug(msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.handler().bug(msg);
        }
        fn next_id() -> ast::node_id {
            return parse::next_node_id(self.parse_sess);
        }
        pure fn trace_macros() -> bool {
            self.trace_mac
        }
        fn set_trace_macros(x: bool) {
            self.trace_mac = x
        }

        fn str_of(id: ast::ident) -> ~str {
            *self.parse_sess.interner.get(id)
        }
        fn ident_of(st: ~str) -> ast::ident {
            self.parse_sess.interner.intern(@st)
        }
    }
    let imp: ctxt_repr = {
        parse_sess: parse_sess,
        cfg: cfg,
        mut backtrace: None,
        mut mod_path: ~[],
        mut trace_mac: false
    };
    move ((move imp) as ext_ctxt)
}

fn expr_to_str(cx: ext_ctxt, expr: @ast::expr, err_msg: ~str) -> ~str {
    match expr.node {
      ast::expr_lit(l) => match l.node {
        ast::lit_str(s) => return *s,
        _ => cx.span_fatal(l.span, err_msg)
      },
      _ => cx.span_fatal(expr.span, err_msg)
    }
}

fn expr_to_ident(cx: ext_ctxt,
                 expr: @ast::expr,
                 err_msg: ~str) -> ast::ident {
    match expr.node {
      ast::expr_path(p) => {
        if vec::len(p.types) > 0u || vec::len(p.idents) != 1u {
            cx.span_fatal(expr.span, err_msg);
        }
        return p.idents[0];
      }
      _ => cx.span_fatal(expr.span, err_msg)
    }
}

fn check_zero_tts(cx: ext_ctxt, sp: span, tts: &[ast::token_tree],
                  name: &str) {
    if tts.len() != 0 {
        cx.span_fatal(sp, fmt!("%s takes no arguments", name));
    }
}

fn get_single_str_from_tts(cx: ext_ctxt, sp: span, tts: &[ast::token_tree],
                           name: &str) -> ~str {
    if tts.len() != 1 {
        cx.span_fatal(sp, fmt!("%s takes 1 argument.", name));
    }

    match tts[0] {
        ast::tt_tok(_, token::LIT_STR(ident)) => cx.str_of(ident),
        _ =>
        cx.span_fatal(sp, fmt!("%s requires a string.", name))
    }
}

fn get_exprs_from_tts(cx: ext_ctxt, tts: ~[ast::token_tree])
    -> ~[@ast::expr] {
    let p = parse::new_parser_from_tts(cx.parse_sess(),
                                       cx.cfg(),
                                       tts);
    let mut es = ~[];
    while p.token != token::EOF {
        if es.len() != 0 {
            p.eat(token::COMMA);
        }
        es.push(p.parse_expr());
    }
    es
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
