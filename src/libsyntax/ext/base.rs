// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast;
use codemap;
use codemap::{CodeMap, span, ExpnInfo, ExpandedFrom};
use codemap::CallInfo;
use diagnostic::span_handler;
use ext;
use parse;
use parse::token;

use core::vec;
use core::hashmap::linear::LinearMap;

// new-style macro! tt code:
//
//    SyntaxExpanderTT, SyntaxExpanderTTItem, MacResult,
//    NormalTT, IdentTT
//
// also note that ast::mac used to have a bunch of extraneous cases and
// is now probably a redundant AST node, can be merged with
// ast::mac_invoc_tt.

pub struct MacroDef {
    name: ~str,
    ext: SyntaxExtension
}

pub type ItemDecorator = @fn(@ext_ctxt,
                             span,
                             @ast::meta_item,
                             ~[@ast::item])
                          -> ~[@ast::item];

pub struct SyntaxExpanderTT {
    expander: SyntaxExpanderTTFun,
    span: Option<span>
}

pub type SyntaxExpanderTTFun = @fn(@ext_ctxt,
                                   span,
                                   &[ast::token_tree])
                                -> MacResult;

pub struct SyntaxExpanderTTItem {
    expander: SyntaxExpanderTTItemFun,
    span: Option<span>
}

pub type SyntaxExpanderTTItemFun = @fn(@ext_ctxt,
                                       span,
                                       ast::ident,
                                       ~[ast::token_tree])
                                    -> MacResult;

pub enum MacResult {
    MRExpr(@ast::expr),
    MRItem(@ast::item),
    MRAny(@fn() -> @ast::expr,
          @fn() -> Option<@ast::item>,
          @fn() -> @ast::stmt),
    MRDef(MacroDef)
}

pub enum SyntaxExtension {

    // #[auto_encode] and such
    ItemDecorator(ItemDecorator),

    // Token-tree expanders
    NormalTT(SyntaxExpanderTT),

    // An IdentTT is a macro that has an
    // identifier in between the name of the
    // macro and the argument. Currently,
    // the only examples of this are
    // macro_rules! and proto!

    // perhaps macro_rules! will lose its odd special identifier argument,
    // and this can go away also
    IdentTT(SyntaxExpanderTTItem),
}

pub type SyntaxEnv = @mut MapChain<Name, Transformer>;

// Name : the domain of SyntaxEnvs
// want to change these to uints....
// note that we use certain strings that are not legal as identifiers
// to indicate, for instance, how blocks are supposed to behave.
type Name = @~str;

// Transformer : the codomain of SyntaxEnvs

// NB: it may seem crazy to lump both of these into one environment;
// what would it mean to bind "foo" to BlockLimit(true)? The idea
// is that this follows the lead of MTWT, and accommodates growth
// toward a more uniform syntax syntax (sorry) where blocks are just
// another kind of transformer.

pub enum Transformer {
    // this identifier maps to a syntax extension or macro
    SE(SyntaxExtension),
    // should blocks occurring here limit macro scopes?
    ScopeMacros(bool)
}

// The base map of methods for expanding syntax extension
// AST nodes into full ASTs
pub fn syntax_expander_table() -> SyntaxEnv {
    // utility function to simplify creating NormalTT syntax extensions
    fn builtin_normal_tt(f: SyntaxExpanderTTFun) -> @Transformer {
        @SE(NormalTT(SyntaxExpanderTT{expander: f, span: None}))
    }
    // utility function to simplify creating IdentTT syntax extensions
    fn builtin_item_tt(f: SyntaxExpanderTTItemFun) -> @Transformer {
        @SE(IdentTT(SyntaxExpanderTTItem{expander: f, span: None}))
    }
    let mut syntax_expanders = LinearMap::new();
    // NB identifier starts with space, and can't conflict with legal idents
    syntax_expanders.insert(@~" block",
                            @ScopeMacros(true));
    syntax_expanders.insert(@~"macro_rules",
                            builtin_item_tt(
                                ext::tt::macro_rules::add_new_extension));
    syntax_expanders.insert(@~"fmt",
                            builtin_normal_tt(ext::fmt::expand_syntax_ext));
    syntax_expanders.insert(
        @~"auto_encode",
        @SE(ItemDecorator(ext::auto_encode::expand_auto_encode)));
    syntax_expanders.insert(
        @~"auto_decode",
        @SE(ItemDecorator(ext::auto_encode::expand_auto_decode)));
    syntax_expanders.insert(@~"env",
                            builtin_normal_tt(ext::env::expand_syntax_ext));
    syntax_expanders.insert(@~"concat_idents",
                            builtin_normal_tt(
                                ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert(@~"log_syntax",
                            builtin_normal_tt(
                                ext::log_syntax::expand_syntax_ext));
    syntax_expanders.insert(@~"deriving",
                            @SE(ItemDecorator(
                                ext::deriving::expand_meta_deriving)));
    syntax_expanders.insert(@~"deriving_eq",
                            @SE(ItemDecorator(
                                ext::deriving::eq::expand_deriving_obsolete)));
    syntax_expanders.insert(@~"deriving_iter_bytes",
                            @SE(ItemDecorator(
                                ext::deriving::iter_bytes::expand_deriving_obsolete)));
    syntax_expanders.insert(@~"deriving_clone",
                            @SE(ItemDecorator(
                                ext::deriving::clone::expand_deriving_obsolete)));

    // Quasi-quoting expanders
    syntax_expanders.insert(@~"quote_tokens",
                       builtin_normal_tt(ext::quote::expand_quote_tokens));
    syntax_expanders.insert(@~"quote_expr",
                       builtin_normal_tt(ext::quote::expand_quote_expr));
    syntax_expanders.insert(@~"quote_ty",
                       builtin_normal_tt(ext::quote::expand_quote_ty));
    syntax_expanders.insert(@~"quote_item",
                       builtin_normal_tt(ext::quote::expand_quote_item));
    syntax_expanders.insert(@~"quote_pat",
                       builtin_normal_tt(ext::quote::expand_quote_pat));
    syntax_expanders.insert(@~"quote_stmt",
                       builtin_normal_tt(ext::quote::expand_quote_stmt));

    syntax_expanders.insert(@~"line",
                            builtin_normal_tt(
                                ext::source_util::expand_line));
    syntax_expanders.insert(@~"col",
                            builtin_normal_tt(
                                ext::source_util::expand_col));
    syntax_expanders.insert(@~"file",
                            builtin_normal_tt(
                                ext::source_util::expand_file));
    syntax_expanders.insert(@~"stringify",
                            builtin_normal_tt(
                                ext::source_util::expand_stringify));
    syntax_expanders.insert(@~"include",
                            builtin_normal_tt(
                                ext::source_util::expand_include));
    syntax_expanders.insert(@~"include_str",
                            builtin_normal_tt(
                                ext::source_util::expand_include_str));
    syntax_expanders.insert(@~"include_bin",
                            builtin_normal_tt(
                                ext::source_util::expand_include_bin));
    syntax_expanders.insert(@~"module_path",
                            builtin_normal_tt(
                                ext::source_util::expand_mod));
    syntax_expanders.insert(@~"proto",
                            builtin_item_tt(ext::pipes::expand_proto));
    syntax_expanders.insert(@~"asm",
                            builtin_normal_tt(ext::asm::expand_asm));
    syntax_expanders.insert(
        @~"trace_macros",
        builtin_normal_tt(ext::trace_macros::expand_trace_macros));
    MapChain::new(~syntax_expanders)
}

// One of these is made during expansion and incrementally updated as we go;
// when a macro expansion occurs, the resulting nodes have the backtrace()
// -> expn_info of their expansion context stored into their span.
pub trait ext_ctxt {
    fn codemap(@mut self) -> @CodeMap;
    fn parse_sess(@mut self) -> @mut parse::ParseSess;
    fn cfg(@mut self) -> ast::crate_cfg;
    fn call_site(@mut self) -> span;
    fn print_backtrace(@mut self);
    fn backtrace(@mut self) -> Option<@ExpnInfo>;
    fn mod_push(@mut self, mod_name: ast::ident);
    fn mod_pop(@mut self);
    fn mod_path(@mut self) -> ~[ast::ident];
    fn bt_push(@mut self, ei: codemap::ExpnInfo);
    fn bt_pop(@mut self);
    fn span_fatal(@mut self, sp: span, msg: &str) -> !;
    fn span_err(@mut self, sp: span, msg: &str);
    fn span_warn(@mut self, sp: span, msg: &str);
    fn span_unimpl(@mut self, sp: span, msg: &str) -> !;
    fn span_bug(@mut self, sp: span, msg: &str) -> !;
    fn bug(@mut self, msg: &str) -> !;
    fn next_id(@mut self) -> ast::node_id;
    fn trace_macros(@mut self) -> bool;
    fn set_trace_macros(@mut self, x: bool);
    /* for unhygienic identifier transformation */
    fn str_of(@mut self, id: ast::ident) -> ~str;
    fn ident_of(@mut self, st: ~str) -> ast::ident;
}

pub fn mk_ctxt(parse_sess: @mut parse::ParseSess, +cfg: ast::crate_cfg)
            -> @ext_ctxt {
    struct CtxtRepr {
        parse_sess: @mut parse::ParseSess,
        cfg: ast::crate_cfg,
        backtrace: @mut Option<@ExpnInfo>,
        mod_path: ~[ast::ident],
        trace_mac: bool
    }
    impl ext_ctxt for CtxtRepr {
        fn codemap(@mut self) -> @CodeMap { self.parse_sess.cm }
        fn parse_sess(@mut self) -> @mut parse::ParseSess { self.parse_sess }
        fn cfg(@mut self) -> ast::crate_cfg { copy self.cfg }
        fn call_site(@mut self) -> span {
            match *self.backtrace {
                Some(@ExpandedFrom(CallInfo {call_site: cs, _})) => cs,
                None => self.bug(~"missing top span")
            }
        }
        fn print_backtrace(@mut self) { }
        fn backtrace(@mut self) -> Option<@ExpnInfo> { *self.backtrace }
        fn mod_push(@mut self, i: ast::ident) { self.mod_path.push(i); }
        fn mod_pop(@mut self) { self.mod_path.pop(); }
        fn mod_path(@mut self) -> ~[ast::ident] { copy self.mod_path }
        fn bt_push(@mut self, ei: codemap::ExpnInfo) {
            match ei {
              ExpandedFrom(CallInfo {call_site: cs, callee: ref callee}) => {
                *self.backtrace =
                    Some(@ExpandedFrom(CallInfo {
                        call_site: span {lo: cs.lo, hi: cs.hi,
                                         expn_info: *self.backtrace},
                        callee: copy *callee}));
              }
            }
        }
        fn bt_pop(@mut self) {
            match *self.backtrace {
              Some(@ExpandedFrom(CallInfo {
                  call_site: span {expn_info: prev, _}, _
              })) => {
                *self.backtrace = prev
              }
              _ => self.bug(~"tried to pop without a push")
            }
        }
        fn span_fatal(@mut self, sp: span, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_fatal(sp, msg);
        }
        fn span_err(@mut self, sp: span, msg: &str) {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_err(sp, msg);
        }
        fn span_warn(@mut self, sp: span, msg: &str) {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_warn(sp, msg);
        }
        fn span_unimpl(@mut self, sp: span, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_unimpl(sp, msg);
        }
        fn span_bug(@mut self, sp: span, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.span_bug(sp, msg);
        }
        fn bug(@mut self, msg: &str) -> ! {
            self.print_backtrace();
            self.parse_sess.span_diagnostic.handler().bug(msg);
        }
        fn next_id(@mut self) -> ast::node_id {
            return parse::next_node_id(self.parse_sess);
        }
        fn trace_macros(@mut self) -> bool {
            self.trace_mac
        }
        fn set_trace_macros(@mut self, x: bool) {
            self.trace_mac = x
        }
        fn str_of(@mut self, id: ast::ident) -> ~str {
            copy *self.parse_sess.interner.get(id)
        }
        fn ident_of(@mut self, st: ~str) -> ast::ident {
            self.parse_sess.interner.intern(@/*bad*/ copy st)
        }
    }
    let imp: @mut CtxtRepr = @mut CtxtRepr {
        parse_sess: parse_sess,
        cfg: cfg,
        backtrace: @mut None,
        mod_path: ~[],
        trace_mac: false
    };
    ((imp) as @ext_ctxt)
}

pub fn expr_to_str(cx: @ext_ctxt, expr: @ast::expr, err_msg: ~str) -> ~str {
    match expr.node {
      ast::expr_lit(l) => match l.node {
        ast::lit_str(s) => copy *s,
        _ => cx.span_fatal(l.span, err_msg)
      },
      _ => cx.span_fatal(expr.span, err_msg)
    }
}

pub fn expr_to_ident(cx: @ext_ctxt,
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

pub fn check_zero_tts(cx: @ext_ctxt, sp: span, tts: &[ast::token_tree],
                      name: &str) {
    if tts.len() != 0 {
        cx.span_fatal(sp, fmt!("%s takes no arguments", name));
    }
}

pub fn get_single_str_from_tts(cx: @ext_ctxt,
                               sp: span,
                               tts: &[ast::token_tree],
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

pub fn get_exprs_from_tts(cx: @ext_ctxt, tts: &[ast::token_tree])
                       -> ~[@ast::expr] {
    let p = parse::new_parser_from_tts(cx.parse_sess(),
                                       cx.cfg(),
                                       vec::from_slice(tts));
    let mut es = ~[];
    while *p.token != token::EOF {
        if es.len() != 0 {
            p.eat(&token::COMMA);
        }
        es.push(p.parse_expr());
    }
    es
}

// in order to have some notion of scoping for macros,
// we want to implement the notion of a transformation
// environment.

// This environment maps Names to Transformers.
// Initially, this includes macro definitions and
// block directives.



// Actually, the following implementation is parameterized
// by both key and value types.

//impl question: how to implement it? Initially, the
// env will contain only macros, so it might be painful
// to add an empty frame for every context. Let's just
// get it working, first....

// NB! the mutability of the underlying maps means that
// if expansion is out-of-order, a deeper scope may be
// able to refer to a macro that was added to an enclosing
// scope lexically later than the deeper scope.

// Note on choice of representation: I've been pushed to
// use a top-level managed pointer by some difficulties
// with pushing and popping functionally, and the ownership
// issues.  As a result, the values returned by the table
// also need to be managed; the &'self ... type that Maps
// return won't work for things that need to get outside
// of that managed pointer.  The easiest way to do this
// is just to insist that the values in the tables are
// managed to begin with.

// a transformer env is either a base map or a map on top
// of another chain.
pub enum MapChain<K,V> {
    BaseMapChain(~LinearMap<K,@V>),
    ConsMapChain(~LinearMap<K,@V>,@mut MapChain<K,V>)
}


// get the map from an env frame
impl <K: Eq + Hash + IterBytes ,V: Copy> MapChain<K,V>{

    // Constructor. I don't think we need a zero-arg one.
    fn new(+init: ~LinearMap<K,@V>) -> @mut MapChain<K,V> {
        @mut BaseMapChain(init)
    }

    // add a new frame to the environment (functionally)
    fn push_frame (@mut self) -> @mut MapChain<K,V> {
        @mut ConsMapChain(~LinearMap::new() ,self)
    }

// no need for pop, it'll just be functional.

    // utility fn...

    // ugh: can't get this to compile with mut because of the
    // lack of flow sensitivity.
    fn get_map(&self) -> &'self LinearMap<K,@V> {
        match *self {
            BaseMapChain (~ref map) => map,
            ConsMapChain (~ref map,_) => map
        }
    }

// traits just don't work anywhere...?
//pub impl Map<Name,SyntaxExtension> for MapChain {

    fn contains_key (&self, key: &K) -> bool {
        match *self {
            BaseMapChain (ref map) => map.contains_key(key),
            ConsMapChain (ref map,ref rest) =>
            (map.contains_key(key)
             || rest.contains_key(key))
        }
    }
    // should each_key and each_value operate on shadowed
    // names? I think not.
    // delaying implementing this....
    fn each_key (&self, _f: &fn (&K)->bool) {
        fail!(~"unimplemented 2013-02-15T10:01");
    }

    fn each_value (&self, _f: &fn (&V) -> bool) {
        fail!(~"unimplemented 2013-02-15T10:02");
    }

    // Returns a copy of the value that the name maps to.
    // Goes down the chain 'til it finds one (or bottom out).
    fn find (&self, key: &K) -> Option<@V> {
        match self.get_map().find (key) {
            Some(ref v) => Some(**v),
            None => match *self {
                BaseMapChain (_) => None,
                ConsMapChain (_,ref rest) => rest.find(key)
            }
        }
    }

    // insert the binding into the top-level map
    fn insert (&mut self, +key: K, +ext: @V) -> bool {
        // can't abstract over get_map because of flow sensitivity...
        match *self {
            BaseMapChain (~ref mut map) => map.insert(key, ext),
            ConsMapChain (~ref mut map,_) => map.insert(key,ext)
        }
    }

}

#[cfg(test)]
mod test {
    use super::MapChain;
    use core::hashmap::linear::LinearMap;

    #[test] fn testenv () {
        let mut a = LinearMap::new();
        a.insert (@~"abc",@15);
        let m = MapChain::new(~a);
        m.insert (@~"def",@16);
        // FIXME: #4492 (ICE)  assert_eq!(m.find(&@~"abc"),Some(@15));
        //  ....               assert_eq!(m.find(&@~"def"),Some(@16));
        assert_eq!(*(m.find(&@~"abc").get()),15);
        assert_eq!(*(m.find(&@~"def").get()),16);
        let n = m.push_frame();
        // old bindings are still present:
        assert_eq!(*(n.find(&@~"abc").get()),15);
        assert_eq!(*(n.find(&@~"def").get()),16);
        n.insert (@~"def",@17);
        // n shows the new binding
        assert_eq!(*(n.find(&@~"abc").get()),15);
        assert_eq!(*(n.find(&@~"def").get()),17);
        // ... but m still has the old ones
        // FIXME: #4492: assert_eq!(m.find(&@~"abc"),Some(@15));
        // FIXME: #4492: assert_eq!(m.find(&@~"def"),Some(@16));
        assert_eq!(*(m.find(&@~"abc").get()),15);
        assert_eq!(*(m.find(&@~"def").get()),16);
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
