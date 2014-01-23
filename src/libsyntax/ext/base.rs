// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::Name;
use codemap;
use codemap::{CodeMap, Span, ExpnInfo};
use ext;
use ext::expand;
use parse;
use parse::token;
use parse::token::{ident_to_str, intern, str_to_ident};
use util::small_vector::SmallVector;

use std::hashmap::HashMap;
use std::unstable::dynamic_lib::DynamicLibrary;

// new-style macro! tt code:
//
//    MacResult, NormalTT, IdentTT
//
// also note that ast::Mac used to have a bunch of extraneous cases and
// is now probably a redundant AST node, can be merged with
// ast::MacInvocTT.

pub struct MacroDef {
    name: @str,
    ext: SyntaxExtension
}

pub type ItemDecorator =
    fn(&ExtCtxt, Span, @ast::MetaItem, ~[@ast::Item]) -> ~[@ast::Item];

pub struct SyntaxExpanderTT {
    expander: SyntaxExpanderTTExpander,
    span: Option<Span>
}

pub trait SyntaxExpanderTTTrait {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              token_tree: &[ast::TokenTree],
              context: ast::SyntaxContext)
              -> MacResult;
}

pub type SyntaxExpanderTTFunNoCtxt =
    fn(ecx: &mut ExtCtxt, span: codemap::Span, token_tree: &[ast::TokenTree])
       -> MacResult;

enum SyntaxExpanderTTExpander {
    SyntaxExpanderTTExpanderWithoutContext(SyntaxExpanderTTFunNoCtxt),
}

impl SyntaxExpanderTTTrait for SyntaxExpanderTT {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              token_tree: &[ast::TokenTree],
              _: ast::SyntaxContext)
              -> MacResult {
        match self.expander {
            SyntaxExpanderTTExpanderWithoutContext(f) => {
                f(ecx, span, token_tree)
            }
        }
    }
}

enum SyntaxExpanderTTItemExpander {
    SyntaxExpanderTTItemExpanderWithContext(SyntaxExpanderTTItemFun),
    SyntaxExpanderTTItemExpanderWithoutContext(SyntaxExpanderTTItemFunNoCtxt),
}

pub struct SyntaxExpanderTTItem {
    expander: SyntaxExpanderTTItemExpander,
    span: Option<Span>
}

pub trait SyntaxExpanderTTItemTrait {
    fn expand(&self,
              cx: &mut ExtCtxt,
              sp: Span,
              ident: ast::Ident,
              token_tree: ~[ast::TokenTree],
              context: ast::SyntaxContext)
              -> MacResult;
}

impl SyntaxExpanderTTItemTrait for SyntaxExpanderTTItem {
    fn expand(&self,
              cx: &mut ExtCtxt,
              sp: Span,
              ident: ast::Ident,
              token_tree: ~[ast::TokenTree],
              context: ast::SyntaxContext)
              -> MacResult {
        match self.expander {
            SyntaxExpanderTTItemExpanderWithContext(fun) => {
                fun(cx, sp, ident, token_tree, context)
            }
            SyntaxExpanderTTItemExpanderWithoutContext(fun) => {
                fun(cx, sp, ident, token_tree)
            }
        }
    }
}

pub type SyntaxExpanderTTItemFun =
    fn(&mut ExtCtxt, Span, ast::Ident, ~[ast::TokenTree], ast::SyntaxContext)
       -> MacResult;

pub type SyntaxExpanderTTItemFunNoCtxt =
    fn(&mut ExtCtxt, Span, ast::Ident, ~[ast::TokenTree]) -> MacResult;

pub type MacroCrateRegistrationFun =
    extern "Rust" fn(|ast::Name, SyntaxExtension|);

pub trait AnyMacro {
    fn make_expr(&self) -> @ast::Expr;
    fn make_items(&self) -> SmallVector<@ast::Item>;
    fn make_stmt(&self) -> @ast::Stmt;
}

pub enum MacResult {
    MRExpr(@ast::Expr),
    MRItem(@ast::Item),
    MRAny(@AnyMacro),
    MRDef(MacroDef),
}
impl MacResult {
    /// Create an empty expression MacResult; useful for satisfying
    /// type signatures after emitting a non-fatal error (which stop
    /// compilation well before the validity (or otherwise)) of the
    /// expression are checked.
    pub fn dummy_expr() -> MacResult {
        MRExpr(@ast::Expr {
                id: ast::DUMMY_NODE_ID, node: ast::ExprLogLevel, span: codemap::DUMMY_SP
            })
    }
}

pub enum SyntaxExtension {
    // #[deriving] and such
    ItemDecorator(ItemDecorator),

    // Token-tree expanders
    NormalTT(~SyntaxExpanderTTTrait:'static, Option<Span>),

    // An IdentTT is a macro that has an
    // identifier in between the name of the
    // macro and the argument. Currently,
    // the only examples of this is
    // macro_rules!

    // perhaps macro_rules! will lose its odd special identifier argument,
    // and this can go away also
    IdentTT(~SyntaxExpanderTTItemTrait:'static, Option<Span>),
}

pub struct BlockInfo {
    // should macros escape from this scope?
    macros_escape : bool,
    // what are the pending renames?
    pending_renames : RenameList,
    // references for crates loaded in this scope
    macro_crates: ~[DynamicLibrary],
}

impl BlockInfo {
    pub fn new() -> BlockInfo {
        BlockInfo {
            macros_escape: false,
            pending_renames: ~[],
            macro_crates: ~[],
        }
    }
}

// a list of ident->name renamings
pub type RenameList = ~[(ast::Ident,Name)];

// The base map of methods for expanding syntax extension
// AST nodes into full ASTs
pub fn syntax_expander_table() -> SyntaxEnv {
    // utility function to simplify creating NormalTT syntax extensions
    fn builtin_normal_tt_no_ctxt(f: SyntaxExpanderTTFunNoCtxt)
                                 -> SyntaxExtension {
        NormalTT(~SyntaxExpanderTT{
            expander: SyntaxExpanderTTExpanderWithoutContext(f),
            span: None,
        },
        None)
    }

    let mut syntax_expanders = SyntaxEnv::new();
    syntax_expanders.insert(intern(&"macro_rules"),
                            IdentTT(~SyntaxExpanderTTItem {
                                expander: SyntaxExpanderTTItemExpanderWithContext(
                                    ext::tt::macro_rules::add_new_extension),
                                span: None,
                            },
                            None));
    syntax_expanders.insert(intern(&"fmt"),
                            builtin_normal_tt_no_ctxt(
                                ext::fmt::expand_syntax_ext));
    syntax_expanders.insert(intern(&"format_args"),
                            builtin_normal_tt_no_ctxt(
                                ext::format::expand_args));
    syntax_expanders.insert(intern(&"env"),
                            builtin_normal_tt_no_ctxt(
                                    ext::env::expand_env));
    syntax_expanders.insert(intern(&"option_env"),
                            builtin_normal_tt_no_ctxt(
                                    ext::env::expand_option_env));
    syntax_expanders.insert(intern("bytes"),
                            builtin_normal_tt_no_ctxt(
                                    ext::bytes::expand_syntax_ext));
    syntax_expanders.insert(intern("concat_idents"),
                            builtin_normal_tt_no_ctxt(
                                    ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert(intern("concat"),
                            builtin_normal_tt_no_ctxt(
                                    ext::concat::expand_syntax_ext));
    syntax_expanders.insert(intern(&"log_syntax"),
                            builtin_normal_tt_no_ctxt(
                                    ext::log_syntax::expand_syntax_ext));
    syntax_expanders.insert(intern(&"deriving"),
                            ItemDecorator(ext::deriving::expand_meta_deriving));

    // Quasi-quoting expanders
    syntax_expanders.insert(intern(&"quote_tokens"),
                       builtin_normal_tt_no_ctxt(
                            ext::quote::expand_quote_tokens));
    syntax_expanders.insert(intern(&"quote_expr"),
                       builtin_normal_tt_no_ctxt(
                            ext::quote::expand_quote_expr));
    syntax_expanders.insert(intern(&"quote_ty"),
                       builtin_normal_tt_no_ctxt(
                            ext::quote::expand_quote_ty));
    syntax_expanders.insert(intern(&"quote_item"),
                       builtin_normal_tt_no_ctxt(
                            ext::quote::expand_quote_item));
    syntax_expanders.insert(intern(&"quote_pat"),
                       builtin_normal_tt_no_ctxt(
                            ext::quote::expand_quote_pat));
    syntax_expanders.insert(intern(&"quote_stmt"),
                       builtin_normal_tt_no_ctxt(
                            ext::quote::expand_quote_stmt));

    syntax_expanders.insert(intern(&"line"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_line));
    syntax_expanders.insert(intern(&"col"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_col));
    syntax_expanders.insert(intern(&"file"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_file));
    syntax_expanders.insert(intern(&"stringify"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_stringify));
    syntax_expanders.insert(intern(&"include"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_include));
    syntax_expanders.insert(intern(&"include_str"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_include_str));
    syntax_expanders.insert(intern(&"include_bin"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_include_bin));
    syntax_expanders.insert(intern(&"module_path"),
                            builtin_normal_tt_no_ctxt(
                                    ext::source_util::expand_mod));
    syntax_expanders.insert(intern(&"asm"),
                            builtin_normal_tt_no_ctxt(
                                    ext::asm::expand_asm));
    syntax_expanders.insert(intern(&"cfg"),
                            builtin_normal_tt_no_ctxt(
                                    ext::cfg::expand_cfg));
    syntax_expanders.insert(intern(&"trace_macros"),
                            builtin_normal_tt_no_ctxt(
                                    ext::trace_macros::expand_trace_macros));
    syntax_expanders
}

pub struct MacroCrate {
    lib: Option<Path>,
    cnum: ast::CrateNum,
}

pub trait CrateLoader {
    fn load_crate(&mut self, crate: &ast::ViewItem) -> MacroCrate;
    fn get_exported_macros(&mut self, crate_num: ast::CrateNum) -> ~[@ast::Item];
    fn get_registrar_symbol(&mut self, crate_num: ast::CrateNum) -> Option<~str>;
}

// One of these is made during expansion and incrementally updated as we go;
// when a macro expansion occurs, the resulting nodes have the backtrace()
// -> expn_info of their expansion context stored into their span.
pub struct ExtCtxt<'a> {
    parse_sess: @parse::ParseSess,
    cfg: ast::CrateConfig,
    backtrace: Option<@ExpnInfo>,
    loader: &'a mut CrateLoader,

    mod_path: ~[ast::Ident],
    trace_mac: bool
}

impl<'a> ExtCtxt<'a> {
    pub fn new<'a>(parse_sess: @parse::ParseSess, cfg: ast::CrateConfig,
               loader: &'a mut CrateLoader) -> ExtCtxt<'a> {
        ExtCtxt {
            parse_sess: parse_sess,
            cfg: cfg,
            backtrace: None,
            loader: loader,
            mod_path: ~[],
            trace_mac: false
        }
    }

    pub fn expand_expr(&mut self, mut e: @ast::Expr) -> @ast::Expr {
        loop {
            match e.node {
                ast::ExprMac(..) => {
                    let mut expander = expand::MacroExpander {
                        extsbox: syntax_expander_table(),
                        cx: self,
                    };
                    e = expand::expand_expr(e, &mut expander);
                }
                _ => return e
            }
        }
    }

    pub fn codemap(&self) -> @CodeMap { self.parse_sess.cm }
    pub fn parse_sess(&self) -> @parse::ParseSess { self.parse_sess }
    pub fn cfg(&self) -> ast::CrateConfig { self.cfg.clone() }
    pub fn call_site(&self) -> Span {
        match self.backtrace {
            Some(expn_info) => expn_info.call_site,
            None => self.bug("missing top span")
        }
    }
    pub fn print_backtrace(&self) { }
    pub fn backtrace(&self) -> Option<@ExpnInfo> { self.backtrace }
    pub fn mod_push(&mut self, i: ast::Ident) { self.mod_path.push(i); }
    pub fn mod_pop(&mut self) { self.mod_path.pop().unwrap(); }
    pub fn mod_path(&self) -> ~[ast::Ident] { self.mod_path.clone() }
    pub fn bt_push(&mut self, ei: codemap::ExpnInfo) {
        match ei {
            ExpnInfo {call_site: cs, callee: ref callee} => {
                self.backtrace =
                    Some(@ExpnInfo {
                        call_site: Span {lo: cs.lo, hi: cs.hi,
                                         expn_info: self.backtrace},
                        callee: *callee});
            }
        }
    }
    pub fn bt_pop(&mut self) {
        match self.backtrace {
            Some(expn_info) => self.backtrace = expn_info.call_site.expn_info,
            _ => self.bug("tried to pop without a push")
        }
    }
    /// Emit `msg` attached to `sp`, and stop compilation immediately.
    ///
    /// `span_err` should be strongly prefered where-ever possible:
    /// this should *only* be used when
    /// - continuing has a high risk of flow-on errors (e.g. errors in
    ///   declaring a macro would cause all uses of that macro to
    ///   complain about "undefined macro"), or
    /// - there is literally nothing else that can be done (however,
    ///   in most cases one can construct a dummy expression/item to
    ///   substitute; we never hit resolve/type-checking so the dummy
    ///   value doesn't have to match anything)
    pub fn span_fatal(&self, sp: Span, msg: &str) -> ! {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.span_fatal(sp, msg);
    }

    /// Emit `msg` attached to `sp`, without immediately stopping
    /// compilation.
    ///
    /// Compilation will be stopped in the near future (at the end of
    /// the macro expansion phase).
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.span_err(sp, msg);
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.span_warn(sp, msg);
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.span_unimpl(sp, msg);
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.span_bug(sp, msg);
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.span_note(sp, msg);
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.print_backtrace();
        self.parse_sess.span_diagnostic.handler().bug(msg);
    }
    pub fn trace_macros(&self) -> bool {
        self.trace_mac
    }
    pub fn set_trace_macros(&mut self, x: bool) {
        self.trace_mac = x
    }
    pub fn str_of(&self, id: ast::Ident) -> @str {
        ident_to_str(&id)
    }
    pub fn ident_of(&self, st: &str) -> ast::Ident {
        str_to_ident(st)
    }
}

/// Extract a string literal from `expr`, emitting `err_msg` if `expr`
/// is not a string literal. This does not stop compilation on error,
/// merely emits a non-fatal error and returns None.
pub fn expr_to_str(cx: &ExtCtxt, expr: @ast::Expr,
                   err_msg: &str) -> Option<(@str, ast::StrStyle)> {
    match expr.node {
        ast::ExprLit(l) => match l.node {
            ast::LitStr(s, style) => return Some((s, style)),
            _ => cx.span_err(l.span, err_msg)
        },
        _ => cx.span_err(expr.span, err_msg)
    }
    None
}

/// Non-fatally assert that `tts` is empty. Note that this function
/// returns even when `tts` is non-empty, macros that *need* to stop
/// compilation should call
/// `cx.parse_sess.span_diagnostic.abort_if_errors()` (this should be
/// done as rarely as possible).
pub fn check_zero_tts(cx: &ExtCtxt, sp: Span, tts: &[ast::TokenTree],
                      name: &str) {
    if tts.len() != 0 {
        cx.span_err(sp, format!("{} takes no arguments", name));
    }
}

/// Extract the string literal from the first token of `tts`. If this
/// is not a string literal, emit an error and return None.
pub fn get_single_str_from_tts(cx: &ExtCtxt,
                               sp: Span,
                               tts: &[ast::TokenTree],
                               name: &str)
                               -> Option<@str> {
    if tts.len() != 1 {
        cx.span_err(sp, format!("{} takes 1 argument.", name));
    } else {
        match tts[0] {
            ast::TTTok(_, token::LIT_STR(ident))
                | ast::TTTok(_, token::LIT_STR_RAW(ident, _)) => return Some(cx.str_of(ident)),
            _ => cx.span_err(sp, format!("{} requires a string.", name)),
        }
    }
    None
}

/// Extract comma-separated expressions from `tts`. If there is a
/// parsing error, emit a non-fatal error and return None.
pub fn get_exprs_from_tts(cx: &ExtCtxt,
                          sp: Span,
                          tts: &[ast::TokenTree]) -> Option<~[@ast::Expr]> {
    let mut p = parse::new_parser_from_tts(cx.parse_sess(),
                                           cx.cfg(),
                                           tts.to_owned());
    let mut es = ~[];
    while p.token != token::EOF {
        if es.len() != 0 && !p.eat(&token::COMMA) {
            cx.span_err(sp, "expected token: `,`");
            return None;
        }
        es.push(p.parse_expr());
    }
    Some(es)
}

// in order to have some notion of scoping for macros,
// we want to implement the notion of a transformation
// environment.

// This environment maps Names to SyntaxExtensions.

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

struct MapChainFrame {
    info: BlockInfo,
    map: HashMap<Name, SyntaxExtension>,
}

#[unsafe_destructor]
impl Drop for MapChainFrame {
    fn drop(&mut self) {
        // make sure that syntax extension dtors run before we drop the libs
        self.map.clear();
    }
}

// Only generic to make it easy to test
pub struct SyntaxEnv {
    priv chain: ~[MapChainFrame],
}

impl SyntaxEnv {
    pub fn new() -> SyntaxEnv {
        let mut map = SyntaxEnv { chain: ~[] };
        map.push_frame();
        map
    }

    pub fn push_frame(&mut self) {
        self.chain.push(MapChainFrame {
            info: BlockInfo::new(),
            map: HashMap::new(),
        });
    }

    pub fn pop_frame(&mut self) {
        assert!(self.chain.len() > 1, "too many pops on MapChain!");
        self.chain.pop();
    }

    fn find_escape_frame<'a>(&'a mut self) -> &'a mut MapChainFrame {
        for (i, frame) in self.chain.mut_iter().enumerate().rev() {
            if !frame.info.macros_escape || i == 0 {
                return frame
            }
        }
        unreachable!()
    }

    pub fn find<'a>(&'a self, k: &Name) -> Option<&'a SyntaxExtension> {
        for frame in self.chain.iter().rev() {
            match frame.map.find(k) {
                Some(v) => return Some(v),
                None => {}
            }
        }
        None
    }

    pub fn insert(&mut self, k: Name, v: SyntaxExtension) {
        self.find_escape_frame().map.insert(k, v);
    }

    pub fn insert_macro_crate(&mut self, lib: DynamicLibrary) {
        self.find_escape_frame().info.macro_crates.push(lib);
    }

    pub fn info<'a>(&'a mut self) -> &'a mut BlockInfo {
        &mut self.chain[self.chain.len()-1].info
    }
}
