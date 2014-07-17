// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use parse::parser;
use parse::token;
use parse::token::{InternedString, intern, str_to_ident};
use util::small_vector::SmallVector;
use ext::mtwt;

use std::collections::HashMap;
use std::gc::{Gc, GC};

// new-style macro! tt code:
//
//    MacResult, NormalTT, IdentTT
//
// also note that ast::Mac used to have a bunch of extraneous cases and
// is now probably a redundant AST node, can be merged with
// ast::MacInvocTT.

pub struct MacroDef {
    pub name: String,
    pub ext: SyntaxExtension
}

pub type ItemDecorator =
    fn(&mut ExtCtxt, Span, Gc<ast::MetaItem>, Gc<ast::Item>, |Gc<ast::Item>|);

pub type ItemModifier =
    fn(&mut ExtCtxt, Span, Gc<ast::MetaItem>, Gc<ast::Item>) -> Gc<ast::Item>;

pub struct BasicMacroExpander {
    pub expander: MacroExpanderFn,
    pub span: Option<Span>
}

/// Represents a thing that maps token trees to Macro Results
pub trait TTMacroExpander {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              token_tree: &[ast::TokenTree])
              -> Box<MacResult>;
}

pub type MacroExpanderFn =
    fn(ecx: &mut ExtCtxt, span: codemap::Span, token_tree: &[ast::TokenTree])
       -> Box<MacResult>;

impl TTMacroExpander for BasicMacroExpander {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              token_tree: &[ast::TokenTree])
              -> Box<MacResult> {
        (self.expander)(ecx, span, token_tree)
    }
}

pub struct BasicIdentMacroExpander {
    pub expander: IdentMacroExpanderFn,
    pub span: Option<Span>
}

pub trait IdentMacroExpander {
    fn expand(&self,
              cx: &mut ExtCtxt,
              sp: Span,
              ident: ast::Ident,
              token_tree: Vec<ast::TokenTree> )
              -> Box<MacResult>;
}

impl IdentMacroExpander for BasicIdentMacroExpander {
    fn expand(&self,
              cx: &mut ExtCtxt,
              sp: Span,
              ident: ast::Ident,
              token_tree: Vec<ast::TokenTree> )
              -> Box<MacResult> {
        (self.expander)(cx, sp, ident, token_tree)
    }
}

pub type IdentMacroExpanderFn =
    fn(&mut ExtCtxt, Span, ast::Ident, Vec<ast::TokenTree>) -> Box<MacResult>;

/// The result of a macro expansion. The return values of the various
/// methods are spliced into the AST at the callsite of the macro (or
/// just into the compiler's internal macro table, for `make_def`).
pub trait MacResult {
    /// Define a new macro.
    // this particular flavor should go away; the idea that a macro might
    // expand into either a macro definition or an expression, depending
    // on what the context wants, is kind of silly.
    fn make_def(&self) -> Option<MacroDef> {
        None
    }
    /// Create an expression.
    fn make_expr(&self) -> Option<Gc<ast::Expr>> {
        None
    }
    /// Create zero or more items.
    fn make_items(&self) -> Option<SmallVector<Gc<ast::Item>>> {
        None
    }

    /// Create zero or more methods.
    fn make_methods(&self) -> Option<SmallVector<Gc<ast::Method>>> {
        None
    }

    /// Create a pattern.
    fn make_pat(&self) -> Option<Gc<ast::Pat>> {
        None
    }

    /// Create a statement.
    ///
    /// By default this attempts to create an expression statement,
    /// returning None if that fails.
    fn make_stmt(&self) -> Option<Gc<ast::Stmt>> {
        self.make_expr()
            .map(|e| box(GC) codemap::respan(e.span, ast::StmtExpr(e, ast::DUMMY_NODE_ID)))
    }
}

/// A convenience type for macros that return a single expression.
pub struct MacExpr {
    e: Gc<ast::Expr>,
}
impl MacExpr {
    pub fn new(e: Gc<ast::Expr>) -> Box<MacResult> {
        box MacExpr { e: e } as Box<MacResult>
    }
}
impl MacResult for MacExpr {
    fn make_expr(&self) -> Option<Gc<ast::Expr>> {
        Some(self.e)
    }
}
/// A convenience type for macros that return a single pattern.
pub struct MacPat {
    p: Gc<ast::Pat>,
}
impl MacPat {
    pub fn new(p: Gc<ast::Pat>) -> Box<MacResult> {
        box MacPat { p: p } as Box<MacResult>
    }
}
impl MacResult for MacPat {
    fn make_pat(&self) -> Option<Gc<ast::Pat>> {
        Some(self.p)
    }
}
/// A convenience type for macros that return a single item.
pub struct MacItem {
    i: Gc<ast::Item>
}
impl MacItem {
    pub fn new(i: Gc<ast::Item>) -> Box<MacResult> {
        box MacItem { i: i } as Box<MacResult>
    }
}
impl MacResult for MacItem {
    fn make_items(&self) -> Option<SmallVector<Gc<ast::Item>>> {
        Some(SmallVector::one(self.i))
    }
    fn make_stmt(&self) -> Option<Gc<ast::Stmt>> {
        Some(box(GC) codemap::respan(
            self.i.span,
            ast::StmtDecl(
                box(GC) codemap::respan(self.i.span, ast::DeclItem(self.i)),
                ast::DUMMY_NODE_ID)))
    }
}

/// Fill-in macro expansion result, to allow compilation to continue
/// after hitting errors.
pub struct DummyResult {
    expr_only: bool,
    span: Span
}

impl DummyResult {
    /// Create a default MacResult that can be anything.
    ///
    /// Use this as a return value after hitting any errors and
    /// calling `span_err`.
    pub fn any(sp: Span) -> Box<MacResult> {
        box DummyResult { expr_only: false, span: sp } as Box<MacResult>
    }

    /// Create a default MacResult that can only be an expression.
    ///
    /// Use this for macros that must expand to an expression, so even
    /// if an error is encountered internally, the user will receive
    /// an error that they also used it in the wrong place.
    pub fn expr(sp: Span) -> Box<MacResult> {
        box DummyResult { expr_only: true, span: sp } as Box<MacResult>
    }

    /// A plain dummy expression.
    pub fn raw_expr(sp: Span) -> Gc<ast::Expr> {
        box(GC) ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node: ast::ExprLit(box(GC) codemap::respan(sp, ast::LitNil)),
            span: sp,
        }
    }

    /// A plain dummy pattern.
    pub fn raw_pat(sp: Span) -> Gc<ast::Pat> {
        box(GC) ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: ast::PatWild,
            span: sp,
        }
    }

}

impl MacResult for DummyResult {
    fn make_expr(&self) -> Option<Gc<ast::Expr>> {
        Some(DummyResult::raw_expr(self.span))
    }
    fn make_pat(&self) -> Option<Gc<ast::Pat>> {
        Some(DummyResult::raw_pat(self.span))
    }
    fn make_items(&self) -> Option<SmallVector<Gc<ast::Item>>> {
        // this code needs a comment... why not always just return the Some() ?
        if self.expr_only {
            None
        } else {
            Some(SmallVector::zero())
        }
    }
    fn make_methods(&self) -> Option<SmallVector<Gc<ast::Method>>> {
        if self.expr_only {
            None
        } else {
            Some(SmallVector::zero())
        }
    }
    fn make_stmt(&self) -> Option<Gc<ast::Stmt>> {
        Some(box(GC) codemap::respan(self.span,
                              ast::StmtExpr(DummyResult::raw_expr(self.span),
                                            ast::DUMMY_NODE_ID)))
    }
}

/// An enum representing the different kinds of syntax extensions.
pub enum SyntaxExtension {
    /// A syntax extension that is attached to an item and creates new items
    /// based upon it.
    ///
    /// `#[deriving(...)]` is an `ItemDecorator`.
    ItemDecorator(ItemDecorator),

    /// A syntax extension that is attached to an item and modifies it
    /// in-place.
    ItemModifier(ItemModifier),

    /// A normal, function-like syntax extension.
    ///
    /// `bytes!` is a `NormalTT`.
    NormalTT(Box<TTMacroExpander + 'static>, Option<Span>),

    /// A function-like syntax extension that has an extra ident before
    /// the block.
    ///
    IdentTT(Box<IdentMacroExpander + 'static>, Option<Span>),

    /// An ident macro that has two properties:
    /// - it adds a macro definition to the environment, and
    /// - the definition it adds doesn't introduce any new
    ///   identifiers.
    ///
    /// `macro_rules!` is a LetSyntaxTT
    LetSyntaxTT(Box<IdentMacroExpander + 'static>, Option<Span>),
}

pub type NamedSyntaxExtension = (Name, SyntaxExtension);

pub struct BlockInfo {
    /// Should macros escape from this scope?
    pub macros_escape: bool,
    /// What are the pending renames?
    pub pending_renames: mtwt::RenameList,
}

impl BlockInfo {
    pub fn new() -> BlockInfo {
        BlockInfo {
            macros_escape: false,
            pending_renames: Vec::new(),
        }
    }
}

/// The base map of methods for expanding syntax extension
/// AST nodes into full ASTs
pub fn syntax_expander_table() -> SyntaxEnv {
    // utility function to simplify creating NormalTT syntax extensions
    fn builtin_normal_expander(f: MacroExpanderFn) -> SyntaxExtension {
        NormalTT(box BasicMacroExpander {
                expander: f,
                span: None,
            },
            None)
    }

    let mut syntax_expanders = SyntaxEnv::new();
    syntax_expanders.insert(intern("macro_rules"),
                            LetSyntaxTT(box BasicIdentMacroExpander {
                                expander: ext::tt::macro_rules::add_new_extension,
                                span: None,
                            },
                            None));
    syntax_expanders.insert(intern("fmt"),
                            builtin_normal_expander(
                                ext::fmt::expand_syntax_ext));
    syntax_expanders.insert(intern("format_args"),
                            builtin_normal_expander(
                                ext::format::expand_format_args));
    syntax_expanders.insert(intern("format_args_method"),
                            builtin_normal_expander(
                                ext::format::expand_format_args_method));
    syntax_expanders.insert(intern("env"),
                            builtin_normal_expander(
                                    ext::env::expand_env));
    syntax_expanders.insert(intern("option_env"),
                            builtin_normal_expander(
                                    ext::env::expand_option_env));
    syntax_expanders.insert(intern("bytes"),
                            builtin_normal_expander(
                                    ext::bytes::expand_syntax_ext));
    syntax_expanders.insert(intern("concat_idents"),
                            builtin_normal_expander(
                                    ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert(intern("concat"),
                            builtin_normal_expander(
                                    ext::concat::expand_syntax_ext));
    syntax_expanders.insert(intern("log_syntax"),
                            builtin_normal_expander(
                                    ext::log_syntax::expand_syntax_ext));
    syntax_expanders.insert(intern("deriving"),
                            ItemDecorator(ext::deriving::expand_meta_deriving));

    // Quasi-quoting expanders
    syntax_expanders.insert(intern("quote_tokens"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_tokens));
    syntax_expanders.insert(intern("quote_expr"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_expr));
    syntax_expanders.insert(intern("quote_ty"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_ty));
    syntax_expanders.insert(intern("quote_method"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_method));
    syntax_expanders.insert(intern("quote_item"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_item));
    syntax_expanders.insert(intern("quote_pat"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_pat));
    syntax_expanders.insert(intern("quote_stmt"),
                       builtin_normal_expander(
                            ext::quote::expand_quote_stmt));

    syntax_expanders.insert(intern("line"),
                            builtin_normal_expander(
                                    ext::source_util::expand_line));
    syntax_expanders.insert(intern("col"),
                            builtin_normal_expander(
                                    ext::source_util::expand_col));
    syntax_expanders.insert(intern("file"),
                            builtin_normal_expander(
                                    ext::source_util::expand_file));
    syntax_expanders.insert(intern("stringify"),
                            builtin_normal_expander(
                                    ext::source_util::expand_stringify));
    syntax_expanders.insert(intern("include"),
                            builtin_normal_expander(
                                    ext::source_util::expand_include));
    syntax_expanders.insert(intern("include_str"),
                            builtin_normal_expander(
                                    ext::source_util::expand_include_str));
    syntax_expanders.insert(intern("include_bin"),
                            builtin_normal_expander(
                                    ext::source_util::expand_include_bin));
    syntax_expanders.insert(intern("module_path"),
                            builtin_normal_expander(
                                    ext::source_util::expand_mod));
    syntax_expanders.insert(intern("asm"),
                            builtin_normal_expander(
                                    ext::asm::expand_asm));
    syntax_expanders.insert(intern("cfg"),
                            builtin_normal_expander(
                                    ext::cfg::expand_cfg));
    syntax_expanders.insert(intern("trace_macros"),
                            builtin_normal_expander(
                                    ext::trace_macros::expand_trace_macros));
    syntax_expanders
}

/// One of these is made during expansion and incrementally updated as we go;
/// when a macro expansion occurs, the resulting nodes have the backtrace()
/// -> expn_info of their expansion context stored into their span.
pub struct ExtCtxt<'a> {
    pub parse_sess: &'a parse::ParseSess,
    pub cfg: ast::CrateConfig,
    pub backtrace: Option<Gc<ExpnInfo>>,
    pub ecfg: expand::ExpansionConfig,

    pub mod_path: Vec<ast::Ident> ,
    pub trace_mac: bool,
    pub exported_macros: Vec<Gc<ast::Item>>
}

impl<'a> ExtCtxt<'a> {
    pub fn new<'a>(parse_sess: &'a parse::ParseSess, cfg: ast::CrateConfig,
                   ecfg: expand::ExpansionConfig) -> ExtCtxt<'a> {
        ExtCtxt {
            parse_sess: parse_sess,
            cfg: cfg,
            backtrace: None,
            mod_path: Vec::new(),
            ecfg: ecfg,
            trace_mac: false,
            exported_macros: Vec::new(),
        }
    }

    pub fn expand_expr(&mut self, mut e: Gc<ast::Expr>) -> Gc<ast::Expr> {
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

    pub fn new_parser_from_tts(&self, tts: &[ast::TokenTree])
        -> parser::Parser<'a> {
        parse::tts_to_parser(self.parse_sess, Vec::from_slice(tts), self.cfg())
    }

    pub fn codemap(&self) -> &'a CodeMap { &self.parse_sess.span_diagnostic.cm }
    pub fn parse_sess(&self) -> &'a parse::ParseSess { self.parse_sess }
    pub fn cfg(&self) -> ast::CrateConfig { self.cfg.clone() }
    pub fn call_site(&self) -> Span {
        match self.backtrace {
            Some(expn_info) => expn_info.call_site,
            None => self.bug("missing top span")
        }
    }
    pub fn print_backtrace(&self) { }
    pub fn backtrace(&self) -> Option<Gc<ExpnInfo>> { self.backtrace }
    pub fn mod_push(&mut self, i: ast::Ident) { self.mod_path.push(i); }
    pub fn mod_pop(&mut self) { self.mod_path.pop().unwrap(); }
    pub fn mod_path(&self) -> Vec<ast::Ident> {
        let mut v = Vec::new();
        v.push(token::str_to_ident(self.ecfg.crate_name.as_slice()));
        v.extend(self.mod_path.iter().map(|a| *a));
        return v;
    }
    pub fn bt_push(&mut self, ei: codemap::ExpnInfo) {
        match ei {
            ExpnInfo {call_site: cs, callee: ref callee} => {
                self.backtrace =
                    Some(box(GC) ExpnInfo {
                        call_site: Span {lo: cs.lo, hi: cs.hi,
                                         expn_info: self.backtrace.clone()},
                        callee: (*callee).clone()
                    });
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
    /// `span_err` should be strongly preferred where-ever possible:
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
    pub fn ident_of(&self, st: &str) -> ast::Ident {
        str_to_ident(st)
    }
    pub fn name_of(&self, st: &str) -> ast::Name {
        token::intern(st)
    }
}

/// Extract a string literal from the macro expanded version of `expr`,
/// emitting `err_msg` if `expr` is not a string literal. This does not stop
/// compilation on error, merely emits a non-fatal error and returns None.
pub fn expr_to_string(cx: &mut ExtCtxt, expr: Gc<ast::Expr>, err_msg: &str)
                   -> Option<(InternedString, ast::StrStyle)> {
    // we want to be able to handle e.g. concat("foo", "bar")
    let expr = cx.expand_expr(expr);
    match expr.node {
        ast::ExprLit(l) => match l.node {
            ast::LitStr(ref s, style) => return Some(((*s).clone(), style)),
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
pub fn check_zero_tts(cx: &ExtCtxt,
                      sp: Span,
                      tts: &[ast::TokenTree],
                      name: &str) {
    if tts.len() != 0 {
        cx.span_err(sp, format!("{} takes no arguments", name).as_slice());
    }
}

/// Extract the string literal from the first token of `tts`. If this
/// is not a string literal, emit an error and return None.
pub fn get_single_str_from_tts(cx: &ExtCtxt,
                               sp: Span,
                               tts: &[ast::TokenTree],
                               name: &str)
                               -> Option<String> {
    if tts.len() != 1 {
        cx.span_err(sp, format!("{} takes 1 argument.", name).as_slice());
    } else {
        match tts[0] {
            ast::TTTok(_, token::LIT_STR(ident)) => return Some(parse::str_lit(ident.as_str())),
            ast::TTTok(_, token::LIT_STR_RAW(ident, _)) => {
                return Some(parse::raw_str_lit(ident.as_str()))
            }
            _ => {
                cx.span_err(sp,
                            format!("{} requires a string.", name).as_slice())
            }
        }
    }
    None
}

/// Extract comma-separated expressions from `tts`. If there is a
/// parsing error, emit a non-fatal error and return None.
pub fn get_exprs_from_tts(cx: &mut ExtCtxt,
                          sp: Span,
                          tts: &[ast::TokenTree]) -> Option<Vec<Gc<ast::Expr>>> {
    let mut p = cx.new_parser_from_tts(tts);
    let mut es = Vec::new();
    while p.token != token::EOF {
        es.push(cx.expand_expr(p.parse_expr()));
        if p.eat(&token::COMMA) {
            continue;
        }
        if p.token != token::EOF {
            cx.span_err(sp, "expected token: `,`");
            return None;
        }
    }
    Some(es)
}

/// In order to have some notion of scoping for macros,
/// we want to implement the notion of a transformation
/// environment.

/// This environment maps Names to SyntaxExtensions.

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

pub struct SyntaxEnv {
    chain: Vec<MapChainFrame> ,
}

impl SyntaxEnv {
    pub fn new() -> SyntaxEnv {
        let mut map = SyntaxEnv { chain: Vec::new() };
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

    pub fn info<'a>(&'a mut self) -> &'a mut BlockInfo {
        let last_chain_index = self.chain.len() - 1;
        &mut self.chain.get_mut(last_chain_index).info
    }
}
