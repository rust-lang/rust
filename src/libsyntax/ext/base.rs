// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::SyntaxExtension::*;

use ast;
use ast::Name;
use codemap;
use codemap::{CodeMap, Span, ExpnId, ExpnInfo, NO_EXPANSION};
use ext;
use ext::expand;
use ext::tt::macro_rules;
use feature_gate::GatedCfg;
use parse;
use parse::parser;
use parse::token;
use parse::token::{InternedString, intern, str_to_ident};
use ptr::P;
use util::small_vector::SmallVector;
use ext::mtwt;
use fold::Folder;

use std::collections::HashMap;
use std::rc::Rc;
use std::default::Default;


#[derive(Debug,Clone)]
pub enum Annotatable {
    Item(P<ast::Item>),
    TraitItem(P<ast::TraitItem>),
    ImplItem(P<ast::ImplItem>),
}

impl Annotatable {
    pub fn attrs(&self) -> &[ast::Attribute] {
        match *self {
            Annotatable::Item(ref i) => &i.attrs,
            Annotatable::TraitItem(ref ti) => &ti.attrs,
            Annotatable::ImplItem(ref ii) => &ii.attrs,
        }
    }

    pub fn fold_attrs(self, attrs: Vec<ast::Attribute>) -> Annotatable {
        match self {
            Annotatable::Item(i) => Annotatable::Item(i.map(|i| ast::Item {
                attrs: attrs,
                ..i
            })),
            Annotatable::TraitItem(i) => Annotatable::TraitItem(i.map(|ti| {
                ast::TraitItem { attrs: attrs, ..ti }
            })),
            Annotatable::ImplItem(i) => Annotatable::ImplItem(i.map(|ii| {
                ast::ImplItem { attrs: attrs, ..ii }
            })),
        }
    }

    pub fn expect_item(self) -> P<ast::Item> {
        match self {
            Annotatable::Item(i) => i,
            _ => panic!("expected Item")
        }
    }

    pub fn map_item_or<F, G>(self, mut f: F, mut or: G) -> Annotatable
        where F: FnMut(P<ast::Item>) -> P<ast::Item>,
              G: FnMut(Annotatable) -> Annotatable
    {
        match self {
            Annotatable::Item(i) => Annotatable::Item(f(i)),
            _ => or(self)
        }
    }

    pub fn expect_trait_item(self) -> P<ast::TraitItem> {
        match self {
            Annotatable::TraitItem(i) => i,
            _ => panic!("expected Item")
        }
    }

    pub fn expect_impl_item(self) -> P<ast::ImplItem> {
        match self {
            Annotatable::ImplItem(i) => i,
            _ => panic!("expected Item")
        }
    }
}

// A more flexible ItemDecorator.
pub trait MultiItemDecorator {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              sp: Span,
              meta_item: &ast::MetaItem,
              item: &Annotatable,
              push: &mut FnMut(Annotatable));
}

impl<F> MultiItemDecorator for F
    where F : Fn(&mut ExtCtxt, Span, &ast::MetaItem, &Annotatable, &mut FnMut(Annotatable))
{
    fn expand(&self,
              ecx: &mut ExtCtxt,
              sp: Span,
              meta_item: &ast::MetaItem,
              item: &Annotatable,
              push: &mut FnMut(Annotatable)) {
        (*self)(ecx, sp, meta_item, item, push)
    }
}

// A more flexible ItemModifier (ItemModifier should go away, eventually, FIXME).
// meta_item is the annotation, item is the item being modified, parent_item
// is the impl or trait item is declared in if item is part of such a thing.
// FIXME Decorators should follow the same pattern too.
pub trait MultiItemModifier {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Annotatable;
}

impl<F> MultiItemModifier for F
    where F: Fn(&mut ExtCtxt,
                Span,
                &ast::MetaItem,
                Annotatable) -> Annotatable
{
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Annotatable {
        (*self)(ecx, span, meta_item, item)
    }
}

/// Represents a thing that maps token trees to Macro Results
pub trait TTMacroExpander {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   token_tree: &[ast::TokenTree])
                   -> Box<MacResult+'cx>;
}

pub type MacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt, Span, &[ast::TokenTree]) -> Box<MacResult+'cx>;

impl<F> TTMacroExpander for F
    where F : for<'cx> Fn(&'cx mut ExtCtxt, Span, &[ast::TokenTree]) -> Box<MacResult+'cx>
{
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   token_tree: &[ast::TokenTree])
                   -> Box<MacResult+'cx> {
        (*self)(ecx, span, token_tree)
    }
}

pub trait IdentMacroExpander {
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   ident: ast::Ident,
                   token_tree: Vec<ast::TokenTree> )
                   -> Box<MacResult+'cx>;
}

pub type IdentMacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt, Span, ast::Ident, Vec<ast::TokenTree>) -> Box<MacResult+'cx>;

impl<F> IdentMacroExpander for F
    where F : for<'cx> Fn(&'cx mut ExtCtxt, Span, ast::Ident,
                          Vec<ast::TokenTree>) -> Box<MacResult+'cx>
{
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   ident: ast::Ident,
                   token_tree: Vec<ast::TokenTree> )
                   -> Box<MacResult+'cx>
    {
        (*self)(cx, sp, ident, token_tree)
    }
}

// Use a macro because forwarding to a simple function has type system issues
macro_rules! make_stmts_default {
    ($me:expr) => {
        $me.make_expr().map(|e| {
            SmallVector::one(P(codemap::respan(
                e.span, ast::StmtExpr(e, ast::DUMMY_NODE_ID))))
        })
    }
}

/// The result of a macro expansion. The return values of the various
/// methods are spliced into the AST at the callsite of the macro.
pub trait MacResult {
    /// Create an expression.
    fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
        None
    }
    /// Create zero or more items.
    fn make_items(self: Box<Self>) -> Option<SmallVector<P<ast::Item>>> {
        None
    }

    /// Create zero or more impl items.
    fn make_impl_items(self: Box<Self>) -> Option<SmallVector<P<ast::ImplItem>>> {
        None
    }

    /// Create a pattern.
    fn make_pat(self: Box<Self>) -> Option<P<ast::Pat>> {
        None
    }

    /// Create zero or more statements.
    ///
    /// By default this attempts to create an expression statement,
    /// returning None if that fails.
    fn make_stmts(self: Box<Self>) -> Option<SmallVector<P<ast::Stmt>>> {
        make_stmts_default!(self)
    }

    fn make_ty(self: Box<Self>) -> Option<P<ast::Ty>> {
        None
    }
}

macro_rules! make_MacEager {
    ( $( $fld:ident: $t:ty, )* ) => {
        /// `MacResult` implementation for the common case where you've already
        /// built each form of AST that you might return.
        #[derive(Default)]
        pub struct MacEager {
            $(
                pub $fld: Option<$t>,
            )*
        }

        impl MacEager {
            $(
                pub fn $fld(v: $t) -> Box<MacResult> {
                    Box::new(MacEager {
                        $fld: Some(v),
                        ..Default::default()
                    })
                }
            )*
        }
    }
}

make_MacEager! {
    expr: P<ast::Expr>,
    pat: P<ast::Pat>,
    items: SmallVector<P<ast::Item>>,
    impl_items: SmallVector<P<ast::ImplItem>>,
    stmts: SmallVector<P<ast::Stmt>>,
    ty: P<ast::Ty>,
}

impl MacResult for MacEager {
    fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
        self.expr
    }

    fn make_items(self: Box<Self>) -> Option<SmallVector<P<ast::Item>>> {
        self.items
    }

    fn make_impl_items(self: Box<Self>) -> Option<SmallVector<P<ast::ImplItem>>> {
        self.impl_items
    }

    fn make_stmts(self: Box<Self>) -> Option<SmallVector<P<ast::Stmt>>> {
        match self.stmts.as_ref().map_or(0, |s| s.len()) {
            0 => make_stmts_default!(self),
            _ => self.stmts,
        }
    }

    fn make_pat(self: Box<Self>) -> Option<P<ast::Pat>> {
        if let Some(p) = self.pat {
            return Some(p);
        }
        if let Some(e) = self.expr {
            if let ast::ExprLit(_) = e.node {
                return Some(P(ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    span: e.span,
                    node: ast::PatLit(e),
                }));
            }
        }
        None
    }

    fn make_ty(self: Box<Self>) -> Option<P<ast::Ty>> {
        self.ty
    }
}

/// Fill-in macro expansion result, to allow compilation to continue
/// after hitting errors.
#[derive(Copy, Clone)]
pub struct DummyResult {
    expr_only: bool,
    span: Span
}

impl DummyResult {
    /// Create a default MacResult that can be anything.
    ///
    /// Use this as a return value after hitting any errors and
    /// calling `span_err`.
    pub fn any(sp: Span) -> Box<MacResult+'static> {
        Box::new(DummyResult { expr_only: false, span: sp })
    }

    /// Create a default MacResult that can only be an expression.
    ///
    /// Use this for macros that must expand to an expression, so even
    /// if an error is encountered internally, the user will receive
    /// an error that they also used it in the wrong place.
    pub fn expr(sp: Span) -> Box<MacResult+'static> {
        Box::new(DummyResult { expr_only: true, span: sp })
    }

    /// A plain dummy expression.
    pub fn raw_expr(sp: Span) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node: ast::ExprLit(P(codemap::respan(sp, ast::LitBool(false)))),
            span: sp,
        })
    }

    /// A plain dummy pattern.
    pub fn raw_pat(sp: Span) -> ast::Pat {
        ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: ast::PatWild,
            span: sp,
        }
    }

    pub fn raw_ty(sp: Span) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            node: ast::TyInfer,
            span: sp
        })
    }
}

impl MacResult for DummyResult {
    fn make_expr(self: Box<DummyResult>) -> Option<P<ast::Expr>> {
        Some(DummyResult::raw_expr(self.span))
    }

    fn make_pat(self: Box<DummyResult>) -> Option<P<ast::Pat>> {
        Some(P(DummyResult::raw_pat(self.span)))
    }

    fn make_items(self: Box<DummyResult>) -> Option<SmallVector<P<ast::Item>>> {
        // this code needs a comment... why not always just return the Some() ?
        if self.expr_only {
            None
        } else {
            Some(SmallVector::zero())
        }
    }

    fn make_impl_items(self: Box<DummyResult>) -> Option<SmallVector<P<ast::ImplItem>>> {
        if self.expr_only {
            None
        } else {
            Some(SmallVector::zero())
        }
    }

    fn make_stmts(self: Box<DummyResult>) -> Option<SmallVector<P<ast::Stmt>>> {
        Some(SmallVector::one(P(
            codemap::respan(self.span,
                            ast::StmtExpr(DummyResult::raw_expr(self.span),
                                          ast::DUMMY_NODE_ID)))))
    }
}

/// An enum representing the different kinds of syntax extensions.
pub enum SyntaxExtension {
    /// A syntax extension that is attached to an item and creates new items
    /// based upon it.
    ///
    /// `#[derive(...)]` is a `MultiItemDecorator`.
    MultiDecorator(Box<MultiItemDecorator + 'static>),

    /// A syntax extension that is attached to an item and modifies it
    /// in-place. More flexible version than Modifier.
    MultiModifier(Box<MultiItemModifier + 'static>),

    /// A normal, function-like syntax extension.
    ///
    /// `bytes!` is a `NormalTT`.
    ///
    /// The `bool` dictates whether the contents of the macro can
    /// directly use `#[unstable]` things (true == yes).
    NormalTT(Box<TTMacroExpander + 'static>, Option<Span>, bool),

    /// A function-like syntax extension that has an extra ident before
    /// the block.
    ///
    IdentTT(Box<IdentMacroExpander + 'static>, Option<Span>, bool),

    /// Represents `macro_rules!` itself.
    MacroRulesTT,
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
fn initial_syntax_expander_table<'feat>(ecfg: &expand::ExpansionConfig<'feat>)
                                        -> SyntaxEnv {
    // utility function to simplify creating NormalTT syntax extensions
    fn builtin_normal_expander(f: MacroExpanderFn) -> SyntaxExtension {
        NormalTT(Box::new(f), None, false)
    }

    let mut syntax_expanders = SyntaxEnv::new();
    syntax_expanders.insert(intern("macro_rules"), MacroRulesTT);
    syntax_expanders.insert(intern("format_args"),
                            // format_args uses `unstable` things internally.
                            NormalTT(Box::new(ext::format::expand_format_args), None, true));
    syntax_expanders.insert(intern("env"),
                            builtin_normal_expander(
                                    ext::env::expand_env));
    syntax_expanders.insert(intern("option_env"),
                            builtin_normal_expander(
                                    ext::env::expand_option_env));
    syntax_expanders.insert(intern("concat_idents"),
                            builtin_normal_expander(
                                    ext::concat_idents::expand_syntax_ext));
    syntax_expanders.insert(intern("concat"),
                            builtin_normal_expander(
                                    ext::concat::expand_syntax_ext));
    syntax_expanders.insert(intern("log_syntax"),
                            builtin_normal_expander(
                                    ext::log_syntax::expand_syntax_ext));

    ext::deriving::register_all(&mut syntax_expanders);

    if ecfg.enable_quotes() {
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
        syntax_expanders.insert(intern("quote_item"),
                           builtin_normal_expander(
                                ext::quote::expand_quote_item));
        syntax_expanders.insert(intern("quote_pat"),
                           builtin_normal_expander(
                                ext::quote::expand_quote_pat));
        syntax_expanders.insert(intern("quote_arm"),
                           builtin_normal_expander(
                                ext::quote::expand_quote_arm));
        syntax_expanders.insert(intern("quote_stmt"),
                           builtin_normal_expander(
                                ext::quote::expand_quote_stmt));
        syntax_expanders.insert(intern("quote_matcher"),
                           builtin_normal_expander(
                                ext::quote::expand_quote_matcher));
        syntax_expanders.insert(intern("quote_attr"),
                           builtin_normal_expander(
                                ext::quote::expand_quote_attr));
    }

    syntax_expanders.insert(intern("line"),
                            builtin_normal_expander(
                                    ext::source_util::expand_line));
    syntax_expanders.insert(intern("column"),
                            builtin_normal_expander(
                                    ext::source_util::expand_column));
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
    syntax_expanders.insert(intern("include_bytes"),
                            builtin_normal_expander(
                                    ext::source_util::expand_include_bytes));
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
    pub backtrace: ExpnId,
    pub ecfg: expand::ExpansionConfig<'a>,
    pub crate_root: Option<&'static str>,
    pub feature_gated_cfgs: &'a mut Vec<GatedCfg>,

    pub mod_path: Vec<ast::Ident> ,
    pub exported_macros: Vec<ast::MacroDef>,

    pub syntax_env: SyntaxEnv,
    pub recursion_count: usize,
}

impl<'a> ExtCtxt<'a> {
    pub fn new(parse_sess: &'a parse::ParseSess, cfg: ast::CrateConfig,
               ecfg: expand::ExpansionConfig<'a>,
               feature_gated_cfgs: &'a mut Vec<GatedCfg>) -> ExtCtxt<'a> {
        let env = initial_syntax_expander_table(&ecfg);
        ExtCtxt {
            parse_sess: parse_sess,
            cfg: cfg,
            backtrace: NO_EXPANSION,
            mod_path: Vec::new(),
            ecfg: ecfg,
            crate_root: None,
            feature_gated_cfgs: feature_gated_cfgs,
            exported_macros: Vec::new(),
            syntax_env: env,
            recursion_count: 0,
        }
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[deprecated(since = "1.0.0",
                 reason = "Replaced with `expander().fold_expr()`")]
    pub fn expand_expr(&mut self, e: P<ast::Expr>) -> P<ast::Expr> {
        self.expander().fold_expr(e)
    }

    /// Returns a `Folder` for deeply expanding all macros in an AST node.
    pub fn expander<'b>(&'b mut self) -> expand::MacroExpander<'b, 'a> {
        expand::MacroExpander::new(self)
    }

    pub fn new_parser_from_tts(&self, tts: &[ast::TokenTree])
        -> parser::Parser<'a> {
        parse::tts_to_parser(self.parse_sess, tts.to_vec(), self.cfg())
    }

    pub fn codemap(&self) -> &'a CodeMap { self.parse_sess.codemap() }
    pub fn parse_sess(&self) -> &'a parse::ParseSess { self.parse_sess }
    pub fn cfg(&self) -> ast::CrateConfig { self.cfg.clone() }
    pub fn call_site(&self) -> Span {
        self.codemap().with_expn_info(self.backtrace, |ei| match ei {
            Some(expn_info) => expn_info.call_site,
            None => self.bug("missing top span")
        })
    }
    pub fn backtrace(&self) -> ExpnId { self.backtrace }

    /// Original span that caused the current exapnsion to happen.
    pub fn original_span(&self) -> Span {
        let mut expn_id = self.backtrace;
        let mut call_site = None;
        loop {
            match self.codemap().with_expn_info(expn_id, |ei| ei.map(|ei| ei.call_site)) {
                None => break,
                Some(cs) => {
                    call_site = Some(cs);
                    expn_id = cs.expn_id;
                }
            }
        }
        call_site.expect("missing expansion backtrace")
    }

    /// Returns span for the macro which originally caused the current expansion to happen.
    ///
    /// Stops backtracing at include! boundary.
    pub fn expansion_cause(&self) -> Span {
        let mut expn_id = self.backtrace;
        let mut last_macro = None;
        loop {
            if self.codemap().with_expn_info(expn_id, |info| {
                info.map_or(None, |i| {
                    if i.callee.name().as_str() == "include" {
                        // Stop going up the backtrace once include! is encountered
                        return None;
                    }
                    expn_id = i.call_site.expn_id;
                    last_macro = Some(i.call_site);
                    return Some(());
                })
            }).is_none() {
                break
            }
        }
        last_macro.expect("missing expansion backtrace")
    }

    pub fn mod_push(&mut self, i: ast::Ident) { self.mod_path.push(i); }
    pub fn mod_pop(&mut self) { self.mod_path.pop().unwrap(); }
    pub fn mod_path(&self) -> Vec<ast::Ident> {
        let mut v = Vec::new();
        v.push(token::str_to_ident(&self.ecfg.crate_name));
        v.extend(self.mod_path.iter().cloned());
        return v;
    }
    pub fn bt_push(&mut self, ei: ExpnInfo) {
        self.recursion_count += 1;
        if self.recursion_count > self.ecfg.recursion_limit {
            panic!(self.span_fatal(ei.call_site,
                            &format!("recursion limit reached while expanding the macro `{}`",
                                    ei.callee.name())));
        }

        let mut call_site = ei.call_site;
        call_site.expn_id = self.backtrace;
        self.backtrace = self.codemap().record_expansion(ExpnInfo {
            call_site: call_site,
            callee: ei.callee
        });
    }
    pub fn bt_pop(&mut self) {
        match self.backtrace {
            NO_EXPANSION => self.bug("tried to pop without a push"),
            expn_id => {
                self.recursion_count -= 1;
                self.backtrace = self.codemap().with_expn_info(expn_id, |expn_info| {
                    expn_info.map_or(NO_EXPANSION, |ei| ei.call_site.expn_id)
                });
            }
        }
    }

    pub fn insert_macro(&mut self, def: ast::MacroDef) {
        if def.export {
            self.exported_macros.push(def.clone());
        }
        if def.use_locally {
            let ext = macro_rules::compile(self, &def);
            self.syntax_env.insert(def.ident.name, ext);
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
        panic!(self.parse_sess.span_diagnostic.span_fatal(sp, msg));
    }

    /// Emit `msg` attached to `sp`, without immediately stopping
    /// compilation.
    ///
    /// Compilation will be stopped in the near future (at the end of
    /// the macro expansion phase).
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.parse_sess.span_diagnostic.span_err(sp, msg);
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.parse_sess.span_diagnostic.span_warn(sp, msg);
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.span_unimpl(sp, msg);
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.span_bug(sp, msg);
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.parse_sess.span_diagnostic.span_note(sp, msg);
    }
    pub fn span_help(&self, sp: Span, msg: &str) {
        self.parse_sess.span_diagnostic.span_help(sp, msg);
    }
    pub fn fileline_help(&self, sp: Span, msg: &str) {
        self.parse_sess.span_diagnostic.fileline_help(sp, msg);
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.handler().bug(msg);
    }
    pub fn trace_macros(&self) -> bool {
        self.ecfg.trace_mac
    }
    pub fn set_trace_macros(&mut self, x: bool) {
        self.ecfg.trace_mac = x
    }
    pub fn ident_of(&self, st: &str) -> ast::Ident {
        str_to_ident(st)
    }
    pub fn std_path(&self, components: &[&str]) -> Vec<ast::Ident> {
        let mut v = Vec::new();
        if let Some(s) = self.crate_root {
            v.push(self.ident_of(s));
        }
        v.extend(components.iter().map(|s| self.ident_of(s)));
        return v
    }
    pub fn name_of(&self, st: &str) -> ast::Name {
        token::intern(st)
    }
}

/// Extract a string literal from the macro expanded version of `expr`,
/// emitting `err_msg` if `expr` is not a string literal. This does not stop
/// compilation on error, merely emits a non-fatal error and returns None.
pub fn expr_to_string(cx: &mut ExtCtxt, expr: P<ast::Expr>, err_msg: &str)
                      -> Option<(InternedString, ast::StrStyle)> {
    // we want to be able to handle e.g. concat("foo", "bar")
    let expr = cx.expander().fold_expr(expr);
    match expr.node {
        ast::ExprLit(ref l) => match l.node {
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
    if !tts.is_empty() {
        cx.span_err(sp, &format!("{} takes no arguments", name));
    }
}

/// Extract the string literal from the first token of `tts`. If this
/// is not a string literal, emit an error and return None.
pub fn get_single_str_from_tts(cx: &mut ExtCtxt,
                               sp: Span,
                               tts: &[ast::TokenTree],
                               name: &str)
                               -> Option<String> {
    let mut p = cx.new_parser_from_tts(tts);
    if p.token == token::Eof {
        cx.span_err(sp, &format!("{} takes 1 argument", name));
        return None
    }
    let ret = cx.expander().fold_expr(p.parse_expr());
    if p.token != token::Eof {
        cx.span_err(sp, &format!("{} takes 1 argument", name));
    }
    expr_to_string(cx, ret, "argument must be a string literal").map(|(s, _)| {
        s.to_string()
    })
}

/// Extract comma-separated expressions from `tts`. If there is a
/// parsing error, emit a non-fatal error and return None.
pub fn get_exprs_from_tts(cx: &mut ExtCtxt,
                          sp: Span,
                          tts: &[ast::TokenTree]) -> Option<Vec<P<ast::Expr>>> {
    let mut p = cx.new_parser_from_tts(tts);
    let mut es = Vec::new();
    while p.token != token::Eof {
        es.push(cx.expander().fold_expr(p.parse_expr()));
        if panictry!(p.eat(&token::Comma)){
            continue;
        }
        if p.token != token::Eof {
            cx.span_err(sp, "expected token: `,`");
            return None;
        }
    }
    Some(es)
}

/// In order to have some notion of scoping for macros,
/// we want to implement the notion of a transformation
/// environment.
///
/// This environment maps Names to SyntaxExtensions.
pub struct SyntaxEnv {
    chain: Vec<MapChainFrame> ,
}

// impl question: how to implement it? Initially, the
// env will contain only macros, so it might be painful
// to add an empty frame for every context. Let's just
// get it working, first....

// NB! the mutability of the underlying maps means that
// if expansion is out-of-order, a deeper scope may be
// able to refer to a macro that was added to an enclosing
// scope lexically later than the deeper scope.

struct MapChainFrame {
    info: BlockInfo,
    map: HashMap<Name, Rc<SyntaxExtension>>,
}

impl SyntaxEnv {
    fn new() -> SyntaxEnv {
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
        for (i, frame) in self.chain.iter_mut().enumerate().rev() {
            if !frame.info.macros_escape || i == 0 {
                return frame
            }
        }
        unreachable!()
    }

    pub fn find(&self, k: Name) -> Option<Rc<SyntaxExtension>> {
        for frame in self.chain.iter().rev() {
            match frame.map.get(&k) {
                Some(v) => return Some(v.clone()),
                None => {}
            }
        }
        None
    }

    pub fn insert(&mut self, k: Name, v: SyntaxExtension) {
        self.find_escape_frame().map.insert(k, Rc::new(v));
    }

    pub fn info<'a>(&'a mut self) -> &'a mut BlockInfo {
        let last_chain_index = self.chain.len() - 1;
        &mut self.chain[last_chain_index].info
    }
}
