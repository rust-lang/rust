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

use ast::{self, Attribute, Name, PatKind, MetaItem};
use attr::HasAttrs;
use source_map::{self, SourceMap, Spanned, respan};
use syntax_pos::{Span, MultiSpan, DUMMY_SP};
use edition::Edition;
use errors::{DiagnosticBuilder, DiagnosticId};
use ext::expand::{self, AstFragment, Invocation};
use ext::hygiene::{self, Mark, SyntaxContext, Transparency};
use fold::{self, Folder};
use parse::{self, parser, DirectoryOwnership};
use parse::token;
use ptr::P;
use OneVector;
use symbol::{keywords, Ident, Symbol};
use ThinVec;

use std::collections::HashMap;
use std::iter;
use std::path::PathBuf;
use std::rc::Rc;
use rustc_data_structures::sync::{self, Lrc};
use std::default::Default;
use tokenstream::{self, TokenStream};


#[derive(Debug,Clone)]
pub enum Annotatable {
    Item(P<ast::Item>),
    TraitItem(P<ast::TraitItem>),
    ImplItem(P<ast::ImplItem>),
    ForeignItem(P<ast::ForeignItem>),
    Stmt(P<ast::Stmt>),
    Expr(P<ast::Expr>),
}

impl HasAttrs for Annotatable {
    fn attrs(&self) -> &[Attribute] {
        match *self {
            Annotatable::Item(ref item) => &item.attrs,
            Annotatable::TraitItem(ref trait_item) => &trait_item.attrs,
            Annotatable::ImplItem(ref impl_item) => &impl_item.attrs,
            Annotatable::ForeignItem(ref foreign_item) => &foreign_item.attrs,
            Annotatable::Stmt(ref stmt) => stmt.attrs(),
            Annotatable::Expr(ref expr) => &expr.attrs,
        }
    }

    fn map_attrs<F: FnOnce(Vec<Attribute>) -> Vec<Attribute>>(self, f: F) -> Self {
        match self {
            Annotatable::Item(item) => Annotatable::Item(item.map_attrs(f)),
            Annotatable::TraitItem(trait_item) => Annotatable::TraitItem(trait_item.map_attrs(f)),
            Annotatable::ImplItem(impl_item) => Annotatable::ImplItem(impl_item.map_attrs(f)),
            Annotatable::ForeignItem(foreign_item) =>
                Annotatable::ForeignItem(foreign_item.map_attrs(f)),
            Annotatable::Stmt(stmt) => Annotatable::Stmt(stmt.map_attrs(f)),
            Annotatable::Expr(expr) => Annotatable::Expr(expr.map_attrs(f)),
        }
    }
}

impl Annotatable {
    pub fn span(&self) -> Span {
        match *self {
            Annotatable::Item(ref item) => item.span,
            Annotatable::TraitItem(ref trait_item) => trait_item.span,
            Annotatable::ImplItem(ref impl_item) => impl_item.span,
            Annotatable::ForeignItem(ref foreign_item) => foreign_item.span,
            Annotatable::Stmt(ref stmt) => stmt.span,
            Annotatable::Expr(ref expr) => expr.span,
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

    pub fn expect_trait_item(self) -> ast::TraitItem {
        match self {
            Annotatable::TraitItem(i) => i.into_inner(),
            _ => panic!("expected Item")
        }
    }

    pub fn expect_impl_item(self) -> ast::ImplItem {
        match self {
            Annotatable::ImplItem(i) => i.into_inner(),
            _ => panic!("expected Item")
        }
    }

    pub fn expect_foreign_item(self) -> ast::ForeignItem {
        match self {
            Annotatable::ForeignItem(i) => i.into_inner(),
            _ => panic!("expected foreign item")
        }
    }

    pub fn expect_stmt(self) -> ast::Stmt {
        match self {
            Annotatable::Stmt(stmt) => stmt.into_inner(),
            _ => panic!("expected statement"),
        }
    }

    pub fn expect_expr(self) -> P<ast::Expr> {
        match self {
            Annotatable::Expr(expr) => expr,
            _ => panic!("expected expression"),
        }
    }

    pub fn derive_allowed(&self) -> bool {
        match *self {
            Annotatable::Item(ref item) => match item.node {
                ast::ItemKind::Struct(..) |
                ast::ItemKind::Enum(..) |
                ast::ItemKind::Union(..) => true,
                _ => false,
            },
            _ => false,
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
              push: &mut dyn FnMut(Annotatable));
}

impl<F> MultiItemDecorator for F
    where F : Fn(&mut ExtCtxt, Span, &ast::MetaItem, &Annotatable, &mut dyn FnMut(Annotatable))
{
    fn expand(&self,
              ecx: &mut ExtCtxt,
              sp: Span,
              meta_item: &ast::MetaItem,
              item: &Annotatable,
              push: &mut dyn FnMut(Annotatable)) {
        (*self)(ecx, sp, meta_item, item, push)
    }
}

// `meta_item` is the annotation, and `item` is the item being modified.
// FIXME Decorators should follow the same pattern too.
pub trait MultiItemModifier {
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Vec<Annotatable>;
}

impl<F, T> MultiItemModifier for F
    where F: Fn(&mut ExtCtxt, Span, &ast::MetaItem, Annotatable) -> T,
          T: Into<Vec<Annotatable>>,
{
    fn expand(&self,
              ecx: &mut ExtCtxt,
              span: Span,
              meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Vec<Annotatable> {
        (*self)(ecx, span, meta_item, item).into()
    }
}

impl Into<Vec<Annotatable>> for Annotatable {
    fn into(self) -> Vec<Annotatable> {
        vec![self]
    }
}

pub trait ProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   ts: TokenStream)
                   -> TokenStream;
}

impl<F> ProcMacro for F
    where F: Fn(TokenStream) -> TokenStream
{
    fn expand<'cx>(&self,
                   _ecx: &'cx mut ExtCtxt,
                   _span: Span,
                   ts: TokenStream)
                   -> TokenStream {
        // FIXME setup implicit context in TLS before calling self.
        (*self)(ts)
    }
}

pub trait AttrProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt,
                   span: Span,
                   annotation: TokenStream,
                   annotated: TokenStream)
                   -> TokenStream;
}

impl<F> AttrProcMacro for F
    where F: Fn(TokenStream, TokenStream) -> TokenStream
{
    fn expand<'cx>(&self,
                   _ecx: &'cx mut ExtCtxt,
                   _span: Span,
                   annotation: TokenStream,
                   annotated: TokenStream)
                   -> TokenStream {
        // FIXME setup implicit context in TLS before calling self.
        (*self)(annotation, annotated)
    }
}

/// Represents a thing that maps token trees to Macro Results
pub trait TTMacroExpander {
    fn expand<'cx>(&self, ecx: &'cx mut ExtCtxt, span: Span, input: TokenStream)
                   -> Box<dyn MacResult+'cx>;
}

pub type MacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt, Span, &[tokenstream::TokenTree])
                -> Box<dyn MacResult+'cx>;

impl<F> TTMacroExpander for F
    where F: for<'cx> Fn(&'cx mut ExtCtxt, Span, &[tokenstream::TokenTree])
    -> Box<dyn MacResult+'cx>
{
    fn expand<'cx>(&self, ecx: &'cx mut ExtCtxt, span: Span, input: TokenStream)
                   -> Box<dyn MacResult+'cx> {
        struct AvoidInterpolatedIdents;

        impl Folder for AvoidInterpolatedIdents {
            fn fold_tt(&mut self, tt: tokenstream::TokenTree) -> tokenstream::TokenTree {
                if let tokenstream::TokenTree::Token(_, token::Interpolated(ref nt)) = tt {
                    if let token::NtIdent(ident, is_raw) = nt.0 {
                        return tokenstream::TokenTree::Token(ident.span,
                                                             token::Ident(ident, is_raw));
                    }
                }
                fold::noop_fold_tt(tt, self)
            }

            fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
                fold::noop_fold_mac(mac, self)
            }
        }

        let input: Vec<_> =
            input.trees().map(|tt| AvoidInterpolatedIdents.fold_tt(tt)).collect();
        (*self)(ecx, span, &input)
    }
}

pub trait IdentMacroExpander {
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   ident: ast::Ident,
                   token_tree: Vec<tokenstream::TokenTree>)
                   -> Box<dyn MacResult+'cx>;
}

pub type IdentMacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt, Span, ast::Ident, Vec<tokenstream::TokenTree>)
                -> Box<dyn MacResult+'cx>;

impl<F> IdentMacroExpander for F
    where F : for<'cx> Fn(&'cx mut ExtCtxt, Span, ast::Ident,
                          Vec<tokenstream::TokenTree>) -> Box<dyn MacResult+'cx>
{
    fn expand<'cx>(&self,
                   cx: &'cx mut ExtCtxt,
                   sp: Span,
                   ident: ast::Ident,
                   token_tree: Vec<tokenstream::TokenTree>)
                   -> Box<dyn MacResult+'cx>
    {
        (*self)(cx, sp, ident, token_tree)
    }
}

// Use a macro because forwarding to a simple function has type system issues
macro_rules! make_stmts_default {
    ($me:expr) => {
        $me.make_expr().map(|e| smallvec![ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            span: e.span,
            node: ast::StmtKind::Expr(e),
        }])
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
    fn make_items(self: Box<Self>) -> Option<OneVector<P<ast::Item>>> {
        None
    }

    /// Create zero or more impl items.
    fn make_impl_items(self: Box<Self>) -> Option<OneVector<ast::ImplItem>> {
        None
    }

    /// Create zero or more trait items.
    fn make_trait_items(self: Box<Self>) -> Option<OneVector<ast::TraitItem>> {
        None
    }

    /// Create zero or more items in an `extern {}` block
    fn make_foreign_items(self: Box<Self>) -> Option<OneVector<ast::ForeignItem>> { None }

    /// Create a pattern.
    fn make_pat(self: Box<Self>) -> Option<P<ast::Pat>> {
        None
    }

    /// Create zero or more statements.
    ///
    /// By default this attempts to create an expression statement,
    /// returning None if that fails.
    fn make_stmts(self: Box<Self>) -> Option<OneVector<ast::Stmt>> {
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
                pub fn $fld(v: $t) -> Box<dyn MacResult> {
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
    items: OneVector<P<ast::Item>>,
    impl_items: OneVector<ast::ImplItem>,
    trait_items: OneVector<ast::TraitItem>,
    foreign_items: OneVector<ast::ForeignItem>,
    stmts: OneVector<ast::Stmt>,
    ty: P<ast::Ty>,
}

impl MacResult for MacEager {
    fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
        self.expr
    }

    fn make_items(self: Box<Self>) -> Option<OneVector<P<ast::Item>>> {
        self.items
    }

    fn make_impl_items(self: Box<Self>) -> Option<OneVector<ast::ImplItem>> {
        self.impl_items
    }

    fn make_trait_items(self: Box<Self>) -> Option<OneVector<ast::TraitItem>> {
        self.trait_items
    }

    fn make_foreign_items(self: Box<Self>) -> Option<OneVector<ast::ForeignItem>> {
        self.foreign_items
    }

    fn make_stmts(self: Box<Self>) -> Option<OneVector<ast::Stmt>> {
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
            if let ast::ExprKind::Lit(_) = e.node {
                return Some(P(ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    span: e.span,
                    node: PatKind::Lit(e),
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
    pub fn any(sp: Span) -> Box<dyn MacResult+'static> {
        Box::new(DummyResult { expr_only: false, span: sp })
    }

    /// Create a default MacResult that can only be an expression.
    ///
    /// Use this for macros that must expand to an expression, so even
    /// if an error is encountered internally, the user will receive
    /// an error that they also used it in the wrong place.
    pub fn expr(sp: Span) -> Box<dyn MacResult+'static> {
        Box::new(DummyResult { expr_only: true, span: sp })
    }

    /// A plain dummy expression.
    pub fn raw_expr(sp: Span) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node: ast::ExprKind::Lit(P(source_map::respan(sp, ast::LitKind::Bool(false)))),
            span: sp,
            attrs: ThinVec::new(),
        })
    }

    /// A plain dummy pattern.
    pub fn raw_pat(sp: Span) -> ast::Pat {
        ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: PatKind::Wild,
            span: sp,
        }
    }

    pub fn raw_ty(sp: Span) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            node: ast::TyKind::Infer,
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

    fn make_items(self: Box<DummyResult>) -> Option<OneVector<P<ast::Item>>> {
        // this code needs a comment... why not always just return the Some() ?
        if self.expr_only {
            None
        } else {
            Some(OneVector::new())
        }
    }

    fn make_impl_items(self: Box<DummyResult>) -> Option<OneVector<ast::ImplItem>> {
        if self.expr_only {
            None
        } else {
            Some(OneVector::new())
        }
    }

    fn make_trait_items(self: Box<DummyResult>) -> Option<OneVector<ast::TraitItem>> {
        if self.expr_only {
            None
        } else {
            Some(OneVector::new())
        }
    }

    fn make_foreign_items(self: Box<Self>) -> Option<OneVector<ast::ForeignItem>> {
        if self.expr_only {
            None
        } else {
            Some(OneVector::new())
        }
    }

    fn make_stmts(self: Box<DummyResult>) -> Option<OneVector<ast::Stmt>> {
        Some(smallvec![ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Expr(DummyResult::raw_expr(self.span)),
            span: self.span,
        }])
    }

    fn make_ty(self: Box<DummyResult>) -> Option<P<ast::Ty>> {
        Some(DummyResult::raw_ty(self.span))
    }
}

pub type BuiltinDeriveFn =
    for<'cx> fn(&'cx mut ExtCtxt, Span, &MetaItem, &Annotatable, &mut dyn FnMut(Annotatable));

/// Represents different kinds of macro invocations that can be resolved.
#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum MacroKind {
    /// A bang macro - foo!()
    Bang,
    /// An attribute macro - #[foo]
    Attr,
    /// A derive attribute macro - #[derive(Foo)]
    Derive,
    /// A view of a procedural macro from the same crate that defines it.
    ProcMacroStub,
}

impl MacroKind {
    pub fn descr(self) -> &'static str {
        match self {
            MacroKind::Bang => "macro",
            MacroKind::Attr => "attribute macro",
            MacroKind::Derive => "derive macro",
            MacroKind::ProcMacroStub => "crate-local procedural macro",
        }
    }
}

/// An enum representing the different kinds of syntax extensions.
pub enum SyntaxExtension {
    /// A trivial "extension" that does nothing, only keeps the attribute and marks it as known.
    NonMacroAttr { mark_used: bool },

    /// A syntax extension that is attached to an item and creates new items
    /// based upon it.
    ///
    /// `#[derive(...)]` is a `MultiItemDecorator`.
    ///
    /// Prefer ProcMacro or MultiModifier since they are more flexible.
    MultiDecorator(Box<dyn MultiItemDecorator + sync::Sync + sync::Send>),

    /// A syntax extension that is attached to an item and modifies it
    /// in-place. Also allows decoration, i.e., creating new items.
    MultiModifier(Box<dyn MultiItemModifier + sync::Sync + sync::Send>),

    /// A function-like procedural macro. TokenStream -> TokenStream.
    ProcMacro {
        expander: Box<dyn ProcMacro + sync::Sync + sync::Send>,
        allow_internal_unstable: bool,
        edition: Edition,
    },

    /// An attribute-like procedural macro. TokenStream, TokenStream -> TokenStream.
    /// The first TokenSteam is the attribute, the second is the annotated item.
    /// Allows modification of the input items and adding new items, similar to
    /// MultiModifier, but uses TokenStreams, rather than AST nodes.
    AttrProcMacro(Box<dyn AttrProcMacro + sync::Sync + sync::Send>, Edition),

    /// A normal, function-like syntax extension.
    ///
    /// `bytes!` is a `NormalTT`.
    NormalTT {
        expander: Box<dyn TTMacroExpander + sync::Sync + sync::Send>,
        def_info: Option<(ast::NodeId, Span)>,
        /// Whether the contents of the macro can
        /// directly use `#[unstable]` things (true == yes).
        allow_internal_unstable: bool,
        /// Whether the contents of the macro can use `unsafe`
        /// without triggering the `unsafe_code` lint.
        allow_internal_unsafe: bool,
        /// Enables the macro helper hack (`ident!(...)` -> `$crate::ident!(...)`)
        /// for a given macro.
        local_inner_macros: bool,
        /// The macro's feature name if it is unstable, and the stability feature
        unstable_feature: Option<(Symbol, u32)>,
        /// Edition of the crate in which the macro is defined
        edition: Edition,
    },

    /// A function-like syntax extension that has an extra ident before
    /// the block.
    ///
    IdentTT(Box<dyn IdentMacroExpander + sync::Sync + sync::Send>, Option<Span>, bool),

    /// An attribute-like procedural macro. TokenStream -> TokenStream.
    /// The input is the annotated item.
    /// Allows generating code to implement a Trait for a given struct
    /// or enum item.
    ProcMacroDerive(Box<dyn MultiItemModifier + sync::Sync + sync::Send>,
                    Vec<Symbol> /* inert attribute names */, Edition),

    /// An attribute-like procedural macro that derives a builtin trait.
    BuiltinDerive(BuiltinDeriveFn),

    /// A declarative macro, e.g. `macro m() {}`.
    DeclMacro {
        expander: Box<dyn TTMacroExpander + sync::Sync + sync::Send>,
        def_info: Option<(ast::NodeId, Span)>,
        is_transparent: bool,
        edition: Edition,
    }
}

impl SyntaxExtension {
    /// Return which kind of macro calls this syntax extension.
    pub fn kind(&self) -> MacroKind {
        match *self {
            SyntaxExtension::DeclMacro { .. } |
            SyntaxExtension::NormalTT { .. } |
            SyntaxExtension::IdentTT(..) |
            SyntaxExtension::ProcMacro { .. } =>
                MacroKind::Bang,
            SyntaxExtension::NonMacroAttr { .. } |
            SyntaxExtension::MultiDecorator(..) |
            SyntaxExtension::MultiModifier(..) |
            SyntaxExtension::AttrProcMacro(..) =>
                MacroKind::Attr,
            SyntaxExtension::ProcMacroDerive(..) |
            SyntaxExtension::BuiltinDerive(..) =>
                MacroKind::Derive,
        }
    }

    pub fn default_transparency(&self) -> Transparency {
        match *self {
            SyntaxExtension::ProcMacro { .. } |
            SyntaxExtension::AttrProcMacro(..) |
            SyntaxExtension::ProcMacroDerive(..) |
            SyntaxExtension::DeclMacro { is_transparent: false, .. } => Transparency::Opaque,
            SyntaxExtension::DeclMacro { is_transparent: true, .. } => Transparency::Transparent,
            _ => Transparency::SemiTransparent,
        }
    }

    pub fn edition(&self) -> Edition {
        match *self {
            SyntaxExtension::NormalTT { edition, .. } |
            SyntaxExtension::DeclMacro { edition, .. } |
            SyntaxExtension::ProcMacro { edition, .. } |
            SyntaxExtension::AttrProcMacro(.., edition) |
            SyntaxExtension::ProcMacroDerive(.., edition) => edition,
            // Unstable legacy stuff
            SyntaxExtension::NonMacroAttr { .. } |
            SyntaxExtension::IdentTT(..) |
            SyntaxExtension::MultiDecorator(..) |
            SyntaxExtension::MultiModifier(..) |
            SyntaxExtension::BuiltinDerive(..) => hygiene::default_edition(),
        }
    }
}

pub type NamedSyntaxExtension = (Name, SyntaxExtension);

pub trait Resolver {
    fn next_node_id(&mut self) -> ast::NodeId;
    fn get_module_scope(&mut self, id: ast::NodeId) -> Mark;
    fn eliminate_crate_var(&mut self, item: P<ast::Item>) -> P<ast::Item>;
    fn is_whitelisted_legacy_custom_derive(&self, name: Name) -> bool;

    fn visit_ast_fragment_with_placeholders(&mut self, mark: Mark, fragment: &AstFragment,
                                            derives: &[Mark]);
    fn add_builtin(&mut self, ident: ast::Ident, ext: Lrc<SyntaxExtension>);

    fn resolve_imports(&mut self);
    // Resolves attribute and derive legacy macros from `#![plugin(..)]`.
    fn find_legacy_attr_invoc(&mut self, attrs: &mut Vec<Attribute>, allow_derive: bool)
                              -> Option<Attribute>;

    fn resolve_macro_invocation(&mut self, invoc: &Invocation, scope: Mark, force: bool)
                                -> Result<Option<Lrc<SyntaxExtension>>, Determinacy>;
    fn resolve_macro_path(&mut self, path: &ast::Path, kind: MacroKind, scope: Mark,
                          derives_in_scope: &[ast::Path], force: bool)
                          -> Result<Lrc<SyntaxExtension>, Determinacy>;

    fn check_unused_macros(&self);
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Determinacy {
    Determined,
    Undetermined,
}

impl Determinacy {
    pub fn determined(determined: bool) -> Determinacy {
        if determined { Determinacy::Determined } else { Determinacy::Undetermined }
    }
}

pub struct DummyResolver;

impl Resolver for DummyResolver {
    fn next_node_id(&mut self) -> ast::NodeId { ast::DUMMY_NODE_ID }
    fn get_module_scope(&mut self, _id: ast::NodeId) -> Mark { Mark::root() }
    fn eliminate_crate_var(&mut self, item: P<ast::Item>) -> P<ast::Item> { item }
    fn is_whitelisted_legacy_custom_derive(&self, _name: Name) -> bool { false }

    fn visit_ast_fragment_with_placeholders(&mut self, _invoc: Mark, _fragment: &AstFragment,
                                            _derives: &[Mark]) {}
    fn add_builtin(&mut self, _ident: ast::Ident, _ext: Lrc<SyntaxExtension>) {}

    fn resolve_imports(&mut self) {}
    fn find_legacy_attr_invoc(&mut self, _attrs: &mut Vec<Attribute>, _allow_derive: bool)
                              -> Option<Attribute> { None }
    fn resolve_macro_invocation(&mut self, _invoc: &Invocation, _scope: Mark, _force: bool)
                                -> Result<Option<Lrc<SyntaxExtension>>, Determinacy> {
        Err(Determinacy::Determined)
    }
    fn resolve_macro_path(&mut self, _path: &ast::Path, _kind: MacroKind, _scope: Mark,
                          _derives_in_scope: &[ast::Path], _force: bool)
                          -> Result<Lrc<SyntaxExtension>, Determinacy> {
        Err(Determinacy::Determined)
    }
    fn check_unused_macros(&self) {}
}

#[derive(Clone)]
pub struct ModuleData {
    pub mod_path: Vec<ast::Ident>,
    pub directory: PathBuf,
}

#[derive(Clone)]
pub struct ExpansionData {
    pub mark: Mark,
    pub depth: usize,
    pub module: Rc<ModuleData>,
    pub directory_ownership: DirectoryOwnership,
    pub crate_span: Option<Span>,
}

/// One of these is made during expansion and incrementally updated as we go;
/// when a macro expansion occurs, the resulting nodes have the `backtrace()
/// -> expn_info` of their expansion context stored into their span.
pub struct ExtCtxt<'a> {
    pub parse_sess: &'a parse::ParseSess,
    pub ecfg: expand::ExpansionConfig<'a>,
    pub root_path: PathBuf,
    pub resolver: &'a mut dyn Resolver,
    pub resolve_err_count: usize,
    pub current_expansion: ExpansionData,
    pub expansions: HashMap<Span, Vec<String>>,
}

impl<'a> ExtCtxt<'a> {
    pub fn new(parse_sess: &'a parse::ParseSess,
               ecfg: expand::ExpansionConfig<'a>,
               resolver: &'a mut dyn Resolver)
               -> ExtCtxt<'a> {
        ExtCtxt {
            parse_sess,
            ecfg,
            root_path: PathBuf::new(),
            resolver,
            resolve_err_count: 0,
            current_expansion: ExpansionData {
                mark: Mark::root(),
                depth: 0,
                module: Rc::new(ModuleData { mod_path: Vec::new(), directory: PathBuf::new() }),
                directory_ownership: DirectoryOwnership::Owned { relative: None },
                crate_span: None,
            },
            expansions: HashMap::new(),
        }
    }

    /// Returns a `Folder` for deeply expanding all macros in an AST node.
    pub fn expander<'b>(&'b mut self) -> expand::MacroExpander<'b, 'a> {
        expand::MacroExpander::new(self, false)
    }

    /// Returns a `Folder` that deeply expands all macros and assigns all node ids in an AST node.
    /// Once node ids are assigned, the node may not be expanded, removed, or otherwise modified.
    pub fn monotonic_expander<'b>(&'b mut self) -> expand::MacroExpander<'b, 'a> {
        expand::MacroExpander::new(self, true)
    }

    pub fn new_parser_from_tts(&self, tts: &[tokenstream::TokenTree]) -> parser::Parser<'a> {
        parse::stream_to_parser(self.parse_sess, tts.iter().cloned().collect())
    }
    pub fn source_map(&self) -> &'a SourceMap { self.parse_sess.source_map() }
    pub fn parse_sess(&self) -> &'a parse::ParseSess { self.parse_sess }
    pub fn cfg(&self) -> &ast::CrateConfig { &self.parse_sess.config }
    pub fn call_site(&self) -> Span {
        match self.current_expansion.mark.expn_info() {
            Some(expn_info) => expn_info.call_site,
            None => DUMMY_SP,
        }
    }
    pub fn backtrace(&self) -> SyntaxContext {
        SyntaxContext::empty().apply_mark(self.current_expansion.mark)
    }

    /// Returns span for the macro which originally caused the current expansion to happen.
    ///
    /// Stops backtracing at include! boundary.
    pub fn expansion_cause(&self) -> Option<Span> {
        let mut ctxt = self.backtrace();
        let mut last_macro = None;
        loop {
            if ctxt.outer().expn_info().map_or(None, |info| {
                if info.format.name() == "include" {
                    // Stop going up the backtrace once include! is encountered
                    return None;
                }
                ctxt = info.call_site.ctxt();
                last_macro = Some(info.call_site);
                Some(())
            }).is_none() {
                break
            }
        }
        last_macro
    }

    pub fn struct_span_warn<S: Into<MultiSpan>>(&self,
                                                sp: S,
                                                msg: &str)
                                                -> DiagnosticBuilder<'a> {
        self.parse_sess.span_diagnostic.struct_span_warn(sp, msg)
    }
    pub fn struct_span_err<S: Into<MultiSpan>>(&self,
                                               sp: S,
                                               msg: &str)
                                               -> DiagnosticBuilder<'a> {
        self.parse_sess.span_diagnostic.struct_span_err(sp, msg)
    }
    pub fn struct_span_fatal<S: Into<MultiSpan>>(&self,
                                                 sp: S,
                                                 msg: &str)
                                                 -> DiagnosticBuilder<'a> {
        self.parse_sess.span_diagnostic.struct_span_fatal(sp, msg)
    }

    /// Emit `msg` attached to `sp`, and stop compilation immediately.
    ///
    /// `span_err` should be strongly preferred where-ever possible:
    /// this should *only* be used when:
    ///
    /// - continuing has a high risk of flow-on errors (e.g. errors in
    ///   declaring a macro would cause all uses of that macro to
    ///   complain about "undefined macro"), or
    /// - there is literally nothing else that can be done (however,
    ///   in most cases one can construct a dummy expression/item to
    ///   substitute; we never hit resolve/type-checking so the dummy
    ///   value doesn't have to match anything)
    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.span_fatal(sp, msg).raise();
    }

    /// Emit `msg` attached to `sp`, without immediately stopping
    /// compilation.
    ///
    /// Compilation will be stopped in the near future (at the end of
    /// the macro expansion phase).
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.parse_sess.span_diagnostic.span_err(sp, msg);
    }
    pub fn span_err_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: DiagnosticId) {
        self.parse_sess.span_diagnostic.span_err_with_code(sp, msg, code);
    }
    pub fn mut_span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str)
                        -> DiagnosticBuilder<'a> {
        self.parse_sess.span_diagnostic.mut_span_err(sp, msg)
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.parse_sess.span_diagnostic.span_warn(sp, msg);
    }
    pub fn span_unimpl<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.span_unimpl(sp, msg);
    }
    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.span_bug(sp, msg);
    }
    pub fn trace_macros_diag(&mut self) {
        for (sp, notes) in self.expansions.iter() {
            let mut db = self.parse_sess.span_diagnostic.span_note_diag(*sp, "trace_macro");
            for note in notes {
                db.note(note);
            }
            db.emit();
        }
        // Fixme: does this result in errors?
        self.expansions.clear();
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.parse_sess.span_diagnostic.bug(msg);
    }
    pub fn trace_macros(&self) -> bool {
        self.ecfg.trace_mac
    }
    pub fn set_trace_macros(&mut self, x: bool) {
        self.ecfg.trace_mac = x
    }
    pub fn ident_of(&self, st: &str) -> ast::Ident {
        ast::Ident::from_str(st)
    }
    pub fn std_path(&self, components: &[&str]) -> Vec<ast::Ident> {
        let def_site = DUMMY_SP.apply_mark(self.current_expansion.mark);
        iter::once(Ident::new(keywords::DollarCrate.name(), def_site))
            .chain(components.iter().map(|s| self.ident_of(s)))
            .collect()
    }
    pub fn name_of(&self, st: &str) -> ast::Name {
        Symbol::intern(st)
    }

    pub fn check_unused_macros(&self) {
        self.resolver.check_unused_macros();
    }
}

/// Extract a string literal from the macro expanded version of `expr`,
/// emitting `err_msg` if `expr` is not a string literal. This does not stop
/// compilation on error, merely emits a non-fatal error and returns None.
pub fn expr_to_spanned_string<'a>(
    cx: &'a mut ExtCtxt,
    expr: P<ast::Expr>,
    err_msg: &str,
) -> Result<Spanned<(Symbol, ast::StrStyle)>, DiagnosticBuilder<'a>> {
    // Update `expr.span`'s ctxt now in case expr is an `include!` macro invocation.
    let expr = expr.map(|mut expr| {
        expr.span = expr.span.apply_mark(cx.current_expansion.mark);
        expr
    });

    // we want to be able to handle e.g. `concat!("foo", "bar")`
    let expr = cx.expander().fold_expr(expr);
    Err(match expr.node {
        ast::ExprKind::Lit(ref l) => match l.node {
            ast::LitKind::Str(s, style) => return Ok(respan(expr.span, (s, style))),
            _ => cx.struct_span_err(l.span, err_msg)
        },
        _ => cx.struct_span_err(expr.span, err_msg)
    })
}

pub fn expr_to_string(cx: &mut ExtCtxt, expr: P<ast::Expr>, err_msg: &str)
                      -> Option<(Symbol, ast::StrStyle)> {
    expr_to_spanned_string(cx, expr, err_msg)
        .map_err(|mut err| err.emit())
        .ok()
        .map(|s| s.node)
}

/// Non-fatally assert that `tts` is empty. Note that this function
/// returns even when `tts` is non-empty, macros that *need* to stop
/// compilation should call
/// `cx.parse_sess.span_diagnostic.abort_if_errors()` (this should be
/// done as rarely as possible).
pub fn check_zero_tts(cx: &ExtCtxt,
                      sp: Span,
                      tts: &[tokenstream::TokenTree],
                      name: &str) {
    if !tts.is_empty() {
        cx.span_err(sp, &format!("{} takes no arguments", name));
    }
}

/// Interpreting `tts` as a comma-separated sequence of expressions,
/// expect exactly one string literal, or emit an error and return None.
pub fn get_single_str_from_tts(cx: &mut ExtCtxt,
                               sp: Span,
                               tts: &[tokenstream::TokenTree],
                               name: &str)
                               -> Option<String> {
    let mut p = cx.new_parser_from_tts(tts);
    if p.token == token::Eof {
        cx.span_err(sp, &format!("{} takes 1 argument", name));
        return None
    }
    let ret = panictry!(p.parse_expr());
    let _ = p.eat(&token::Comma);

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
                          tts: &[tokenstream::TokenTree]) -> Option<Vec<P<ast::Expr>>> {
    let mut p = cx.new_parser_from_tts(tts);
    let mut es = Vec::new();
    while p.token != token::Eof {
        es.push(cx.expander().fold_expr(panictry!(p.parse_expr())));
        if p.eat(&token::Comma) {
            continue;
        }
        if p.token != token::Eof {
            cx.span_err(sp, "expected token: `,`");
            return None;
        }
    }
    Some(es)
}
