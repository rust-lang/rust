use crate::ast::{self, Attribute, Name, PatKind};
use crate::attr::HasAttrs;
use crate::source_map::{SourceMap, Spanned, respan};
use crate::edition::Edition;
use crate::ext::expand::{self, AstFragment, Invocation};
use crate::ext::hygiene::{Mark, SyntaxContext, Transparency};
use crate::mut_visit::{self, MutVisitor};
use crate::parse::{self, parser, DirectoryOwnership};
use crate::parse::token;
use crate::ptr::P;
use crate::symbol::{kw, sym, Ident, Symbol};
use crate::{ThinVec, MACRO_ARGUMENTS};
use crate::tokenstream::{self, TokenStream};

use errors::{DiagnosticBuilder, DiagnosticId};
use smallvec::{smallvec, SmallVec};
use syntax_pos::{Span, MultiSpan, DUMMY_SP};
use syntax_pos::hygiene::{ExpnInfo, ExpnFormat};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{self, Lrc};
use std::iter;
use std::path::PathBuf;
use std::rc::Rc;
use std::default::Default;


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

    fn visit_attrs<F: FnOnce(&mut Vec<Attribute>)>(&mut self, f: F) {
        match self {
            Annotatable::Item(item) => item.visit_attrs(f),
            Annotatable::TraitItem(trait_item) => trait_item.visit_attrs(f),
            Annotatable::ImplItem(impl_item) => impl_item.visit_attrs(f),
            Annotatable::ForeignItem(foreign_item) => foreign_item.visit_attrs(f),
            Annotatable::Stmt(stmt) => stmt.visit_attrs(f),
            Annotatable::Expr(expr) => expr.visit_attrs(f),
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

// `meta_item` is the annotation, and `item` is the item being modified.
// FIXME Decorators should follow the same pattern too.
pub trait MultiItemModifier {
    fn expand(&self,
              ecx: &mut ExtCtxt<'_>,
              span: Span,
              meta_item: &ast::MetaItem,
              item: Annotatable)
              -> Vec<Annotatable>;
}

impl<F, T> MultiItemModifier for F
    where F: Fn(&mut ExtCtxt<'_>, Span, &ast::MetaItem, Annotatable) -> T,
          T: Into<Vec<Annotatable>>,
{
    fn expand(&self,
              ecx: &mut ExtCtxt<'_>,
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
                   ecx: &'cx mut ExtCtxt<'_>,
                   span: Span,
                   ts: TokenStream)
                   -> TokenStream;
}

impl<F> ProcMacro for F
    where F: Fn(TokenStream) -> TokenStream
{
    fn expand<'cx>(&self,
                   _ecx: &'cx mut ExtCtxt<'_>,
                   _span: Span,
                   ts: TokenStream)
                   -> TokenStream {
        // FIXME setup implicit context in TLS before calling self.
        (*self)(ts)
    }
}

pub trait AttrProcMacro {
    fn expand<'cx>(&self,
                   ecx: &'cx mut ExtCtxt<'_>,
                   span: Span,
                   annotation: TokenStream,
                   annotated: TokenStream)
                   -> TokenStream;
}

impl<F> AttrProcMacro for F
    where F: Fn(TokenStream, TokenStream) -> TokenStream
{
    fn expand<'cx>(&self,
                   _ecx: &'cx mut ExtCtxt<'_>,
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
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
        def_span: Option<Span>,
    ) -> Box<dyn MacResult+'cx>;
}

pub type MacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt<'_>, Span, &[tokenstream::TokenTree])
                -> Box<dyn MacResult+'cx>;

impl<F> TTMacroExpander for F
    where F: for<'cx> Fn(&'cx mut ExtCtxt<'_>, Span, &[tokenstream::TokenTree])
    -> Box<dyn MacResult+'cx>
{
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
        _def_span: Option<Span>,
    ) -> Box<dyn MacResult+'cx> {
        struct AvoidInterpolatedIdents;

        impl MutVisitor for AvoidInterpolatedIdents {
            fn visit_tt(&mut self, tt: &mut tokenstream::TokenTree) {
                if let tokenstream::TokenTree::Token(token) = tt {
                    if let token::Interpolated(nt) = &token.kind {
                        if let token::NtIdent(ident, is_raw) = **nt {
                            *tt = tokenstream::TokenTree::token(
                                token::Ident(ident.name, is_raw), ident.span
                            );
                        }
                    }
                }
                mut_visit::noop_visit_tt(tt, self)
            }

            fn visit_mac(&mut self, mac: &mut ast::Mac) {
                mut_visit::noop_visit_mac(mac, self)
            }
        }

        let input: Vec<_> =
            input.trees().map(|mut tt| { AvoidInterpolatedIdents.visit_tt(&mut tt); tt }).collect();
        (*self)(ecx, span, &input)
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
    /// Creates an expression.
    fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
        None
    }
    /// Creates zero or more items.
    fn make_items(self: Box<Self>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
        None
    }

    /// Creates zero or more impl items.
    fn make_impl_items(self: Box<Self>) -> Option<SmallVec<[ast::ImplItem; 1]>> {
        None
    }

    /// Creates zero or more trait items.
    fn make_trait_items(self: Box<Self>) -> Option<SmallVec<[ast::TraitItem; 1]>> {
        None
    }

    /// Creates zero or more items in an `extern {}` block
    fn make_foreign_items(self: Box<Self>) -> Option<SmallVec<[ast::ForeignItem; 1]>> { None }

    /// Creates a pattern.
    fn make_pat(self: Box<Self>) -> Option<P<ast::Pat>> {
        None
    }

    /// Creates zero or more statements.
    ///
    /// By default this attempts to create an expression statement,
    /// returning None if that fails.
    fn make_stmts(self: Box<Self>) -> Option<SmallVec<[ast::Stmt; 1]>> {
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
    items: SmallVec<[P<ast::Item>; 1]>,
    impl_items: SmallVec<[ast::ImplItem; 1]>,
    trait_items: SmallVec<[ast::TraitItem; 1]>,
    foreign_items: SmallVec<[ast::ForeignItem; 1]>,
    stmts: SmallVec<[ast::Stmt; 1]>,
    ty: P<ast::Ty>,
}

impl MacResult for MacEager {
    fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
        self.expr
    }

    fn make_items(self: Box<Self>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
        self.items
    }

    fn make_impl_items(self: Box<Self>) -> Option<SmallVec<[ast::ImplItem; 1]>> {
        self.impl_items
    }

    fn make_trait_items(self: Box<Self>) -> Option<SmallVec<[ast::TraitItem; 1]>> {
        self.trait_items
    }

    fn make_foreign_items(self: Box<Self>) -> Option<SmallVec<[ast::ForeignItem; 1]>> {
        self.foreign_items
    }

    fn make_stmts(self: Box<Self>) -> Option<SmallVec<[ast::Stmt; 1]>> {
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
    is_error: bool,
    span: Span,
}

impl DummyResult {
    /// Creates a default MacResult that can be anything.
    ///
    /// Use this as a return value after hitting any errors and
    /// calling `span_err`.
    pub fn any(span: Span) -> Box<dyn MacResult+'static> {
        Box::new(DummyResult { expr_only: false, is_error: true, span })
    }

    /// Same as `any`, but must be a valid fragment, not error.
    pub fn any_valid(span: Span) -> Box<dyn MacResult+'static> {
        Box::new(DummyResult { expr_only: false, is_error: false, span })
    }

    /// Creates a default MacResult that can only be an expression.
    ///
    /// Use this for macros that must expand to an expression, so even
    /// if an error is encountered internally, the user will receive
    /// an error that they also used it in the wrong place.
    pub fn expr(span: Span) -> Box<dyn MacResult+'static> {
        Box::new(DummyResult { expr_only: true, is_error: true, span })
    }

    /// A plain dummy expression.
    pub fn raw_expr(sp: Span, is_error: bool) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node: if is_error { ast::ExprKind::Err } else { ast::ExprKind::Tup(Vec::new()) },
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

    /// A plain dummy type.
    pub fn raw_ty(sp: Span, is_error: bool) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            node: if is_error { ast::TyKind::Err } else { ast::TyKind::Tup(Vec::new()) },
            span: sp
        })
    }
}

impl MacResult for DummyResult {
    fn make_expr(self: Box<DummyResult>) -> Option<P<ast::Expr>> {
        Some(DummyResult::raw_expr(self.span, self.is_error))
    }

    fn make_pat(self: Box<DummyResult>) -> Option<P<ast::Pat>> {
        Some(P(DummyResult::raw_pat(self.span)))
    }

    fn make_items(self: Box<DummyResult>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
        // this code needs a comment... why not always just return the Some() ?
        if self.expr_only {
            None
        } else {
            Some(SmallVec::new())
        }
    }

    fn make_impl_items(self: Box<DummyResult>) -> Option<SmallVec<[ast::ImplItem; 1]>> {
        if self.expr_only {
            None
        } else {
            Some(SmallVec::new())
        }
    }

    fn make_trait_items(self: Box<DummyResult>) -> Option<SmallVec<[ast::TraitItem; 1]>> {
        if self.expr_only {
            None
        } else {
            Some(SmallVec::new())
        }
    }

    fn make_foreign_items(self: Box<Self>) -> Option<SmallVec<[ast::ForeignItem; 1]>> {
        if self.expr_only {
            None
        } else {
            Some(SmallVec::new())
        }
    }

    fn make_stmts(self: Box<DummyResult>) -> Option<SmallVec<[ast::Stmt; 1]>> {
        Some(smallvec![ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            node: ast::StmtKind::Expr(DummyResult::raw_expr(self.span, self.is_error)),
            span: self.span,
        }])
    }

    fn make_ty(self: Box<DummyResult>) -> Option<P<ast::Ty>> {
        Some(DummyResult::raw_ty(self.span, self.is_error))
    }
}

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

    pub fn article(self) -> &'static str {
        match self {
            MacroKind::Attr => "an",
            _ => "a",
        }
    }
}

/// A syntax extension kind.
pub enum SyntaxExtensionKind {
    /// A token-based function-like macro.
    Bang(
        /// An expander with signature TokenStream -> TokenStream.
        Box<dyn ProcMacro + sync::Sync + sync::Send>,
    ),

    /// An AST-based function-like macro.
    LegacyBang(
        /// An expander with signature TokenStream -> AST.
        Box<dyn TTMacroExpander + sync::Sync + sync::Send>,
    ),

    /// A token-based attribute macro.
    Attr(
        /// An expander with signature (TokenStream, TokenStream) -> TokenStream.
        /// The first TokenSteam is the attribute itself, the second is the annotated item.
        /// The produced TokenSteam replaces the input TokenSteam.
        Box<dyn AttrProcMacro + sync::Sync + sync::Send>,
    ),

    /// An AST-based attribute macro.
    LegacyAttr(
        /// An expander with signature (AST, AST) -> AST.
        /// The first AST fragment is the attribute itself, the second is the annotated item.
        /// The produced AST fragment replaces the input AST fragment.
        Box<dyn MultiItemModifier + sync::Sync + sync::Send>,
    ),

    /// A trivial attribute "macro" that does nothing,
    /// only keeps the attribute and marks it as inert,
    /// thus making it ineligible for further expansion.
    NonMacroAttr {
        /// Suppresses the `unused_attributes` lint for this attribute.
        mark_used: bool,
    },

    /// A token-based derive macro.
    Derive(
        /// An expander with signature TokenStream -> TokenStream (not yet).
        /// The produced TokenSteam is appended to the input TokenSteam.
        Box<dyn MultiItemModifier + sync::Sync + sync::Send>,
    ),

    /// An AST-based derive macro.
    LegacyDerive(
        /// An expander with signature AST -> AST.
        /// The produced AST fragment is appended to the input AST fragment.
        Box<dyn MultiItemModifier + sync::Sync + sync::Send>,
    ),
}

/// A struct representing a macro definition in "lowered" form ready for expansion.
pub struct SyntaxExtension {
    /// A syntax extension kind.
    pub kind: SyntaxExtensionKind,
    /// Some info about the macro's definition point.
    pub def_info: Option<(ast::NodeId, Span)>,
    /// Hygienic properties of spans produced by this macro by default.
    pub default_transparency: Transparency,
    /// Whitelist of unstable features that are treated as stable inside this macro.
    pub allow_internal_unstable: Option<Lrc<[Symbol]>>,
    /// Suppresses the `unsafe_code` lint for code produced by this macro.
    pub allow_internal_unsafe: bool,
    /// Enables the macro helper hack (`ident!(...)` -> `$crate::ident!(...)`) for this macro.
    pub local_inner_macros: bool,
    /// The macro's feature name and tracking issue number if it is unstable.
    pub unstable_feature: Option<(Symbol, u32)>,
    /// Names of helper attributes registered by this macro.
    pub helper_attrs: Vec<Symbol>,
    /// Edition of the crate in which this macro is defined.
    pub edition: Edition,
}

impl SyntaxExtensionKind {
    /// When a syntax extension is constructed,
    /// its transparency can often be inferred from its kind.
    fn default_transparency(&self) -> Transparency {
        match self {
            SyntaxExtensionKind::Bang(..) |
            SyntaxExtensionKind::Attr(..) |
            SyntaxExtensionKind::Derive(..) |
            SyntaxExtensionKind::NonMacroAttr { .. } => Transparency::Opaque,
            SyntaxExtensionKind::LegacyBang(..) |
            SyntaxExtensionKind::LegacyAttr(..) |
            SyntaxExtensionKind::LegacyDerive(..) => Transparency::SemiTransparent,
        }
    }
}

impl SyntaxExtension {
    /// Returns which kind of macro calls this syntax extension.
    pub fn macro_kind(&self) -> MacroKind {
        match self.kind {
            SyntaxExtensionKind::Bang(..) |
            SyntaxExtensionKind::LegacyBang(..) => MacroKind::Bang,
            SyntaxExtensionKind::Attr(..) |
            SyntaxExtensionKind::LegacyAttr(..) |
            SyntaxExtensionKind::NonMacroAttr { .. } => MacroKind::Attr,
            SyntaxExtensionKind::Derive(..) |
            SyntaxExtensionKind::LegacyDerive(..) => MacroKind::Derive,
        }
    }

    /// Constructs a syntax extension with default properties.
    pub fn default(kind: SyntaxExtensionKind, edition: Edition) -> SyntaxExtension {
        SyntaxExtension {
            def_info: None,
            default_transparency: kind.default_transparency(),
            allow_internal_unstable: None,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            unstable_feature: None,
            helper_attrs: Vec::new(),
            edition,
            kind,
        }
    }

    fn expn_format(&self, symbol: Symbol) -> ExpnFormat {
        match self.kind {
            SyntaxExtensionKind::Bang(..) |
            SyntaxExtensionKind::LegacyBang(..) => ExpnFormat::MacroBang(symbol),
            _ => ExpnFormat::MacroAttribute(symbol),
        }
    }

    pub fn expn_info(&self, call_site: Span, format: &str) -> ExpnInfo {
        ExpnInfo {
            call_site,
            format: self.expn_format(Symbol::intern(format)),
            def_site: self.def_info.map(|(_, span)| span),
            default_transparency: self.default_transparency,
            allow_internal_unstable: self.allow_internal_unstable.clone(),
            allow_internal_unsafe: self.allow_internal_unsafe,
            local_inner_macros: self.local_inner_macros,
            edition: self.edition,
        }
    }
}

pub type NamedSyntaxExtension = (Name, SyntaxExtension);

pub trait Resolver {
    fn next_node_id(&mut self) -> ast::NodeId;

    fn get_module_scope(&mut self, id: ast::NodeId) -> Mark;

    fn resolve_dollar_crates(&mut self, fragment: &AstFragment);
    fn visit_ast_fragment_with_placeholders(&mut self, mark: Mark, fragment: &AstFragment,
                                            derives: &[Mark]);
    fn add_builtin(&mut self, ident: ast::Ident, ext: Lrc<SyntaxExtension>);

    fn resolve_imports(&mut self);

    fn resolve_macro_invocation(&mut self, invoc: &Invocation, invoc_id: Mark, force: bool)
                                -> Result<Option<Lrc<SyntaxExtension>>, Determinacy>;
    fn resolve_macro_path(&mut self, path: &ast::Path, kind: MacroKind, invoc_id: Mark,
                          derives_in_scope: Vec<ast::Path>, force: bool)
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
    pub current_expansion: ExpansionData,
    pub expansions: FxHashMap<Span, Vec<String>>,
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
            current_expansion: ExpansionData {
                mark: Mark::root(),
                depth: 0,
                module: Rc::new(ModuleData { mod_path: Vec::new(), directory: PathBuf::new() }),
                directory_ownership: DirectoryOwnership::Owned { relative: None },
                crate_span: None,
            },
            expansions: FxHashMap::default(),
        }
    }

    /// Returns a `Folder` for deeply expanding all macros in an AST node.
    pub fn expander<'b>(&'b mut self) -> expand::MacroExpander<'b, 'a> {
        expand::MacroExpander::new(self, false)
    }

    /// Returns a `Folder` that deeply expands all macros and assigns all `NodeId`s in an AST node.
    /// Once `NodeId`s are assigned, the node may not be expanded, removed, or otherwise modified.
    pub fn monotonic_expander<'b>(&'b mut self) -> expand::MacroExpander<'b, 'a> {
        expand::MacroExpander::new(self, true)
    }

    pub fn new_parser_from_tts(&self, tts: &[tokenstream::TokenTree]) -> parser::Parser<'a> {
        parse::stream_to_parser(self.parse_sess, tts.iter().cloned().collect(), MACRO_ARGUMENTS)
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
            if ctxt.outer_expn_info().map_or(None, |info| {
                if info.format.name() == sym::include {
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
    /// - continuing has a high risk of flow-on errors (e.g., errors in
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
    pub fn std_path(&self, components: &[Symbol]) -> Vec<ast::Ident> {
        let def_site = DUMMY_SP.apply_mark(self.current_expansion.mark);
        iter::once(Ident::new(kw::DollarCrate, def_site))
            .chain(components.iter().map(|&s| Ident::with_empty_ctxt(s)))
            .collect()
    }
    pub fn name_of(&self, st: &str) -> ast::Name {
        Symbol::intern(st)
    }

    pub fn check_unused_macros(&self) {
        self.resolver.check_unused_macros();
    }
}

/// Extracts a string literal from the macro expanded version of `expr`,
/// emitting `err_msg` if `expr` is not a string literal. This does not stop
/// compilation on error, merely emits a non-fatal error and returns `None`.
pub fn expr_to_spanned_string<'a>(
    cx: &'a mut ExtCtxt<'_>,
    mut expr: P<ast::Expr>,
    err_msg: &str,
) -> Result<Spanned<(Symbol, ast::StrStyle)>, Option<DiagnosticBuilder<'a>>> {
    // Update `expr.span`'s ctxt now in case expr is an `include!` macro invocation.
    expr.span = expr.span.apply_mark(cx.current_expansion.mark);

    // we want to be able to handle e.g., `concat!("foo", "bar")`
    cx.expander().visit_expr(&mut expr);
    Err(match expr.node {
        ast::ExprKind::Lit(ref l) => match l.node {
            ast::LitKind::Str(s, style) => return Ok(respan(expr.span, (s, style))),
            ast::LitKind::Err(_) => None,
            _ => Some(cx.struct_span_err(l.span, err_msg))
        },
        ast::ExprKind::Err => None,
        _ => Some(cx.struct_span_err(expr.span, err_msg))
    })
}

pub fn expr_to_string(cx: &mut ExtCtxt<'_>, expr: P<ast::Expr>, err_msg: &str)
                      -> Option<(Symbol, ast::StrStyle)> {
    expr_to_spanned_string(cx, expr, err_msg)
        .map_err(|err| err.map(|mut err| err.emit()))
        .ok()
        .map(|s| s.node)
}

/// Non-fatally assert that `tts` is empty. Note that this function
/// returns even when `tts` is non-empty, macros that *need* to stop
/// compilation should call
/// `cx.parse_sess.span_diagnostic.abort_if_errors()` (this should be
/// done as rarely as possible).
pub fn check_zero_tts(cx: &ExtCtxt<'_>,
                      sp: Span,
                      tts: &[tokenstream::TokenTree],
                      name: &str) {
    if !tts.is_empty() {
        cx.span_err(sp, &format!("{} takes no arguments", name));
    }
}

/// Interpreting `tts` as a comma-separated sequence of expressions,
/// expect exactly one string literal, or emit an error and return `None`.
pub fn get_single_str_from_tts(cx: &mut ExtCtxt<'_>,
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

/// Extracts comma-separated expressions from `tts`. If there is a
/// parsing error, emit a non-fatal error and return `None`.
pub fn get_exprs_from_tts(cx: &mut ExtCtxt<'_>,
                          sp: Span,
                          tts: &[tokenstream::TokenTree]) -> Option<Vec<P<ast::Expr>>> {
    let mut p = cx.new_parser_from_tts(tts);
    let mut es = Vec::new();
    while p.token != token::Eof {
        let mut expr = panictry!(p.parse_expr());
        cx.expander().visit_expr(&mut expr);
        es.push(expr);
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
