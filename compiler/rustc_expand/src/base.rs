#![deny(rustc::untranslatable_diagnostic)]

use crate::errors;
use crate::expand::{self, AstFragment, Invocation};
use crate::module::DirOwnership;

use rustc_ast::attr::MarkedAttrs;
use rustc_ast::mut_visit::DummyAstNode;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Nonterminal};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::visit::{AssocCtxt, Visitor};
use rustc_ast::{self as ast, AttrVec, Attribute, HasAttrs, Item, NodeId, PatKind};
use rustc_attr::{self as attr, Deprecation, Stability};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::{self, Lrc};
use rustc_errors::{
    Applicability, DiagnosticBuilder, ErrorGuaranteed, IntoDiagnostic, MultiSpan, PResult,
};
use rustc_lint_defs::builtin::PROC_MACRO_BACK_COMPAT;
use rustc_lint_defs::{BufferedEarlyLint, BuiltinLintDiagnostics, RegisteredTools};
use rustc_parse::{self, parser, MACRO_ARGUMENTS};
use rustc_session::errors::report_lit_error;
use rustc_session::{parse::ParseSess, Limit, Session};
use rustc_span::def_id::{CrateNum, DefId, LocalDefId};
use rustc_span::edition::Edition;
use rustc_span::hygiene::{AstPass, ExpnData, ExpnKind, LocalExpnId};
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{BytePos, FileName, Span, DUMMY_SP};
use smallvec::{smallvec, SmallVec};
use std::default::Default;
use std::iter;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use thin_vec::ThinVec;

pub(crate) use rustc_span::hygiene::MacroKind;

// When adding new variants, make sure to
// adjust the `visit_*` / `flat_map_*` calls in `InvocationCollector`
// to use `assign_id!`
#[derive(Debug, Clone)]
pub enum Annotatable {
    Item(P<ast::Item>),
    TraitItem(P<ast::AssocItem>),
    ImplItem(P<ast::AssocItem>),
    ForeignItem(P<ast::ForeignItem>),
    Stmt(P<ast::Stmt>),
    Expr(P<ast::Expr>),
    Arm(ast::Arm),
    ExprField(ast::ExprField),
    PatField(ast::PatField),
    GenericParam(ast::GenericParam),
    Param(ast::Param),
    FieldDef(ast::FieldDef),
    Variant(ast::Variant),
    Crate(ast::Crate),
}

impl Annotatable {
    pub fn span(&self) -> Span {
        match self {
            Annotatable::Item(item) => item.span,
            Annotatable::TraitItem(trait_item) => trait_item.span,
            Annotatable::ImplItem(impl_item) => impl_item.span,
            Annotatable::ForeignItem(foreign_item) => foreign_item.span,
            Annotatable::Stmt(stmt) => stmt.span,
            Annotatable::Expr(expr) => expr.span,
            Annotatable::Arm(arm) => arm.span,
            Annotatable::ExprField(field) => field.span,
            Annotatable::PatField(fp) => fp.pat.span,
            Annotatable::GenericParam(gp) => gp.ident.span,
            Annotatable::Param(p) => p.span,
            Annotatable::FieldDef(sf) => sf.span,
            Annotatable::Variant(v) => v.span,
            Annotatable::Crate(c) => c.spans.inner_span,
        }
    }

    pub fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        match self {
            Annotatable::Item(item) => item.visit_attrs(f),
            Annotatable::TraitItem(trait_item) => trait_item.visit_attrs(f),
            Annotatable::ImplItem(impl_item) => impl_item.visit_attrs(f),
            Annotatable::ForeignItem(foreign_item) => foreign_item.visit_attrs(f),
            Annotatable::Stmt(stmt) => stmt.visit_attrs(f),
            Annotatable::Expr(expr) => expr.visit_attrs(f),
            Annotatable::Arm(arm) => arm.visit_attrs(f),
            Annotatable::ExprField(field) => field.visit_attrs(f),
            Annotatable::PatField(fp) => fp.visit_attrs(f),
            Annotatable::GenericParam(gp) => gp.visit_attrs(f),
            Annotatable::Param(p) => p.visit_attrs(f),
            Annotatable::FieldDef(sf) => sf.visit_attrs(f),
            Annotatable::Variant(v) => v.visit_attrs(f),
            Annotatable::Crate(c) => c.visit_attrs(f),
        }
    }

    pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) {
        match self {
            Annotatable::Item(item) => visitor.visit_item(item),
            Annotatable::TraitItem(item) => visitor.visit_assoc_item(item, AssocCtxt::Trait),
            Annotatable::ImplItem(item) => visitor.visit_assoc_item(item, AssocCtxt::Impl),
            Annotatable::ForeignItem(foreign_item) => visitor.visit_foreign_item(foreign_item),
            Annotatable::Stmt(stmt) => visitor.visit_stmt(stmt),
            Annotatable::Expr(expr) => visitor.visit_expr(expr),
            Annotatable::Arm(arm) => visitor.visit_arm(arm),
            Annotatable::ExprField(field) => visitor.visit_expr_field(field),
            Annotatable::PatField(fp) => visitor.visit_pat_field(fp),
            Annotatable::GenericParam(gp) => visitor.visit_generic_param(gp),
            Annotatable::Param(p) => visitor.visit_param(p),
            Annotatable::FieldDef(sf) => visitor.visit_field_def(sf),
            Annotatable::Variant(v) => visitor.visit_variant(v),
            Annotatable::Crate(c) => visitor.visit_crate(c),
        }
    }

    pub fn to_tokens(&self) -> TokenStream {
        match self {
            Annotatable::Item(node) => TokenStream::from_ast(node),
            Annotatable::TraitItem(node) | Annotatable::ImplItem(node) => {
                TokenStream::from_ast(node)
            }
            Annotatable::ForeignItem(node) => TokenStream::from_ast(node),
            Annotatable::Stmt(node) => {
                assert!(!matches!(node.kind, ast::StmtKind::Empty));
                TokenStream::from_ast(node)
            }
            Annotatable::Expr(node) => TokenStream::from_ast(node),
            Annotatable::Arm(..)
            | Annotatable::ExprField(..)
            | Annotatable::PatField(..)
            | Annotatable::GenericParam(..)
            | Annotatable::Param(..)
            | Annotatable::FieldDef(..)
            | Annotatable::Variant(..)
            | Annotatable::Crate(..) => panic!("unexpected annotatable"),
        }
    }

    pub fn expect_item(self) -> P<ast::Item> {
        match self {
            Annotatable::Item(i) => i,
            _ => panic!("expected Item"),
        }
    }

    pub fn expect_trait_item(self) -> P<ast::AssocItem> {
        match self {
            Annotatable::TraitItem(i) => i,
            _ => panic!("expected Item"),
        }
    }

    pub fn expect_impl_item(self) -> P<ast::AssocItem> {
        match self {
            Annotatable::ImplItem(i) => i,
            _ => panic!("expected Item"),
        }
    }

    pub fn expect_foreign_item(self) -> P<ast::ForeignItem> {
        match self {
            Annotatable::ForeignItem(i) => i,
            _ => panic!("expected foreign item"),
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

    pub fn expect_arm(self) -> ast::Arm {
        match self {
            Annotatable::Arm(arm) => arm,
            _ => panic!("expected match arm"),
        }
    }

    pub fn expect_expr_field(self) -> ast::ExprField {
        match self {
            Annotatable::ExprField(field) => field,
            _ => panic!("expected field"),
        }
    }

    pub fn expect_pat_field(self) -> ast::PatField {
        match self {
            Annotatable::PatField(fp) => fp,
            _ => panic!("expected field pattern"),
        }
    }

    pub fn expect_generic_param(self) -> ast::GenericParam {
        match self {
            Annotatable::GenericParam(gp) => gp,
            _ => panic!("expected generic parameter"),
        }
    }

    pub fn expect_param(self) -> ast::Param {
        match self {
            Annotatable::Param(param) => param,
            _ => panic!("expected parameter"),
        }
    }

    pub fn expect_field_def(self) -> ast::FieldDef {
        match self {
            Annotatable::FieldDef(sf) => sf,
            _ => panic!("expected struct field"),
        }
    }

    pub fn expect_variant(self) -> ast::Variant {
        match self {
            Annotatable::Variant(v) => v,
            _ => panic!("expected variant"),
        }
    }

    pub fn expect_crate(self) -> ast::Crate {
        match self {
            Annotatable::Crate(krate) => krate,
            _ => panic!("expected krate"),
        }
    }
}

/// Result of an expansion that may need to be retried.
/// Consider using this for non-`MultiItemModifier` expanders as well.
pub enum ExpandResult<T, U> {
    /// Expansion produced a result (possibly dummy).
    Ready(T),
    /// Expansion could not produce a result and needs to be retried.
    Retry(U),
}

pub trait MultiItemModifier {
    /// `meta_item` is the attribute, and `item` is the item being modified.
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
        is_derive_const: bool,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable>;
}

impl<F> MultiItemModifier for F
where
    F: Fn(&mut ExtCtxt<'_>, Span, &ast::MetaItem, Annotatable) -> Vec<Annotatable>,
{
    fn expand(
        &self,
        ecx: &mut ExtCtxt<'_>,
        span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
        _is_derive_const: bool,
    ) -> ExpandResult<Vec<Annotatable>, Annotatable> {
        ExpandResult::Ready(self(ecx, span, meta_item, item))
    }
}

pub trait BangProcMacro {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        ts: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed>;
}

impl<F> BangProcMacro for F
where
    F: Fn(TokenStream) -> TokenStream,
{
    fn expand<'cx>(
        &self,
        _ecx: &'cx mut ExtCtxt<'_>,
        _span: Span,
        ts: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        // FIXME setup implicit context in TLS before calling self.
        Ok(self(ts))
    }
}

pub trait AttrProcMacro {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed>;
}

impl<F> AttrProcMacro for F
where
    F: Fn(TokenStream, TokenStream) -> TokenStream,
{
    fn expand<'cx>(
        &self,
        _ecx: &'cx mut ExtCtxt<'_>,
        _span: Span,
        annotation: TokenStream,
        annotated: TokenStream,
    ) -> Result<TokenStream, ErrorGuaranteed> {
        // FIXME setup implicit context in TLS before calling self.
        Ok(self(annotation, annotated))
    }
}

/// Represents a thing that maps token trees to Macro Results
pub trait TTMacroExpander {
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
    ) -> Box<dyn MacResult + 'cx>;
}

pub type MacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt<'_>, Span, TokenStream) -> Box<dyn MacResult + 'cx>;

impl<F> TTMacroExpander for F
where
    F: for<'cx> Fn(&'cx mut ExtCtxt<'_>, Span, TokenStream) -> Box<dyn MacResult + 'cx>,
{
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
    ) -> Box<dyn MacResult + 'cx> {
        self(ecx, span, input)
    }
}

// Use a macro because forwarding to a simple function has type system issues
macro_rules! make_stmts_default {
    ($me:expr) => {
        $me.make_expr().map(|e| {
            smallvec![ast::Stmt {
                id: ast::DUMMY_NODE_ID,
                span: e.span,
                kind: ast::StmtKind::Expr(e),
            }]
        })
    };
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
    fn make_impl_items(self: Box<Self>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        None
    }

    /// Creates zero or more trait items.
    fn make_trait_items(self: Box<Self>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        None
    }

    /// Creates zero or more items in an `extern {}` block
    fn make_foreign_items(self: Box<Self>) -> Option<SmallVec<[P<ast::ForeignItem>; 1]>> {
        None
    }

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

    fn make_arms(self: Box<Self>) -> Option<SmallVec<[ast::Arm; 1]>> {
        None
    }

    fn make_expr_fields(self: Box<Self>) -> Option<SmallVec<[ast::ExprField; 1]>> {
        None
    }

    fn make_pat_fields(self: Box<Self>) -> Option<SmallVec<[ast::PatField; 1]>> {
        None
    }

    fn make_generic_params(self: Box<Self>) -> Option<SmallVec<[ast::GenericParam; 1]>> {
        None
    }

    fn make_params(self: Box<Self>) -> Option<SmallVec<[ast::Param; 1]>> {
        None
    }

    fn make_field_defs(self: Box<Self>) -> Option<SmallVec<[ast::FieldDef; 1]>> {
        None
    }

    fn make_variants(self: Box<Self>) -> Option<SmallVec<[ast::Variant; 1]>> {
        None
    }

    fn make_crate(self: Box<Self>) -> Option<ast::Crate> {
        // Fn-like macros cannot produce a crate.
        unreachable!()
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
    impl_items: SmallVec<[P<ast::AssocItem>; 1]>,
    trait_items: SmallVec<[P<ast::AssocItem>; 1]>,
    foreign_items: SmallVec<[P<ast::ForeignItem>; 1]>,
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

    fn make_impl_items(self: Box<Self>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        self.impl_items
    }

    fn make_trait_items(self: Box<Self>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        self.trait_items
    }

    fn make_foreign_items(self: Box<Self>) -> Option<SmallVec<[P<ast::ForeignItem>; 1]>> {
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
            if matches!(e.kind, ast::ExprKind::Lit(_) | ast::ExprKind::IncludedBytes(_)) {
                return Some(P(ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    span: e.span,
                    kind: PatKind::Lit(e),
                    tokens: None,
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
    is_error: bool,
    span: Span,
}

impl DummyResult {
    /// Creates a default MacResult that can be anything.
    ///
    /// Use this as a return value after hitting any errors and
    /// calling `span_err`.
    pub fn any(span: Span) -> Box<dyn MacResult + 'static> {
        Box::new(DummyResult { is_error: true, span })
    }

    /// Same as `any`, but must be a valid fragment, not error.
    pub fn any_valid(span: Span) -> Box<dyn MacResult + 'static> {
        Box::new(DummyResult { is_error: false, span })
    }

    /// A plain dummy expression.
    pub fn raw_expr(sp: Span, is_error: bool) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            kind: if is_error { ast::ExprKind::Err } else { ast::ExprKind::Tup(ThinVec::new()) },
            span: sp,
            attrs: ast::AttrVec::new(),
            tokens: None,
        })
    }

    /// A plain dummy pattern.
    pub fn raw_pat(sp: Span) -> ast::Pat {
        ast::Pat { id: ast::DUMMY_NODE_ID, kind: PatKind::Wild, span: sp, tokens: None }
    }

    /// A plain dummy type.
    pub fn raw_ty(sp: Span, is_error: bool) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            kind: if is_error { ast::TyKind::Err } else { ast::TyKind::Tup(ThinVec::new()) },
            span: sp,
            tokens: None,
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
        Some(SmallVec::new())
    }

    fn make_impl_items(self: Box<DummyResult>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        Some(SmallVec::new())
    }

    fn make_trait_items(self: Box<DummyResult>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        Some(SmallVec::new())
    }

    fn make_foreign_items(self: Box<Self>) -> Option<SmallVec<[P<ast::ForeignItem>; 1]>> {
        Some(SmallVec::new())
    }

    fn make_stmts(self: Box<DummyResult>) -> Option<SmallVec<[ast::Stmt; 1]>> {
        Some(smallvec![ast::Stmt {
            id: ast::DUMMY_NODE_ID,
            kind: ast::StmtKind::Expr(DummyResult::raw_expr(self.span, self.is_error)),
            span: self.span,
        }])
    }

    fn make_ty(self: Box<DummyResult>) -> Option<P<ast::Ty>> {
        Some(DummyResult::raw_ty(self.span, self.is_error))
    }

    fn make_arms(self: Box<DummyResult>) -> Option<SmallVec<[ast::Arm; 1]>> {
        Some(SmallVec::new())
    }

    fn make_expr_fields(self: Box<DummyResult>) -> Option<SmallVec<[ast::ExprField; 1]>> {
        Some(SmallVec::new())
    }

    fn make_pat_fields(self: Box<DummyResult>) -> Option<SmallVec<[ast::PatField; 1]>> {
        Some(SmallVec::new())
    }

    fn make_generic_params(self: Box<DummyResult>) -> Option<SmallVec<[ast::GenericParam; 1]>> {
        Some(SmallVec::new())
    }

    fn make_params(self: Box<DummyResult>) -> Option<SmallVec<[ast::Param; 1]>> {
        Some(SmallVec::new())
    }

    fn make_field_defs(self: Box<DummyResult>) -> Option<SmallVec<[ast::FieldDef; 1]>> {
        Some(SmallVec::new())
    }

    fn make_variants(self: Box<DummyResult>) -> Option<SmallVec<[ast::Variant; 1]>> {
        Some(SmallVec::new())
    }

    fn make_crate(self: Box<DummyResult>) -> Option<ast::Crate> {
        Some(DummyAstNode::dummy())
    }
}

/// A syntax extension kind.
pub enum SyntaxExtensionKind {
    /// A token-based function-like macro.
    Bang(
        /// An expander with signature TokenStream -> TokenStream.
        Box<dyn BangProcMacro + sync::Sync + sync::Send>,
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
    NonMacroAttr,

    /// A token-based derive macro.
    Derive(
        /// An expander with signature TokenStream -> TokenStream.
        /// The produced TokenSteam is appended to the input TokenSteam.
        ///
        /// FIXME: The text above describes how this should work. Currently it
        /// is handled identically to `LegacyDerive`. It should be migrated to
        /// a token-based representation like `Bang` and `Attr`, instead of
        /// using `MultiItemModifier`.
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
    /// Span of the macro definition.
    pub span: Span,
    /// List of unstable features that are treated as stable inside this macro.
    pub allow_internal_unstable: Option<Lrc<[Symbol]>>,
    /// The macro's stability info.
    pub stability: Option<Stability>,
    /// The macro's deprecation info.
    pub deprecation: Option<Deprecation>,
    /// Names of helper attributes registered by this macro.
    pub helper_attrs: Vec<Symbol>,
    /// Edition of the crate in which this macro is defined.
    pub edition: Edition,
    /// Built-in macros have a couple of special properties like availability
    /// in `#[no_implicit_prelude]` modules, so we have to keep this flag.
    pub builtin_name: Option<Symbol>,
    /// Suppresses the `unsafe_code` lint for code produced by this macro.
    pub allow_internal_unsafe: bool,
    /// Enables the macro helper hack (`ident!(...)` -> `$crate::ident!(...)`) for this macro.
    pub local_inner_macros: bool,
    /// Should debuginfo for the macro be collapsed to the outermost expansion site (in other
    /// words, was the macro definition annotated with `#[collapse_debuginfo]`)?
    pub collapse_debuginfo: bool,
}

impl SyntaxExtension {
    /// Returns which kind of macro calls this syntax extension.
    pub fn macro_kind(&self) -> MacroKind {
        match self.kind {
            SyntaxExtensionKind::Bang(..) | SyntaxExtensionKind::LegacyBang(..) => MacroKind::Bang,
            SyntaxExtensionKind::Attr(..)
            | SyntaxExtensionKind::LegacyAttr(..)
            | SyntaxExtensionKind::NonMacroAttr => MacroKind::Attr,
            SyntaxExtensionKind::Derive(..) | SyntaxExtensionKind::LegacyDerive(..) => {
                MacroKind::Derive
            }
        }
    }

    /// Constructs a syntax extension with default properties.
    pub fn default(kind: SyntaxExtensionKind, edition: Edition) -> SyntaxExtension {
        SyntaxExtension {
            span: DUMMY_SP,
            allow_internal_unstable: None,
            stability: None,
            deprecation: None,
            helper_attrs: Vec::new(),
            edition,
            builtin_name: None,
            kind,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            collapse_debuginfo: false,
        }
    }

    /// Constructs a syntax extension with the given properties
    /// and other properties converted from attributes.
    pub fn new(
        sess: &Session,
        kind: SyntaxExtensionKind,
        span: Span,
        helper_attrs: Vec<Symbol>,
        edition: Edition,
        name: Symbol,
        attrs: &[ast::Attribute],
    ) -> SyntaxExtension {
        let allow_internal_unstable =
            attr::allow_internal_unstable(sess, &attrs).collect::<Vec<Symbol>>();

        let allow_internal_unsafe = attr::contains_name(attrs, sym::allow_internal_unsafe);
        let local_inner_macros = attr::find_by_name(attrs, sym::macro_export)
            .and_then(|macro_export| macro_export.meta_item_list())
            .map_or(false, |l| attr::list_contains_name(&l, sym::local_inner_macros));
        let collapse_debuginfo = attr::contains_name(attrs, sym::collapse_debuginfo);
        tracing::debug!(?local_inner_macros, ?collapse_debuginfo, ?allow_internal_unsafe);

        let (builtin_name, helper_attrs) = attr::find_by_name(attrs, sym::rustc_builtin_macro)
            .map(|attr| {
                // Override `helper_attrs` passed above if it's a built-in macro,
                // marking `proc_macro_derive` macros as built-in is not a realistic use case.
                parse_macro_name_and_helper_attrs(sess.diagnostic(), attr, "built-in").map_or_else(
                    || (Some(name), Vec::new()),
                    |(name, helper_attrs)| (Some(name), helper_attrs),
                )
            })
            .unwrap_or_else(|| (None, helper_attrs));
        let (stability, const_stability, body_stability) = attr::find_stability(&sess, attrs, span);
        if let Some((_, sp)) = const_stability {
            sess.emit_err(errors::MacroConstStability {
                span: sp,
                head_span: sess.source_map().guess_head_span(span),
            });
        }
        if let Some((_, sp)) = body_stability {
            sess.emit_err(errors::MacroBodyStability {
                span: sp,
                head_span: sess.source_map().guess_head_span(span),
            });
        }

        SyntaxExtension {
            kind,
            span,
            allow_internal_unstable: (!allow_internal_unstable.is_empty())
                .then(|| allow_internal_unstable.into()),
            stability: stability.map(|(s, _)| s),
            deprecation: attr::find_deprecation(&sess, attrs).map(|(d, _)| d),
            helper_attrs,
            edition,
            builtin_name,
            allow_internal_unsafe,
            local_inner_macros,
            collapse_debuginfo,
        }
    }

    pub fn dummy_bang(edition: Edition) -> SyntaxExtension {
        fn expander<'cx>(
            _: &'cx mut ExtCtxt<'_>,
            span: Span,
            _: TokenStream,
        ) -> Box<dyn MacResult + 'cx> {
            DummyResult::any(span)
        }
        SyntaxExtension::default(SyntaxExtensionKind::LegacyBang(Box::new(expander)), edition)
    }

    pub fn dummy_derive(edition: Edition) -> SyntaxExtension {
        fn expander(
            _: &mut ExtCtxt<'_>,
            _: Span,
            _: &ast::MetaItem,
            _: Annotatable,
        ) -> Vec<Annotatable> {
            Vec::new()
        }
        SyntaxExtension::default(SyntaxExtensionKind::Derive(Box::new(expander)), edition)
    }

    pub fn non_macro_attr(edition: Edition) -> SyntaxExtension {
        SyntaxExtension::default(SyntaxExtensionKind::NonMacroAttr, edition)
    }

    pub fn expn_data(
        &self,
        parent: LocalExpnId,
        call_site: Span,
        descr: Symbol,
        macro_def_id: Option<DefId>,
        parent_module: Option<DefId>,
    ) -> ExpnData {
        ExpnData::new(
            ExpnKind::Macro(self.macro_kind(), descr),
            parent.to_expn_id(),
            call_site,
            self.span,
            self.allow_internal_unstable.clone(),
            self.edition,
            macro_def_id,
            parent_module,
            self.allow_internal_unsafe,
            self.local_inner_macros,
            self.collapse_debuginfo,
        )
    }
}

/// Error type that denotes indeterminacy.
pub struct Indeterminate;

pub type DeriveResolutions = Vec<(ast::Path, Annotatable, Option<Lrc<SyntaxExtension>>, bool)>;

pub trait ResolverExpand {
    fn next_node_id(&mut self) -> NodeId;
    fn invocation_parent(&self, id: LocalExpnId) -> LocalDefId;

    fn resolve_dollar_crates(&mut self);
    fn visit_ast_fragment_with_placeholders(
        &mut self,
        expn_id: LocalExpnId,
        fragment: &AstFragment,
    );
    fn register_builtin_macro(&mut self, name: Symbol, ext: SyntaxExtensionKind);

    fn expansion_for_ast_pass(
        &mut self,
        call_site: Span,
        pass: AstPass,
        features: &[Symbol],
        parent_module_id: Option<NodeId>,
    ) -> LocalExpnId;

    fn resolve_imports(&mut self);

    fn resolve_macro_invocation(
        &mut self,
        invoc: &Invocation,
        eager_expansion_root: LocalExpnId,
        force: bool,
    ) -> Result<Lrc<SyntaxExtension>, Indeterminate>;

    fn record_macro_rule_usage(&mut self, mac_id: NodeId, rule_index: usize);

    fn check_unused_macros(&mut self);

    // Resolver interfaces for specific built-in macros.
    /// Does `#[derive(...)]` attribute with the given `ExpnId` have built-in `Copy` inside it?
    fn has_derive_copy(&self, expn_id: LocalExpnId) -> bool;
    /// Resolve paths inside the `#[derive(...)]` attribute with the given `ExpnId`.
    fn resolve_derives(
        &mut self,
        expn_id: LocalExpnId,
        force: bool,
        derive_paths: &dyn Fn() -> DeriveResolutions,
    ) -> Result<(), Indeterminate>;
    /// Take resolutions for paths inside the `#[derive(...)]` attribute with the given `ExpnId`
    /// back from resolver.
    fn take_derive_resolutions(&mut self, expn_id: LocalExpnId) -> Option<DeriveResolutions>;
    /// Path resolution logic for `#[cfg_accessible(path)]`.
    fn cfg_accessible(
        &mut self,
        expn_id: LocalExpnId,
        path: &ast::Path,
    ) -> Result<bool, Indeterminate>;

    /// Decodes the proc-macro quoted span in the specified crate, with the specified id.
    /// No caching is performed.
    fn get_proc_macro_quoted_span(&self, krate: CrateNum, id: usize) -> Span;

    /// The order of items in the HIR is unrelated to the order of
    /// items in the AST. However, we generate proc macro harnesses
    /// based on the AST order, and later refer to these harnesses
    /// from the HIR. This field keeps track of the order in which
    /// we generated proc macros harnesses, so that we can map
    /// HIR proc macros items back to their harness items.
    fn declare_proc_macro(&mut self, id: NodeId);

    /// Tools registered with `#![register_tool]` and used by tool attributes and lints.
    fn registered_tools(&self) -> &RegisteredTools;
}

pub trait LintStoreExpand {
    fn pre_expansion_lint(
        &self,
        sess: &Session,
        registered_tools: &RegisteredTools,
        node_id: NodeId,
        attrs: &[Attribute],
        items: &[P<Item>],
        name: Symbol,
    );
}

type LintStoreExpandDyn<'a> = Option<&'a (dyn LintStoreExpand + 'a)>;

#[derive(Clone, Default)]
pub struct ModuleData {
    /// Path to the module starting from the crate name, like `my_crate::foo::bar`.
    pub mod_path: Vec<Ident>,
    /// Stack of paths to files loaded by out-of-line module items,
    /// used to detect and report recursive module inclusions.
    pub file_path_stack: Vec<PathBuf>,
    /// Directory to search child module files in,
    /// often (but not necessarily) the parent of the top file path on the `file_path_stack`.
    pub dir_path: PathBuf,
}

impl ModuleData {
    pub fn with_dir_path(&self, dir_path: PathBuf) -> ModuleData {
        ModuleData {
            mod_path: self.mod_path.clone(),
            file_path_stack: self.file_path_stack.clone(),
            dir_path,
        }
    }
}

#[derive(Clone)]
pub struct ExpansionData {
    pub id: LocalExpnId,
    pub depth: usize,
    pub module: Rc<ModuleData>,
    pub dir_ownership: DirOwnership,
    pub prior_type_ascription: Option<(Span, bool)>,
    /// Some parent node that is close to this macro call
    pub lint_node_id: NodeId,
    pub is_trailing_mac: bool,
}

/// One of these is made during expansion and incrementally updated as we go;
/// when a macro expansion occurs, the resulting nodes have the `backtrace()
/// -> expn_data` of their expansion context stored into their span.
pub struct ExtCtxt<'a> {
    pub sess: &'a Session,
    pub ecfg: expand::ExpansionConfig<'a>,
    pub num_standard_library_imports: usize,
    pub reduced_recursion_limit: Option<Limit>,
    pub root_path: PathBuf,
    pub resolver: &'a mut dyn ResolverExpand,
    pub current_expansion: ExpansionData,
    /// Error recovery mode entered when expansion is stuck
    /// (or during eager expansion, but that's a hack).
    pub force_mode: bool,
    pub expansions: FxIndexMap<Span, Vec<String>>,
    /// Used for running pre-expansion lints on freshly loaded modules.
    pub(super) lint_store: LintStoreExpandDyn<'a>,
    /// Used for storing lints generated during expansion, like `NAMED_ARGUMENTS_USED_POSITIONALLY`
    pub buffered_early_lint: Vec<BufferedEarlyLint>,
    /// When we 'expand' an inert attribute, we leave it
    /// in the AST, but insert it here so that we know
    /// not to expand it again.
    pub(super) expanded_inert_attrs: MarkedAttrs,
}

impl<'a> ExtCtxt<'a> {
    pub fn new(
        sess: &'a Session,
        ecfg: expand::ExpansionConfig<'a>,
        resolver: &'a mut dyn ResolverExpand,
        lint_store: LintStoreExpandDyn<'a>,
    ) -> ExtCtxt<'a> {
        ExtCtxt {
            sess,
            ecfg,
            num_standard_library_imports: 0,
            reduced_recursion_limit: None,
            resolver,
            lint_store,
            root_path: PathBuf::new(),
            current_expansion: ExpansionData {
                id: LocalExpnId::ROOT,
                depth: 0,
                module: Default::default(),
                dir_ownership: DirOwnership::Owned { relative: None },
                prior_type_ascription: None,
                lint_node_id: ast::CRATE_NODE_ID,
                is_trailing_mac: false,
            },
            force_mode: false,
            expansions: FxIndexMap::default(),
            expanded_inert_attrs: MarkedAttrs::new(),
            buffered_early_lint: vec![],
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
    pub fn new_parser_from_tts(&self, stream: TokenStream) -> parser::Parser<'a> {
        rustc_parse::stream_to_parser(&self.sess.parse_sess, stream, MACRO_ARGUMENTS)
    }
    pub fn source_map(&self) -> &'a SourceMap {
        self.sess.parse_sess.source_map()
    }
    pub fn parse_sess(&self) -> &'a ParseSess {
        &self.sess.parse_sess
    }
    pub fn call_site(&self) -> Span {
        self.current_expansion.id.expn_data().call_site
    }

    /// Returns the current expansion kind's description.
    pub(crate) fn expansion_descr(&self) -> String {
        let expn_data = self.current_expansion.id.expn_data();
        expn_data.kind.descr()
    }

    /// Equivalent of `Span::def_site` from the proc macro API,
    /// except that the location is taken from the span passed as an argument.
    pub fn with_def_site_ctxt(&self, span: Span) -> Span {
        span.with_def_site_ctxt(self.current_expansion.id.to_expn_id())
    }

    /// Equivalent of `Span::call_site` from the proc macro API,
    /// except that the location is taken from the span passed as an argument.
    pub fn with_call_site_ctxt(&self, span: Span) -> Span {
        span.with_call_site_ctxt(self.current_expansion.id.to_expn_id())
    }

    /// Equivalent of `Span::mixed_site` from the proc macro API,
    /// except that the location is taken from the span passed as an argument.
    pub fn with_mixed_site_ctxt(&self, span: Span) -> Span {
        span.with_mixed_site_ctxt(self.current_expansion.id.to_expn_id())
    }

    /// Returns span for the macro which originally caused the current expansion to happen.
    ///
    /// Stops backtracing at include! boundary.
    pub fn expansion_cause(&self) -> Option<Span> {
        self.current_expansion.id.expansion_cause()
    }

    #[rustc_lint_diagnostics]
    pub fn struct_span_err<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: &str,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        self.sess.parse_sess.span_diagnostic.struct_span_err(sp, msg)
    }

    pub fn create_err(
        &self,
        err: impl IntoDiagnostic<'a>,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        self.sess.create_err(err)
    }

    pub fn emit_err(&self, err: impl IntoDiagnostic<'a>) -> ErrorGuaranteed {
        self.sess.emit_err(err)
    }

    /// Emit `msg` attached to `sp`, without immediately stopping
    /// compilation.
    ///
    /// Compilation will be stopped in the near future (at the end of
    /// the macro expansion phase).
    #[rustc_lint_diagnostics]
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.sess.parse_sess.span_diagnostic.span_err(sp, msg);
    }
    #[rustc_lint_diagnostics]
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.sess.parse_sess.span_diagnostic.span_warn(sp, msg);
    }
    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.sess.parse_sess.span_diagnostic.span_bug(sp, msg);
    }
    pub fn trace_macros_diag(&mut self) {
        for (span, notes) in self.expansions.iter() {
            let mut db = self.sess.parse_sess.create_note(errors::TraceMacro { span: *span });
            for note in notes {
                db.note(note);
            }
            db.emit();
        }
        // Fixme: does this result in errors?
        self.expansions.clear();
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.sess.parse_sess.span_diagnostic.bug(msg);
    }
    pub fn trace_macros(&self) -> bool {
        self.ecfg.trace_mac
    }
    pub fn set_trace_macros(&mut self, x: bool) {
        self.ecfg.trace_mac = x
    }
    pub fn std_path(&self, components: &[Symbol]) -> Vec<Ident> {
        let def_site = self.with_def_site_ctxt(DUMMY_SP);
        iter::once(Ident::new(kw::DollarCrate, def_site))
            .chain(components.iter().map(|&s| Ident::with_dummy_span(s)))
            .collect()
    }
    pub fn def_site_path(&self, components: &[Symbol]) -> Vec<Ident> {
        let def_site = self.with_def_site_ctxt(DUMMY_SP);
        components.iter().map(|&s| Ident::new(s, def_site)).collect()
    }

    pub fn check_unused_macros(&mut self) {
        self.resolver.check_unused_macros();
    }
}

/// Resolves a `path` mentioned inside Rust code, returning an absolute path.
///
/// This unifies the logic used for resolving `include_X!`.
pub fn resolve_path(
    parse_sess: &ParseSess,
    path: impl Into<PathBuf>,
    span: Span,
) -> PResult<'_, PathBuf> {
    let path = path.into();

    // Relative paths are resolved relative to the file in which they are found
    // after macro expansion (that is, they are unhygienic).
    if !path.is_absolute() {
        let callsite = span.source_callsite();
        let mut result = match parse_sess.source_map().span_to_filename(callsite) {
            FileName::Real(name) => name
                .into_local_path()
                .expect("attempting to resolve a file path in an external file"),
            FileName::DocTest(path, _) => path,
            other => {
                return Err(errors::ResolveRelativePath {
                    span,
                    path: parse_sess.source_map().filename_for_diagnostics(&other).to_string(),
                }
                .into_diagnostic(&parse_sess.span_diagnostic));
            }
        };
        result.pop();
        result.push(path);
        Ok(result)
    } else {
        Ok(path)
    }
}

/// Extracts a string literal from the macro expanded version of `expr`,
/// returning a diagnostic error of `err_msg` if `expr` is not a string literal.
/// The returned bool indicates whether an applicable suggestion has already been
/// added to the diagnostic to avoid emitting multiple suggestions. `Err(None)`
/// indicates that an ast error was encountered.
// FIXME(Nilstrieb) Make this function setup translatable
#[allow(rustc::untranslatable_diagnostic)]
pub fn expr_to_spanned_string<'a>(
    cx: &'a mut ExtCtxt<'_>,
    expr: P<ast::Expr>,
    err_msg: &str,
) -> Result<(Symbol, ast::StrStyle, Span), Option<(DiagnosticBuilder<'a, ErrorGuaranteed>, bool)>> {
    // Perform eager expansion on the expression.
    // We want to be able to handle e.g., `concat!("foo", "bar")`.
    let expr = cx.expander().fully_expand_fragment(AstFragment::Expr(expr)).make_expr();

    Err(match expr.kind {
        ast::ExprKind::Lit(token_lit) => match ast::LitKind::from_token_lit(token_lit) {
            Ok(ast::LitKind::Str(s, style)) => return Ok((s, style, expr.span)),
            Ok(ast::LitKind::ByteStr(..)) => {
                let mut err = cx.struct_span_err(expr.span, err_msg);
                let span = expr.span.shrink_to_lo();
                err.span_suggestion(
                    span.with_hi(span.lo() + BytePos(1)),
                    "consider removing the leading `b`",
                    "",
                    Applicability::MaybeIncorrect,
                );
                Some((err, true))
            }
            Ok(ast::LitKind::Err) => None,
            Err(err) => {
                report_lit_error(&cx.sess.parse_sess, err, token_lit, expr.span);
                None
            }
            _ => Some((cx.struct_span_err(expr.span, err_msg), false)),
        },
        ast::ExprKind::Err => None,
        _ => Some((cx.struct_span_err(expr.span, err_msg), false)),
    })
}

/// Extracts a string literal from the macro expanded version of `expr`,
/// emitting `err_msg` if `expr` is not a string literal. This does not stop
/// compilation on error, merely emits a non-fatal error and returns `None`.
pub fn expr_to_string(
    cx: &mut ExtCtxt<'_>,
    expr: P<ast::Expr>,
    err_msg: &str,
) -> Option<(Symbol, ast::StrStyle)> {
    expr_to_spanned_string(cx, expr, err_msg)
        .map_err(|err| {
            err.map(|(mut err, _)| {
                err.emit();
            })
        })
        .ok()
        .map(|(symbol, style, _)| (symbol, style))
}

/// Non-fatally assert that `tts` is empty. Note that this function
/// returns even when `tts` is non-empty, macros that *need* to stop
/// compilation should call
/// `cx.parse_sess.span_diagnostic.abort_if_errors()` (this should be
/// done as rarely as possible).
pub fn check_zero_tts(cx: &ExtCtxt<'_>, span: Span, tts: TokenStream, name: &str) {
    if !tts.is_empty() {
        cx.emit_err(errors::TakesNoArguments { span, name });
    }
}

/// Parse an expression. On error, emit it, advancing to `Eof`, and return `None`.
pub fn parse_expr(p: &mut parser::Parser<'_>) -> Option<P<ast::Expr>> {
    match p.parse_expr() {
        Ok(e) => return Some(e),
        Err(mut err) => {
            err.emit();
        }
    }
    while p.token != token::Eof {
        p.bump();
    }
    None
}

/// Interpreting `tts` as a comma-separated sequence of expressions,
/// expect exactly one string literal, or emit an error and return `None`.
pub fn get_single_str_from_tts(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
    name: &str,
) -> Option<Symbol> {
    let mut p = cx.new_parser_from_tts(tts);
    if p.token == token::Eof {
        cx.emit_err(errors::OnlyOneArgument { span, name });
        return None;
    }
    let ret = parse_expr(&mut p)?;
    let _ = p.eat(&token::Comma);

    if p.token != token::Eof {
        cx.emit_err(errors::OnlyOneArgument { span, name });
    }
    expr_to_string(cx, ret, "argument must be a string literal").map(|(s, _)| s)
}

/// Extracts comma-separated expressions from `tts`.
/// On error, emit it, and return `None`.
pub fn get_exprs_from_tts(cx: &mut ExtCtxt<'_>, tts: TokenStream) -> Option<Vec<P<ast::Expr>>> {
    let mut p = cx.new_parser_from_tts(tts);
    let mut es = Vec::new();
    while p.token != token::Eof {
        let expr = parse_expr(&mut p)?;

        // Perform eager expansion on the expression.
        // We want to be able to handle e.g., `concat!("foo", "bar")`.
        let expr = cx.expander().fully_expand_fragment(AstFragment::Expr(expr)).make_expr();

        es.push(expr);
        if p.eat(&token::Comma) {
            continue;
        }
        if p.token != token::Eof {
            cx.emit_err(errors::ExpectedCommaInList { span: p.token.span });
            return None;
        }
    }
    Some(es)
}

pub fn parse_macro_name_and_helper_attrs(
    diag: &rustc_errors::Handler,
    attr: &Attribute,
    macro_type: &str,
) -> Option<(Symbol, Vec<Symbol>)> {
    // Once we've located the `#[proc_macro_derive]` attribute, verify
    // that it's of the form `#[proc_macro_derive(Foo)]` or
    // `#[proc_macro_derive(Foo, attributes(A, ..))]`
    let list = attr.meta_item_list()?;
    if list.len() != 1 && list.len() != 2 {
        diag.emit_err(errors::AttrNoArguments { span: attr.span });
        return None;
    }
    let Some(trait_attr) = list[0].meta_item() else {
        diag.emit_err(errors::NotAMetaItem {span: list[0].span()});
        return None;
    };
    let trait_ident = match trait_attr.ident() {
        Some(trait_ident) if trait_attr.is_word() => trait_ident,
        _ => {
            diag.emit_err(errors::OnlyOneWord { span: trait_attr.span });
            return None;
        }
    };

    if !trait_ident.name.can_be_raw() {
        diag.emit_err(errors::CannotBeNameOfMacro {
            span: trait_attr.span,
            trait_ident,
            macro_type,
        });
    }

    let attributes_attr = list.get(1);
    let proc_attrs: Vec<_> = if let Some(attr) = attributes_attr {
        if !attr.has_name(sym::attributes) {
            diag.emit_err(errors::ArgumentNotAttributes { span: attr.span() });
        }
        attr.meta_item_list()
            .unwrap_or_else(|| {
                diag.emit_err(errors::AttributesWrongForm { span: attr.span() });
                &[]
            })
            .iter()
            .filter_map(|attr| {
                let Some(attr) = attr.meta_item() else {
                    diag.emit_err(errors::AttributeMetaItem { span: attr.span() });
                    return None;
                };

                let ident = match attr.ident() {
                    Some(ident) if attr.is_word() => ident,
                    _ => {
                        diag.emit_err(errors::AttributeSingleWord { span: attr.span });
                        return None;
                    }
                };
                if !ident.name.can_be_raw() {
                    diag.emit_err(errors::HelperAttributeNameInvalid {
                        span: attr.span,
                        name: ident,
                    });
                }

                Some(ident.name)
            })
            .collect()
    } else {
        Vec::new()
    };

    Some((trait_ident.name, proc_attrs))
}

/// This nonterminal looks like some specific enums from
/// `proc-macro-hack` and `procedural-masquerade` crates.
/// We need to maintain some special pretty-printing behavior for them due to incorrect
/// asserts in old versions of those crates and their wide use in the ecosystem.
/// See issue #73345 for more details.
/// FIXME(#73933): Remove this eventually.
fn pretty_printing_compatibility_hack(item: &Item, sess: &ParseSess) -> bool {
    let name = item.ident.name;
    if name == sym::ProceduralMasqueradeDummyType {
        if let ast::ItemKind::Enum(enum_def, _) = &item.kind {
            if let [variant] = &*enum_def.variants {
                if variant.ident.name == sym::Input {
                    let filename = sess.source_map().span_to_filename(item.ident.span);
                    if let FileName::Real(real) = filename {
                        if let Some(c) = real
                            .local_path()
                            .unwrap_or(Path::new(""))
                            .components()
                            .flat_map(|c| c.as_os_str().to_str())
                            .find(|c| c.starts_with("rental") || c.starts_with("allsorts-rental"))
                        {
                            let crate_matches = if c.starts_with("allsorts-rental") {
                                true
                            } else {
                                let mut version = c.trim_start_matches("rental-").split('.');
                                version.next() == Some("0")
                                    && version.next() == Some("5")
                                    && version
                                        .next()
                                        .and_then(|c| c.parse::<u32>().ok())
                                        .map_or(false, |v| v < 6)
                            };

                            if crate_matches {
                                sess.buffer_lint_with_diagnostic(
                                        &PROC_MACRO_BACK_COMPAT,
                                        item.ident.span,
                                        ast::CRATE_NODE_ID,
                                        "using an old version of `rental`",
                                        BuiltinLintDiagnostics::ProcMacroBackCompat(
                                        "older versions of the `rental` crate will stop compiling in future versions of Rust; \
                                        please update to `rental` v0.5.6, or switch to one of the `rental` alternatives".to_string()
                                        )
                                    );
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

pub(crate) fn ann_pretty_printing_compatibility_hack(ann: &Annotatable, sess: &ParseSess) -> bool {
    let item = match ann {
        Annotatable::Item(item) => item,
        Annotatable::Stmt(stmt) => match &stmt.kind {
            ast::StmtKind::Item(item) => item,
            _ => return false,
        },
        _ => return false,
    };
    pretty_printing_compatibility_hack(item, sess)
}

pub(crate) fn nt_pretty_printing_compatibility_hack(nt: &Nonterminal, sess: &ParseSess) -> bool {
    let item = match nt {
        Nonterminal::NtItem(item) => item,
        Nonterminal::NtStmt(stmt) => match &stmt.kind {
            ast::StmtKind::Item(item) => item,
            _ => return false,
        },
        _ => return false,
    };
    pretty_printing_compatibility_hack(item, sess)
}
