use std::default::Default;
use std::iter;
use std::path::Component::Prefix;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use rustc_ast::attr::{AttributeExt, MarkedAttrs};
use rustc_ast::ptr::P;
use rustc_ast::token::MetaVarKind;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::visit::{AssocCtxt, Visitor};
use rustc_ast::{self as ast, AttrVec, Attribute, HasAttrs, Item, NodeId, PatKind};
use rustc_attr_data_structures::{AttributeKind, Deprecation, Stability, find_attr};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync;
use rustc_errors::{DiagCtxtHandle, ErrorGuaranteed, PResult};
use rustc_feature::Features;
use rustc_hir as hir;
use rustc_lint_defs::{BufferedEarlyLint, RegisteredTools};
use rustc_parse::MACRO_ARGUMENTS;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::config::CollapseMacroDebuginfo;
use rustc_session::parse::ParseSess;
use rustc_session::{Limit, Session};
use rustc_span::def_id::{CrateNum, DefId, LocalDefId};
use rustc_span::edition::Edition;
use rustc_span::hygiene::{AstPass, ExpnData, ExpnKind, LocalExpnId, MacroKind};
use rustc_span::source_map::SourceMap;
use rustc_span::{DUMMY_SP, FileName, Ident, Span, Symbol, kw, sym};
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;

use crate::base::ast::MetaItemInner;
use crate::errors;
use crate::expand::{self, AstFragment, Invocation};
use crate::module::DirOwnership;

// When adding new variants, make sure to
// adjust the `visit_*` / `flat_map_*` calls in `InvocationCollector`
// to use `assign_id!`
#[derive(Debug, Clone)]
pub enum Annotatable {
    Item(P<ast::Item>),
    AssocItem(P<ast::AssocItem>, AssocCtxt),
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
    WherePredicate(ast::WherePredicate),
    Crate(ast::Crate),
}

impl Annotatable {
    pub fn span(&self) -> Span {
        match self {
            Annotatable::Item(item) => item.span,
            Annotatable::AssocItem(assoc_item, _) => assoc_item.span,
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
            Annotatable::WherePredicate(wp) => wp.span,
            Annotatable::Crate(c) => c.spans.inner_span,
        }
    }

    pub fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        match self {
            Annotatable::Item(item) => item.visit_attrs(f),
            Annotatable::AssocItem(assoc_item, _) => assoc_item.visit_attrs(f),
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
            Annotatable::WherePredicate(wp) => wp.visit_attrs(f),
            Annotatable::Crate(c) => c.visit_attrs(f),
        }
    }

    pub fn visit_with<'a, V: Visitor<'a>>(&'a self, visitor: &mut V) -> V::Result {
        match self {
            Annotatable::Item(item) => visitor.visit_item(item),
            Annotatable::AssocItem(item, ctxt) => visitor.visit_assoc_item(item, *ctxt),
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
            Annotatable::WherePredicate(wp) => visitor.visit_where_predicate(wp),
            Annotatable::Crate(c) => visitor.visit_crate(c),
        }
    }

    pub fn to_tokens(&self) -> TokenStream {
        match self {
            Annotatable::Item(node) => TokenStream::from_ast(node),
            Annotatable::AssocItem(node, _) => TokenStream::from_ast(node),
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
            | Annotatable::WherePredicate(..)
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
            Annotatable::AssocItem(i, AssocCtxt::Trait) => i,
            _ => panic!("expected Item"),
        }
    }

    pub fn expect_impl_item(self) -> P<ast::AssocItem> {
        match self {
            Annotatable::AssocItem(i, AssocCtxt::Impl { .. }) => i,
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

    pub fn expect_where_predicate(self) -> ast::WherePredicate {
        match self {
            Annotatable::WherePredicate(wp) => wp,
            _ => panic!("expected where predicate"),
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

impl<T, U> ExpandResult<T, U> {
    pub fn map<E, F: FnOnce(T) -> E>(self, f: F) -> ExpandResult<E, U> {
        match self {
            ExpandResult::Ready(t) => ExpandResult::Ready(f(t)),
            ExpandResult::Retry(u) => ExpandResult::Retry(u),
        }
    }
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
    ) -> MacroExpanderResult<'cx>;
}

pub type MacroExpanderResult<'cx> = ExpandResult<Box<dyn MacResult + 'cx>, ()>;

pub type MacroExpanderFn =
    for<'cx> fn(&'cx mut ExtCtxt<'_>, Span, TokenStream) -> MacroExpanderResult<'cx>;

impl<F> TTMacroExpander for F
where
    F: for<'cx> Fn(&'cx mut ExtCtxt<'_>, Span, TokenStream) -> MacroExpanderResult<'cx>,
{
    fn expand<'cx>(
        &self,
        ecx: &'cx mut ExtCtxt<'_>,
        span: Span,
        input: TokenStream,
    ) -> MacroExpanderResult<'cx> {
        self(ecx, span, input)
    }
}

pub trait GlobDelegationExpander {
    fn expand(&self, ecx: &mut ExtCtxt<'_>) -> ExpandResult<Vec<(Ident, Option<Ident>)>, ()>;
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

    /// Creates zero or more impl items.
    fn make_trait_impl_items(self: Box<Self>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
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

    fn make_where_predicates(self: Box<Self>) -> Option<SmallVec<[ast::WherePredicate; 1]>> {
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

    fn make_trait_impl_items(self: Box<Self>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
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
                    kind: PatKind::Expr(e),
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
    guar: Option<ErrorGuaranteed>,
    span: Span,
}

impl DummyResult {
    /// Creates a default MacResult that can be anything.
    ///
    /// Use this as a return value after hitting any errors and
    /// calling `span_err`.
    pub fn any(span: Span, guar: ErrorGuaranteed) -> Box<dyn MacResult + 'static> {
        Box::new(DummyResult { guar: Some(guar), span })
    }

    /// Same as `any`, but must be a valid fragment, not error.
    pub fn any_valid(span: Span) -> Box<dyn MacResult + 'static> {
        Box::new(DummyResult { guar: None, span })
    }

    /// A plain dummy expression.
    pub fn raw_expr(sp: Span, guar: Option<ErrorGuaranteed>) -> P<ast::Expr> {
        P(ast::Expr {
            id: ast::DUMMY_NODE_ID,
            kind: if let Some(guar) = guar {
                ast::ExprKind::Err(guar)
            } else {
                ast::ExprKind::Tup(ThinVec::new())
            },
            span: sp,
            attrs: ast::AttrVec::new(),
            tokens: None,
        })
    }
}

impl MacResult for DummyResult {
    fn make_expr(self: Box<DummyResult>) -> Option<P<ast::Expr>> {
        Some(DummyResult::raw_expr(self.span, self.guar))
    }

    fn make_pat(self: Box<DummyResult>) -> Option<P<ast::Pat>> {
        Some(P(ast::Pat {
            id: ast::DUMMY_NODE_ID,
            kind: PatKind::Wild,
            span: self.span,
            tokens: None,
        }))
    }

    fn make_items(self: Box<DummyResult>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
        Some(SmallVec::new())
    }

    fn make_impl_items(self: Box<DummyResult>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
        Some(SmallVec::new())
    }

    fn make_trait_impl_items(self: Box<DummyResult>) -> Option<SmallVec<[P<ast::AssocItem>; 1]>> {
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
            kind: ast::StmtKind::Expr(DummyResult::raw_expr(self.span, self.guar)),
            span: self.span,
        }])
    }

    fn make_ty(self: Box<DummyResult>) -> Option<P<ast::Ty>> {
        // FIXME(nnethercote): you might expect `ast::TyKind::Dummy` to be used here, but some
        // values produced here end up being lowered to HIR, which `ast::TyKind::Dummy` does not
        // support, so we use an empty tuple instead.
        Some(P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            kind: ast::TyKind::Tup(ThinVec::new()),
            span: self.span,
            tokens: None,
        }))
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
        Some(ast::Crate {
            attrs: Default::default(),
            items: Default::default(),
            spans: Default::default(),
            id: ast::DUMMY_NODE_ID,
            is_placeholder: Default::default(),
        })
    }
}

/// A syntax extension kind.
#[derive(Clone)]
pub enum SyntaxExtensionKind {
    /// A token-based function-like macro.
    Bang(
        /// An expander with signature TokenStream -> TokenStream.
        Arc<dyn BangProcMacro + sync::DynSync + sync::DynSend>,
    ),

    /// An AST-based function-like macro.
    LegacyBang(
        /// An expander with signature TokenStream -> AST.
        Arc<dyn TTMacroExpander + sync::DynSync + sync::DynSend>,
    ),

    /// A token-based attribute macro.
    Attr(
        /// An expander with signature (TokenStream, TokenStream) -> TokenStream.
        /// The first TokenStream is the attribute itself, the second is the annotated item.
        /// The produced TokenStream replaces the input TokenStream.
        Arc<dyn AttrProcMacro + sync::DynSync + sync::DynSend>,
    ),

    /// An AST-based attribute macro.
    LegacyAttr(
        /// An expander with signature (AST, AST) -> AST.
        /// The first AST fragment is the attribute itself, the second is the annotated item.
        /// The produced AST fragment replaces the input AST fragment.
        Arc<dyn MultiItemModifier + sync::DynSync + sync::DynSend>,
    ),

    /// A trivial attribute "macro" that does nothing,
    /// only keeps the attribute and marks it as inert,
    /// thus making it ineligible for further expansion.
    NonMacroAttr,

    /// A token-based derive macro.
    Derive(
        /// An expander with signature TokenStream -> TokenStream.
        /// The produced TokenStream is appended to the input TokenStream.
        ///
        /// FIXME: The text above describes how this should work. Currently it
        /// is handled identically to `LegacyDerive`. It should be migrated to
        /// a token-based representation like `Bang` and `Attr`, instead of
        /// using `MultiItemModifier`.
        Arc<dyn MultiItemModifier + sync::DynSync + sync::DynSend>,
    ),

    /// An AST-based derive macro.
    LegacyDerive(
        /// An expander with signature AST -> AST.
        /// The produced AST fragment is appended to the input AST fragment.
        Arc<dyn MultiItemModifier + sync::DynSync + sync::DynSend>,
    ),

    /// A glob delegation.
    GlobDelegation(Arc<dyn GlobDelegationExpander + sync::DynSync + sync::DynSend>),
}

/// A struct representing a macro definition in "lowered" form ready for expansion.
pub struct SyntaxExtension {
    /// A syntax extension kind.
    pub kind: SyntaxExtensionKind,
    /// Span of the macro definition.
    pub span: Span,
    /// List of unstable features that are treated as stable inside this macro.
    pub allow_internal_unstable: Option<Arc<[Symbol]>>,
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
            SyntaxExtensionKind::Bang(..)
            | SyntaxExtensionKind::LegacyBang(..)
            | SyntaxExtensionKind::GlobDelegation(..) => MacroKind::Bang,
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

    fn collapse_debuginfo_by_name(
        attr: &impl AttributeExt,
    ) -> Result<CollapseMacroDebuginfo, Span> {
        let list = attr.meta_item_list();
        let Some([MetaItemInner::MetaItem(item)]) = list.as_deref() else {
            return Err(attr.span());
        };
        if !item.is_word() {
            return Err(item.span);
        }

        match item.name() {
            Some(sym::no) => Ok(CollapseMacroDebuginfo::No),
            Some(sym::external) => Ok(CollapseMacroDebuginfo::External),
            Some(sym::yes) => Ok(CollapseMacroDebuginfo::Yes),
            _ => Err(item.path.span),
        }
    }

    /// if-ext - if macro from different crate (related to callsite code)
    /// | cmd \ attr    | no  | (unspecified) | external | yes |
    /// | no            | no  | no            | no       | no  |
    /// | (unspecified) | no  | if-ext        | if-ext   | yes |
    /// | external      | no  | if-ext        | if-ext   | yes |
    /// | yes           | yes | yes           | yes      | yes |
    fn get_collapse_debuginfo(sess: &Session, attrs: &[impl AttributeExt], ext: bool) -> bool {
        let flag = sess.opts.cg.collapse_macro_debuginfo;
        let attr = ast::attr::find_by_name(attrs, sym::collapse_debuginfo)
            .and_then(|attr| {
                Self::collapse_debuginfo_by_name(attr)
                    .map_err(|span| {
                        sess.dcx().emit_err(errors::CollapseMacroDebuginfoIllegal { span })
                    })
                    .ok()
            })
            .unwrap_or_else(|| {
                if ast::attr::contains_name(attrs, sym::rustc_builtin_macro) {
                    CollapseMacroDebuginfo::Yes
                } else {
                    CollapseMacroDebuginfo::Unspecified
                }
            });
        #[rustfmt::skip]
        let collapse_table = [
            [false, false, false, false],
            [false, ext,   ext,   true],
            [false, ext,   ext,   true],
            [true,  true,  true,  true],
        ];
        collapse_table[flag as usize][attr as usize]
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
        attrs: &[hir::Attribute],
        is_local: bool,
    ) -> SyntaxExtension {
        let allow_internal_unstable =
            find_attr!(attrs, AttributeKind::AllowInternalUnstable(i) => i)
                .map(|i| i.as_slice())
                .unwrap_or_default();
        // FIXME(jdonszelman): allow_internal_unsafe isn't yet new-style
        // let allow_internal_unsafe = find_attr!(attrs, AttributeKind::AllowInternalUnsafe);
        let allow_internal_unsafe =
            ast::attr::find_by_name(attrs, sym::allow_internal_unsafe).is_some();

        let local_inner_macros = ast::attr::find_by_name(attrs, sym::macro_export)
            .and_then(|macro_export| macro_export.meta_item_list())
            .is_some_and(|l| ast::attr::list_contains_name(&l, sym::local_inner_macros));
        let collapse_debuginfo = Self::get_collapse_debuginfo(sess, attrs, !is_local);
        tracing::debug!(?name, ?local_inner_macros, ?collapse_debuginfo, ?allow_internal_unsafe);

        let (builtin_name, helper_attrs) = ast::attr::find_by_name(attrs, sym::rustc_builtin_macro)
            .map(|attr| {
                // Override `helper_attrs` passed above if it's a built-in macro,
                // marking `proc_macro_derive` macros as built-in is not a realistic use case.
                parse_macro_name_and_helper_attrs(sess.dcx(), attr, "built-in").map_or_else(
                    || (Some(name), Vec::new()),
                    |(name, helper_attrs)| (Some(name), helper_attrs),
                )
            })
            .unwrap_or_else(|| (None, helper_attrs));

        let stability = find_attr!(attrs, AttributeKind::Stability { stability, .. } => *stability);

        // FIXME(jdonszelmann): make it impossible to miss the or_else in the typesystem
        if let Some(sp) = find_attr!(attrs, AttributeKind::ConstStability { span, .. } => *span) {
            sess.dcx().emit_err(errors::MacroConstStability {
                span: sp,
                head_span: sess.source_map().guess_head_span(span),
            });
        }
        if let Some(sp) = find_attr!(attrs, AttributeKind::BodyStability{ span, .. } => *span) {
            sess.dcx().emit_err(errors::MacroBodyStability {
                span: sp,
                head_span: sess.source_map().guess_head_span(span),
            });
        }

        SyntaxExtension {
            kind,
            span,
            allow_internal_unstable: (!allow_internal_unstable.is_empty())
                // FIXME(jdonszelmann): avoid the into_iter/collect?
                .then(|| allow_internal_unstable.iter().map(|i| i.0).collect::<Vec<_>>().into()),
            stability,
            deprecation: find_attr!(
                attrs,
                AttributeKind::Deprecation { deprecation, .. } => *deprecation
            ),
            helper_attrs,
            edition,
            builtin_name,
            allow_internal_unsafe,
            local_inner_macros,
            collapse_debuginfo,
        }
    }

    /// A dummy bang macro `foo!()`.
    pub fn dummy_bang(edition: Edition) -> SyntaxExtension {
        fn expander<'cx>(
            cx: &'cx mut ExtCtxt<'_>,
            span: Span,
            _: TokenStream,
        ) -> MacroExpanderResult<'cx> {
            ExpandResult::Ready(DummyResult::any(
                span,
                cx.dcx().span_delayed_bug(span, "expanded a dummy bang macro"),
            ))
        }
        SyntaxExtension::default(SyntaxExtensionKind::LegacyBang(Arc::new(expander)), edition)
    }

    /// A dummy derive macro `#[derive(Foo)]`.
    pub fn dummy_derive(edition: Edition) -> SyntaxExtension {
        fn expander(
            _: &mut ExtCtxt<'_>,
            _: Span,
            _: &ast::MetaItem,
            _: Annotatable,
        ) -> Vec<Annotatable> {
            Vec::new()
        }
        SyntaxExtension::default(SyntaxExtensionKind::Derive(Arc::new(expander)), edition)
    }

    pub fn non_macro_attr(edition: Edition) -> SyntaxExtension {
        SyntaxExtension::default(SyntaxExtensionKind::NonMacroAttr, edition)
    }

    pub fn glob_delegation(
        trait_def_id: DefId,
        impl_def_id: LocalDefId,
        edition: Edition,
    ) -> SyntaxExtension {
        struct GlobDelegationExpanderImpl {
            trait_def_id: DefId,
            impl_def_id: LocalDefId,
        }
        impl GlobDelegationExpander for GlobDelegationExpanderImpl {
            fn expand(
                &self,
                ecx: &mut ExtCtxt<'_>,
            ) -> ExpandResult<Vec<(Ident, Option<Ident>)>, ()> {
                match ecx.resolver.glob_delegation_suffixes(self.trait_def_id, self.impl_def_id) {
                    Ok(suffixes) => ExpandResult::Ready(suffixes),
                    Err(Indeterminate) if ecx.force_mode => ExpandResult::Ready(Vec::new()),
                    Err(Indeterminate) => ExpandResult::Retry(()),
                }
            }
        }

        let expander = GlobDelegationExpanderImpl { trait_def_id, impl_def_id };
        SyntaxExtension::default(SyntaxExtensionKind::GlobDelegation(Arc::new(expander)), edition)
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
            self.builtin_name.is_some(),
        )
    }
}

/// Error type that denotes indeterminacy.
pub struct Indeterminate;

pub struct DeriveResolution {
    pub path: ast::Path,
    pub item: Annotatable,
    pub exts: Option<Arc<SyntaxExtension>>,
    pub is_const: bool,
}

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
    ) -> Result<Arc<SyntaxExtension>, Indeterminate>;

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
        derive_paths: &dyn Fn() -> Vec<DeriveResolution>,
    ) -> Result<(), Indeterminate>;
    /// Take resolutions for paths inside the `#[derive(...)]` attribute with the given `ExpnId`
    /// back from resolver.
    fn take_derive_resolutions(&mut self, expn_id: LocalExpnId) -> Option<Vec<DeriveResolution>>;
    /// Path resolution logic for `#[cfg_accessible(path)]`.
    fn cfg_accessible(
        &mut self,
        expn_id: LocalExpnId,
        path: &ast::Path,
    ) -> Result<bool, Indeterminate>;
    fn macro_accessible(
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

    fn append_stripped_cfg_item(&mut self, parent_node: NodeId, ident: Ident, cfg: ast::MetaItem);

    /// Tools registered with `#![register_tool]` and used by tool attributes and lints.
    fn registered_tools(&self) -> &RegisteredTools;

    /// Mark this invocation id as a glob delegation.
    fn register_glob_delegation(&mut self, invoc_id: LocalExpnId);

    /// Names of specific methods to which glob delegation expands.
    fn glob_delegation_suffixes(
        &mut self,
        trait_def_id: DefId,
        impl_def_id: LocalDefId,
    ) -> Result<Vec<(Ident, Option<Ident>)>, Indeterminate>;
}

pub trait LintStoreExpand {
    fn pre_expansion_lint(
        &self,
        sess: &Session,
        features: &Features,
        registered_tools: &RegisteredTools,
        node_id: NodeId,
        attrs: &[Attribute],
        items: &[P<Item>],
        name: Symbol,
    );
}

type LintStoreExpandDyn<'a> = Option<&'a (dyn LintStoreExpand + 'a)>;

#[derive(Debug, Clone, Default)]
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
    pub reduced_recursion_limit: Option<(Limit, ErrorGuaranteed)>,
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
                lint_node_id: ast::CRATE_NODE_ID,
                is_trailing_mac: false,
            },
            force_mode: false,
            expansions: FxIndexMap::default(),
            expanded_inert_attrs: MarkedAttrs::new(),
            buffered_early_lint: vec![],
        }
    }

    pub fn dcx(&self) -> DiagCtxtHandle<'a> {
        self.sess.dcx()
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
    pub fn new_parser_from_tts(&self, stream: TokenStream) -> Parser<'a> {
        Parser::new(&self.sess.psess, stream, MACRO_ARGUMENTS)
    }
    pub fn source_map(&self) -> &'a SourceMap {
        self.sess.psess.source_map()
    }
    pub fn psess(&self) -> &'a ParseSess {
        &self.sess.psess
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

    pub fn trace_macros_diag(&mut self) {
        for (span, notes) in self.expansions.iter() {
            let mut db = self.dcx().create_note(errors::TraceMacro { span: *span });
            for note in notes {
                // FIXME: make this translatable
                #[allow(rustc::untranslatable_diagnostic)]
                db.note(note.clone());
            }
            db.emit();
        }
        // Fixme: does this result in errors?
        self.expansions.clear();
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
pub fn resolve_path(sess: &Session, path: impl Into<PathBuf>, span: Span) -> PResult<'_, PathBuf> {
    let path = path.into();

    // Relative paths are resolved relative to the file in which they are found
    // after macro expansion (that is, they are unhygienic).
    if !path.is_absolute() {
        let callsite = span.source_callsite();
        let source_map = sess.source_map();
        let Some(mut base_path) = source_map.span_to_filename(callsite).into_local_path() else {
            return Err(sess.dcx().create_err(errors::ResolveRelativePath {
                span,
                path: source_map
                    .filename_for_diagnostics(&source_map.span_to_filename(callsite))
                    .to_string(),
            }));
        };
        base_path.pop();
        base_path.push(path);
        Ok(base_path)
    } else {
        // This ensures that Windows verbatim paths are fixed if mixed path separators are used,
        // which can happen when `concat!` is used to join paths.
        match path.components().next() {
            Some(Prefix(prefix)) if prefix.kind().is_verbatim() => Ok(path.components().collect()),
            _ => Ok(path),
        }
    }
}

pub fn parse_macro_name_and_helper_attrs(
    dcx: DiagCtxtHandle<'_>,
    attr: &impl AttributeExt,
    macro_type: &str,
) -> Option<(Symbol, Vec<Symbol>)> {
    // Once we've located the `#[proc_macro_derive]` attribute, verify
    // that it's of the form `#[proc_macro_derive(Foo)]` or
    // `#[proc_macro_derive(Foo, attributes(A, ..))]`
    let list = attr.meta_item_list()?;
    let ([trait_attr] | [trait_attr, _]) = list.as_slice() else {
        dcx.emit_err(errors::AttrNoArguments { span: attr.span() });
        return None;
    };
    let Some(trait_attr) = trait_attr.meta_item() else {
        dcx.emit_err(errors::NotAMetaItem { span: trait_attr.span() });
        return None;
    };
    let trait_ident = match trait_attr.ident() {
        Some(trait_ident) if trait_attr.is_word() => trait_ident,
        _ => {
            dcx.emit_err(errors::OnlyOneWord { span: trait_attr.span });
            return None;
        }
    };

    if !trait_ident.name.can_be_raw() {
        dcx.emit_err(errors::CannotBeNameOfMacro {
            span: trait_attr.span,
            trait_ident,
            macro_type,
        });
    }

    let attributes_attr = list.get(1);
    let proc_attrs: Vec<_> = if let Some(attr) = attributes_attr {
        if !attr.has_name(sym::attributes) {
            dcx.emit_err(errors::ArgumentNotAttributes { span: attr.span() });
        }
        attr.meta_item_list()
            .unwrap_or_else(|| {
                dcx.emit_err(errors::AttributesWrongForm { span: attr.span() });
                &[]
            })
            .iter()
            .filter_map(|attr| {
                let Some(attr) = attr.meta_item() else {
                    dcx.emit_err(errors::AttributeMetaItem { span: attr.span() });
                    return None;
                };

                let ident = match attr.ident() {
                    Some(ident) if attr.is_word() => ident,
                    _ => {
                        dcx.emit_err(errors::AttributeSingleWord { span: attr.span });
                        return None;
                    }
                };
                if !ident.name.can_be_raw() {
                    dcx.emit_err(errors::HelperAttributeNameInvalid {
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

/// If this item looks like a specific enums from `rental`, emit a fatal error.
/// See #73345 and #83125 for more details.
/// FIXME(#73933): Remove this eventually.
fn pretty_printing_compatibility_hack(item: &Item, psess: &ParseSess) {
    if let ast::ItemKind::Enum(ident, _, enum_def) = &item.kind
        && ident.name == sym::ProceduralMasqueradeDummyType
        && let [variant] = &*enum_def.variants
        && variant.ident.name == sym::Input
        && let FileName::Real(real) = psess.source_map().span_to_filename(ident.span)
        && let Some(c) = real
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
                && version.next().and_then(|c| c.parse::<u32>().ok()).is_some_and(|v| v < 6)
        };

        if crate_matches {
            psess.dcx().emit_fatal(errors::ProcMacroBackCompat {
                crate_name: "rental".to_string(),
                fixed_version: "0.5.6".to_string(),
            });
        }
    }
}

pub(crate) fn ann_pretty_printing_compatibility_hack(ann: &Annotatable, psess: &ParseSess) {
    let item = match ann {
        Annotatable::Item(item) => item,
        Annotatable::Stmt(stmt) => match &stmt.kind {
            ast::StmtKind::Item(item) => item,
            _ => return,
        },
        _ => return,
    };
    pretty_printing_compatibility_hack(item, psess)
}

pub(crate) fn stream_pretty_printing_compatibility_hack(
    kind: MetaVarKind,
    stream: &TokenStream,
    psess: &ParseSess,
) {
    let item = match kind {
        MetaVarKind::Item => {
            let mut parser = Parser::new(psess, stream.clone(), None);
            // No need to collect tokens for this simple check.
            parser
                .parse_item(ForceCollect::No)
                .expect("failed to reparse item")
                .expect("an actual item")
        }
        MetaVarKind::Stmt => {
            let mut parser = Parser::new(psess, stream.clone(), None);
            // No need to collect tokens for this simple check.
            let stmt = parser
                .parse_stmt(ForceCollect::No)
                .expect("failed to reparse")
                .expect("an actual stmt");
            match &stmt.kind {
                ast::StmtKind::Item(item) => item.clone(),
                _ => return,
            }
        }
        _ => return,
    };
    pretty_printing_compatibility_hack(&item, psess)
}
