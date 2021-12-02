use super::ptr::P;
use super::token::Nonterminal;
use super::tokenstream::LazyTokenStream;
use super::{Arm, Crate, ExprField, FieldDef, GenericParam, Param, PatField, Variant};
use super::{AssocItem, Expr, ForeignItem, Item, Local, MacCallStmt};
use super::{AttrItem, AttrKind, Block, Pat, Path, Ty, Visibility};
use super::{AttrVec, Attribute, Stmt, StmtKind};

use std::fmt::Debug;

/// An `AstLike` represents an AST node (or some wrapper around
/// and AST node) which stores some combination of attributes
/// and tokens.
pub trait AstLike: Sized + Debug {
    /// This is `true` if this `AstLike` might support 'custom' (proc-macro) inner
    /// attributes. Attributes like `#![cfg]` and `#![cfg_attr]` are not
    /// considered 'custom' attributes
    ///
    /// If this is `false`, then this `AstLike` definitely does
    /// not support 'custom' inner attributes, which enables some optimizations
    /// during token collection.
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool;
    fn attrs(&self) -> &[Attribute];
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>));
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>>;
}

impl<T: AstLike + 'static> AstLike for P<T> {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = T::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        (**self).attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        (**self).visit_attrs(f);
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
        (**self).tokens_mut()
    }
}

impl AstLike for crate::token::Nonterminal {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true;
    fn attrs(&self) -> &[Attribute] {
        match self {
            Nonterminal::NtItem(item) => item.attrs(),
            Nonterminal::NtStmt(stmt) => stmt.attrs(),
            Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => expr.attrs(),
            Nonterminal::NtPat(_)
            | Nonterminal::NtTy(_)
            | Nonterminal::NtMeta(_)
            | Nonterminal::NtPath(_)
            | Nonterminal::NtVis(_)
            | Nonterminal::NtTT(_)
            | Nonterminal::NtBlock(_)
            | Nonterminal::NtIdent(..)
            | Nonterminal::NtLifetime(_) => &[],
        }
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        match self {
            Nonterminal::NtItem(item) => item.visit_attrs(f),
            Nonterminal::NtStmt(stmt) => stmt.visit_attrs(f),
            Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => expr.visit_attrs(f),
            Nonterminal::NtPat(_)
            | Nonterminal::NtTy(_)
            | Nonterminal::NtMeta(_)
            | Nonterminal::NtPath(_)
            | Nonterminal::NtVis(_)
            | Nonterminal::NtTT(_)
            | Nonterminal::NtBlock(_)
            | Nonterminal::NtIdent(..)
            | Nonterminal::NtLifetime(_) => {}
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
        match self {
            Nonterminal::NtItem(item) => item.tokens_mut(),
            Nonterminal::NtStmt(stmt) => stmt.tokens_mut(),
            Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => expr.tokens_mut(),
            Nonterminal::NtPat(pat) => pat.tokens_mut(),
            Nonterminal::NtTy(ty) => ty.tokens_mut(),
            Nonterminal::NtMeta(attr_item) => attr_item.tokens_mut(),
            Nonterminal::NtPath(path) => path.tokens_mut(),
            Nonterminal::NtVis(vis) => vis.tokens_mut(),
            Nonterminal::NtBlock(block) => block.tokens_mut(),
            Nonterminal::NtIdent(..) | Nonterminal::NtLifetime(..) | Nonterminal::NtTT(..) => None,
        }
    }
}

fn visit_attrvec(attrs: &mut AttrVec, f: impl FnOnce(&mut Vec<Attribute>)) {
    crate::mut_visit::visit_clobber(attrs, |attrs| {
        let mut vec = attrs.into();
        f(&mut vec);
        vec.into()
    });
}

impl AstLike for StmtKind {
    // This might be an `StmtKind::Item`, which contains
    // an item that supports inner attrs
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true;

    fn attrs(&self) -> &[Attribute] {
        match self {
            StmtKind::Local(local) => local.attrs(),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.attrs(),
            StmtKind::Item(item) => item.attrs(),
            StmtKind::Empty => &[],
            StmtKind::MacCall(mac) => &mac.attrs,
        }
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        match self {
            StmtKind::Local(local) => local.visit_attrs(f),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.visit_attrs(f),
            StmtKind::Item(item) => item.visit_attrs(f),
            StmtKind::Empty => {}
            StmtKind::MacCall(mac) => visit_attrvec(&mut mac.attrs, f),
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
        Some(match self {
            StmtKind::Local(local) => &mut local.tokens,
            StmtKind::Item(item) => &mut item.tokens,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => &mut expr.tokens,
            StmtKind::Empty => return None,
            StmtKind::MacCall(mac) => &mut mac.tokens,
        })
    }
}

impl AstLike for Stmt {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = StmtKind::SUPPORTS_CUSTOM_INNER_ATTRS;

    fn attrs(&self) -> &[Attribute] {
        self.kind.attrs()
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        self.kind.visit_attrs(f);
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
        self.kind.tokens_mut()
    }
}

impl AstLike for Attribute {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;

    fn attrs(&self) -> &[Attribute] {
        &[]
    }
    fn visit_attrs(&mut self, _f: impl FnOnce(&mut Vec<Attribute>)) {}
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
        Some(match &mut self.kind {
            AttrKind::Normal(_, tokens) => tokens,
            kind @ AttrKind::DocComment(..) => {
                panic!("Called tokens_mut on doc comment attr {:?}", kind)
            }
        })
    }
}

impl<T: AstLike> AstLike for Option<T> {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = T::SUPPORTS_CUSTOM_INNER_ATTRS;

    fn attrs(&self) -> &[Attribute] {
        self.as_ref().map(|inner| inner.attrs()).unwrap_or(&[])
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        if let Some(inner) = self.as_mut() {
            inner.visit_attrs(f);
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
        self.as_mut().and_then(|inner| inner.tokens_mut())
    }
}

/// Helper trait for the macros below. Abstracts over
/// the two types of attribute fields that AST nodes
/// may have (`Vec<Attribute>` or `AttrVec`)
trait VecOrAttrVec {
    fn visit(&mut self, f: impl FnOnce(&mut Vec<Attribute>));
}

impl VecOrAttrVec for Vec<Attribute> {
    fn visit(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        f(self)
    }
}

impl VecOrAttrVec for AttrVec {
    fn visit(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        visit_attrvec(self, f)
    }
}

macro_rules! derive_has_tokens_and_attrs {
    (
        const SUPPORTS_CUSTOM_INNER_ATTRS: bool = $inner_attrs:literal;
        $($ty:path),*
    ) => { $(
        impl AstLike for $ty {
            const SUPPORTS_CUSTOM_INNER_ATTRS: bool = $inner_attrs;

            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                VecOrAttrVec::visit(&mut self.attrs, f)
            }

            fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
                Some(&mut self.tokens)
            }

        }
    )* }
}

macro_rules! derive_has_attrs_no_tokens {
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;

            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                VecOrAttrVec::visit(&mut self.attrs, f)
            }

            fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
                None
            }
        }
    )* }
}

macro_rules! derive_has_tokens_no_attrs {
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;

            fn attrs(&self) -> &[Attribute] {
                &[]
            }

            fn visit_attrs(&mut self, _f: impl FnOnce(&mut Vec<Attribute>)) {}
            fn tokens_mut(&mut self) -> Option<&mut Option<LazyTokenStream>> {
                Some(&mut self.tokens)
            }
        }
    )* }
}

// These ast nodes support both active and inert attributes,
// so they have tokens collected to pass to proc macros
derive_has_tokens_and_attrs! {
    // Both `Item` and `AssocItem` can have bodies, which
    // can contain inner attributes
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true;
    Item, AssocItem, ForeignItem
}

derive_has_tokens_and_attrs! {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;
    Local, MacCallStmt, Expr
}

// These ast nodes only support inert attributes, so they don't
// store tokens (since nothing can observe them)
derive_has_attrs_no_tokens! {
    FieldDef, Arm, ExprField, PatField, Variant, Param, GenericParam, Crate
}

// These AST nodes don't support attributes, but can
// be captured by a `macro_rules!` matcher. Therefore,
// they need to store tokens.
derive_has_tokens_no_attrs! {
    Ty, Block, AttrItem, Pat, Path, Visibility
}
