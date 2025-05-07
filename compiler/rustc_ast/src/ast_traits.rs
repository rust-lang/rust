//! A set of traits implemented for various AST nodes,
//! typically those used in AST fragments during macro expansion.
//! The traits are not implemented exhaustively, only when actually necessary.

use std::fmt;
use std::marker::PhantomData;

use crate::ptr::P;
use crate::tokenstream::LazyAttrTokenStream;
use crate::{
    Arm, AssocItem, AttrItem, AttrKind, AttrVec, Attribute, Block, Crate, Expr, ExprField,
    FieldDef, ForeignItem, GenericParam, Item, NodeId, Param, Pat, PatField, Path, Stmt, StmtKind,
    Ty, Variant, Visibility, WherePredicate,
};

/// A utility trait to reduce boilerplate.
/// Standard `Deref(Mut)` cannot be reused due to coherence.
pub trait AstDeref {
    type Target;
    fn ast_deref(&self) -> &Self::Target;
    fn ast_deref_mut(&mut self) -> &mut Self::Target;
}

macro_rules! impl_not_ast_deref {
    ($($T:ty),+ $(,)?) => {
        $(
            impl !AstDeref for $T {}
        )+
    };
}

impl_not_ast_deref!(AssocItem, Expr, ForeignItem, Item, Stmt);

impl<T> AstDeref for P<T> {
    type Target = T;
    fn ast_deref(&self) -> &Self::Target {
        self
    }
    fn ast_deref_mut(&mut self) -> &mut Self::Target {
        self
    }
}

/// A trait for AST nodes having an ID.
pub trait HasNodeId {
    fn node_id(&self) -> NodeId;
    fn node_id_mut(&mut self) -> &mut NodeId;
}

macro_rules! impl_has_node_id {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasNodeId for $T {
                fn node_id(&self) -> NodeId {
                    self.id
                }
                fn node_id_mut(&mut self) -> &mut NodeId {
                    &mut self.id
                }
            }
        )+
    };
}

impl_has_node_id!(
    Arm,
    AssocItem,
    Crate,
    Expr,
    ExprField,
    FieldDef,
    ForeignItem,
    GenericParam,
    Item,
    Param,
    Pat,
    PatField,
    Stmt,
    Ty,
    Variant,
    WherePredicate,
);

impl<T: AstDeref<Target: HasNodeId>> HasNodeId for T {
    fn node_id(&self) -> NodeId {
        self.ast_deref().node_id()
    }
    fn node_id_mut(&mut self) -> &mut NodeId {
        self.ast_deref_mut().node_id_mut()
    }
}

/// A trait for AST nodes having (or not having) collected tokens.
pub trait HasTokens {
    fn tokens(&self) -> Option<&LazyAttrTokenStream>;
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>>;
}

macro_rules! impl_has_tokens {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasTokens for $T {
                fn tokens(&self) -> Option<&LazyAttrTokenStream> {
                    self.tokens.as_ref()
                }
                fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
                    Some(&mut self.tokens)
                }
            }
        )+
    };
}

macro_rules! impl_has_tokens_none {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasTokens for $T {
                fn tokens(&self) -> Option<&LazyAttrTokenStream> {
                    None
                }
                fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
                    None
                }
            }
        )+
    };
}

impl_has_tokens!(AssocItem, AttrItem, Block, Expr, ForeignItem, Item, Pat, Path, Ty, Visibility);
impl_has_tokens_none!(
    Arm,
    ExprField,
    FieldDef,
    GenericParam,
    Param,
    PatField,
    Variant,
    WherePredicate
);

impl<T: AstDeref<Target: HasTokens>> HasTokens for T {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        self.ast_deref().tokens()
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        self.ast_deref_mut().tokens_mut()
    }
}

impl<T: HasTokens> HasTokens for Option<T> {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        self.as_ref().and_then(|inner| inner.tokens())
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        self.as_mut().and_then(|inner| inner.tokens_mut())
    }
}

impl HasTokens for StmtKind {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        match self {
            StmtKind::Let(local) => local.tokens.as_ref(),
            StmtKind::Item(item) => item.tokens(),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.tokens(),
            StmtKind::Empty => None,
            StmtKind::MacCall(mac) => mac.tokens.as_ref(),
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        match self {
            StmtKind::Let(local) => Some(&mut local.tokens),
            StmtKind::Item(item) => item.tokens_mut(),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.tokens_mut(),
            StmtKind::Empty => None,
            StmtKind::MacCall(mac) => Some(&mut mac.tokens),
        }
    }
}

impl HasTokens for Stmt {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        self.kind.tokens()
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        self.kind.tokens_mut()
    }
}

impl HasTokens for Attribute {
    fn tokens(&self) -> Option<&LazyAttrTokenStream> {
        match &self.kind {
            AttrKind::Normal(normal) => normal.tokens.as_ref(),
            kind @ AttrKind::DocComment(..) => {
                panic!("Called tokens on doc comment attr {kind:?}")
            }
        }
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<LazyAttrTokenStream>> {
        Some(match &mut self.kind {
            AttrKind::Normal(normal) => &mut normal.tokens,
            kind @ AttrKind::DocComment(..) => {
                panic!("Called tokens_mut on doc comment attr {kind:?}")
            }
        })
    }
}

/// A trait for AST nodes having (or not having) attributes.
pub trait HasAttrs {
    /// This is `true` if this `HasAttrs` might support 'custom' (proc-macro) inner
    /// attributes. Attributes like `#![cfg]` and `#![cfg_attr]` are not
    /// considered 'custom' attributes.
    ///
    /// If this is `false`, then this `HasAttrs` definitely does
    /// not support 'custom' inner attributes, which enables some optimizations
    /// during token collection.
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool;
    fn attrs(&self) -> &[Attribute];
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec));
}

macro_rules! impl_has_attrs {
    (const SUPPORTS_CUSTOM_INNER_ATTRS: bool = $inner:literal, $($T:ty),+ $(,)?) => {
        $(
            impl HasAttrs for $T {
                const SUPPORTS_CUSTOM_INNER_ATTRS: bool = $inner;

                #[inline]
                fn attrs(&self) -> &[Attribute] {
                    &self.attrs
                }

                fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
                    f(&mut self.attrs)
                }
            }
        )+
    };
}

macro_rules! impl_has_attrs_none {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasAttrs for $T {
                const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false;
                fn attrs(&self) -> &[Attribute] {
                    &[]
                }
                fn visit_attrs(&mut self, _f: impl FnOnce(&mut AttrVec)) {}
            }
        )+
    };
}

impl_has_attrs!(
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true,
    AssocItem,
    ForeignItem,
    Item,
);
impl_has_attrs!(
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = false,
    Arm,
    Crate,
    Expr,
    ExprField,
    FieldDef,
    GenericParam,
    Param,
    PatField,
    Variant,
    WherePredicate,
);
impl_has_attrs_none!(Attribute, AttrItem, Block, Pat, Path, Ty, Visibility);

impl<T: AstDeref<Target: HasAttrs>> HasAttrs for T {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = T::Target::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        self.ast_deref().attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        self.ast_deref_mut().visit_attrs(f)
    }
}

impl<T: HasAttrs> HasAttrs for Option<T> {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = T::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        self.as_ref().map(|inner| inner.attrs()).unwrap_or(&[])
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        if let Some(inner) = self.as_mut() {
            inner.visit_attrs(f);
        }
    }
}

impl HasAttrs for StmtKind {
    // This might be a `StmtKind::Item`, which contains
    // an item that supports inner attrs.
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true;

    fn attrs(&self) -> &[Attribute] {
        match self {
            StmtKind::Let(local) => &local.attrs,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.attrs(),
            StmtKind::Item(item) => item.attrs(),
            StmtKind::Empty => &[],
            StmtKind::MacCall(mac) => &mac.attrs,
        }
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        match self {
            StmtKind::Let(local) => f(&mut local.attrs),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr.visit_attrs(f),
            StmtKind::Item(item) => item.visit_attrs(f),
            StmtKind::Empty => {}
            StmtKind::MacCall(mac) => f(&mut mac.attrs),
        }
    }
}

impl HasAttrs for Stmt {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = StmtKind::SUPPORTS_CUSTOM_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        self.kind.attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut AttrVec)) {
        self.kind.visit_attrs(f);
    }
}

/// A newtype around an AST node that implements the traits above if the node implements them.
pub struct AstNodeWrapper<Wrapped, Tag> {
    pub wrapped: Wrapped,
    pub tag: PhantomData<Tag>,
}

impl<Wrapped, Tag> AstNodeWrapper<Wrapped, Tag> {
    pub fn new(wrapped: Wrapped, _tag: Tag) -> AstNodeWrapper<Wrapped, Tag> {
        AstNodeWrapper { wrapped, tag: Default::default() }
    }
}

impl<Wrapped, Tag> AstDeref for AstNodeWrapper<Wrapped, Tag> {
    type Target = Wrapped;
    fn ast_deref(&self) -> &Self::Target {
        &self.wrapped
    }
    fn ast_deref_mut(&mut self) -> &mut Self::Target {
        &mut self.wrapped
    }
}

impl<Wrapped: fmt::Debug, Tag> fmt::Debug for AstNodeWrapper<Wrapped, Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstNodeWrapper")
            .field("wrapped", &self.wrapped)
            .field("tag", &self.tag)
            .finish()
    }
}
