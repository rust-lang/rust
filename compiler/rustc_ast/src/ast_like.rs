use super::ptr::P;
use super::tokenstream::{LazyTokenStream, Spacing, AttributesData, PreexpTokenTree, PreexpTokenStream};
use super::{Arm, Field, FieldPat, GenericParam, Param, StructField, Variant};
use super::{AssocItem, Expr, ForeignItem, Item, Local, MacCallStmt};
use super::{AttrItem, AttrKind, Block, Pat, Path, Ty, Visibility};
use super::{AttrVec, Attribute, Stmt, StmtKind};
use rustc_span::sym;

/// An `AstLike` represents an AST node (or some wrapper around
/// and AST node) which stores some combination of attributes
/// and tokens.
pub trait AstLike: Sized {
    const SUPPORTS_INNER_ATTRS: bool;
    fn attrs(&self) -> &[Attribute];
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>));
    /// Called by `Parser::collect_tokens` to store the collected
    /// tokens inside an AST node
    fn finalize_tokens(&mut self, _tokens: LazyTokenStream) -> Option<AttributesData> {
        // This default impl makes this trait easier to implement
        // in tools like `rust-analyzer`
        panic!("`finalize_tokens` is not supported!")
    }
    fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>));
}

impl<T: AstLike + 'static> AstLike for P<T> {
    const SUPPORTS_INNER_ATTRS: bool = T::SUPPORTS_INNER_ATTRS;
    fn attrs(&self) -> &[Attribute] {
        (**self).attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        (**self).visit_attrs(f);
    }
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
        (**self).finalize_tokens(tokens)
    }
    fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
        (**self).visit_tokens(f);
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
    const SUPPORTS_INNER_ATTRS: bool = true;

    fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtKind::Local(ref local) => local.attrs(),
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => expr.attrs(),
            StmtKind::Item(ref item) => item.attrs(),
            StmtKind::Empty => &[],
            StmtKind::MacCall(ref mac) => &*mac.attrs,
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
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
        match self {
            StmtKind::Local(ref mut local) => local.finalize_tokens(tokens),
            StmtKind::MacCall(ref mut mac) => mac.finalize_tokens(tokens),
            StmtKind::Expr(ref mut expr) | StmtKind::Semi(ref mut expr) => {
                expr.finalize_tokens(tokens)
            }
            StmtKind::Item(ref mut item) => item.finalize_tokens(tokens),
            StmtKind::Empty => None,
        }
    }
    fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
        let tokens = match self {
            StmtKind::Local(ref mut local) => Some(&mut local.tokens),
            StmtKind::Item(ref mut item) => Some(&mut item.tokens),
            StmtKind::Expr(ref mut expr) | StmtKind::Semi(ref mut expr) => Some(&mut expr.tokens),
            StmtKind::Empty => None,
            StmtKind::MacCall(ref mut mac) => Some(&mut mac.tokens),
        };
        if let Some(tokens) = tokens {
            f(tokens);
        }
    }
}

impl AstLike for Stmt {
    const SUPPORTS_INNER_ATTRS: bool = StmtKind::SUPPORTS_INNER_ATTRS;

    fn attrs(&self) -> &[Attribute] {
        self.kind.attrs()
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        self.kind.visit_attrs(f);
    }
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
        self.kind.finalize_tokens(tokens)
    }
    fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
        self.kind.visit_tokens(f)
    }
}

impl AstLike for Attribute {
    const SUPPORTS_INNER_ATTRS: bool = false;

    fn attrs(&self) -> &[Attribute] {
        &[]
    }
    fn visit_attrs(&mut self, _f: impl FnOnce(&mut Vec<Attribute>)) {}
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
        match &mut self.kind {
            AttrKind::Normal(_, attr_tokens) => {
                if attr_tokens.is_none() {
                    *attr_tokens = Some(tokens);
                }
                None
            }
            AttrKind::DocComment(..) => {
                panic!("Called finalize_tokens on doc comment attr {:?}", self)
            }
        }
    }
    fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
        match &mut self.kind {
            AttrKind::Normal(_, attr_tokens) => {
                f(attr_tokens);
            }
            AttrKind::DocComment(..) => {
                panic!("Called visit_tokens on doc comment attr {:?}", self)
            }
        }

    }
}

impl<T: AstLike> AstLike for Option<T> {
    const SUPPORTS_INNER_ATTRS: bool = T::SUPPORTS_INNER_ATTRS;

    fn attrs(&self) -> &[Attribute] {
        self.as_ref().map(|inner| inner.attrs()).unwrap_or(&[])
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        if let Some(inner) = self.as_mut() {
            inner.visit_attrs(f);
        }
    }
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
        self.as_mut().and_then(|inner| inner.finalize_tokens(tokens))
    }
    fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
        if let Some(inner) = self {
            inner.visit_tokens(f);
        }
    }
}

// NOTE: Builtin attributes like `cfg` and `cfg_attr` cannot be renamed via imports.
// Therefore, the absence of a literal `cfg` or `cfg_attr` guarantees that
// we don't need to do any eager expansion.
pub fn has_cfg_or_cfg_any(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.ident().map_or(false, |ident| ident.name == sym::cfg || ident.name == sym::cfg_attr)
    })
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
        const SUPPORTS_INNER_ATTRS: bool = $inner_attrs:literal;
        $($ty:path),*
    ) => { $(
        impl AstLike for $ty {
            const SUPPORTS_INNER_ATTRS: bool = $inner_attrs;

            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                VecOrAttrVec::visit(&mut self.attrs, f)
            }

            fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
                if self.tokens.is_none() {
                    self.tokens = Some(tokens);
                }

                if has_cfg_or_cfg_any(&self.attrs) {
                    Some(AttributesData { attrs: self.attrs.clone().into(), tokens: self.tokens.clone().unwrap() })
                } else {
                    None
                }
            }

            fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
                f(&mut self.tokens)
            }

        }
    )* }
}

macro_rules! derive_has_attrs_no_tokens {
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            const SUPPORTS_INNER_ATTRS: bool = false;

            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                VecOrAttrVec::visit(&mut self.attrs, f)
            }

            fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
                if has_cfg_or_cfg_any(&self.attrs) {
                    Some(AttributesData { attrs: self.attrs.clone().into(), tokens })
                } else {
                    None
                }
            }
            fn visit_tokens(&mut self, _f: impl FnOnce(&mut Option<LazyTokenStream>)) {}

        }
    )* }
}

macro_rules! derive_has_tokens_no_attrs {
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            const SUPPORTS_INNER_ATTRS: bool = false;

            fn attrs(&self) -> &[Attribute] {
                &[]
            }

            fn visit_attrs(&mut self, _f: impl FnOnce(&mut Vec<Attribute>)) {
            }

            fn finalize_tokens(&mut self, tokens: LazyTokenStream) -> Option<AttributesData> {
                // FIXME - stoer them directly?
                if self.tokens.is_none() {
                    self.tokens = Some(LazyTokenStream::new(PreexpTokenStream::new(vec![
                        (PreexpTokenTree::Attributes(AttributesData {
                            attrs: Default::default(),
                            tokens,
                        }), Spacing::Alone)])));
                }
                None
            }
            fn visit_tokens(&mut self, f: impl FnOnce(&mut Option<LazyTokenStream>)) {
                f(&mut self.tokens)
            }
        }
    )* }
}

// These ast nodes support both active and inert attributes,
// so they have tokens collected to pass to proc macros
derive_has_tokens_and_attrs! {
    // Both `Item` and `AssocItem` can have bodies, which
    // can contain inner attributes
    const SUPPORTS_INNER_ATTRS: bool = true;
    Item, AssocItem
}

derive_has_tokens_and_attrs! {
    const SUPPORTS_INNER_ATTRS: bool = false;
    ForeignItem, Expr, Local, MacCallStmt
}

// These ast nodes only support inert attributes, so they don't
// store tokens (since nothing can observe them)
derive_has_attrs_no_tokens! {
    StructField, Arm,
    Field, FieldPat, Variant, Param, GenericParam
}

// These AST nodes don't support attributes, but can
// be captured by a `macro_rules!` matcher. Therefore,
// they need to store tokens.
derive_has_tokens_no_attrs! {
    Ty, Block, AttrItem, Pat, Path, Visibility
}
