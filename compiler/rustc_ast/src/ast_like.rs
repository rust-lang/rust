use super::ptr::P;
use super::tokenstream::LazyTokenStream;
use super::{Arm, Field, FieldPat, GenericParam, Param, StructField, Variant};
use super::{AssocItem, Expr, ForeignItem, Item, Local};
use super::{AttrItem, AttrKind, Block, Pat, Path, Ty, Visibility};
use super::{AttrVec, Attribute, Stmt, StmtKind};

/// An `AstLike` represents an AST node (or some wrapper around
/// and AST node) which stores some combination of attributes
/// and tokens.
pub trait AstLike: Sized {
    fn attrs(&self) -> &[Attribute];
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>));
    /// Called by `Parser::collect_tokens` to store the collected
    /// tokens inside an AST node
    fn finalize_tokens(&mut self, _tokens: LazyTokenStream) {
        // This default impl makes this trait easier to implement
        // in tools like `rust-analyzer`
        panic!("`finalize_tokens` is not supported!")
    }
}

impl<T: AstLike + 'static> AstLike for P<T> {
    fn attrs(&self) -> &[Attribute] {
        (**self).attrs()
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        (**self).visit_attrs(f);
    }
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
        (**self).finalize_tokens(tokens)
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
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
        let stmt_tokens = match self {
            StmtKind::Local(ref mut local) => &mut local.tokens,
            StmtKind::Item(ref mut item) => &mut item.tokens,
            StmtKind::Expr(ref mut expr) | StmtKind::Semi(ref mut expr) => &mut expr.tokens,
            StmtKind::Empty => return,
            StmtKind::MacCall(ref mut mac) => &mut mac.tokens,
        };
        if stmt_tokens.is_none() {
            *stmt_tokens = Some(tokens);
        }
    }
}

impl AstLike for Stmt {
    fn attrs(&self) -> &[Attribute] {
        self.kind.attrs()
    }

    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        self.kind.visit_attrs(f);
    }
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
        self.kind.finalize_tokens(tokens)
    }
}

impl AstLike for Attribute {
    fn attrs(&self) -> &[Attribute] {
        &[]
    }
    fn visit_attrs(&mut self, _f: impl FnOnce(&mut Vec<Attribute>)) {}
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
        match &mut self.kind {
            AttrKind::Normal(_, attr_tokens) => {
                if attr_tokens.is_none() {
                    *attr_tokens = Some(tokens);
                }
            }
            AttrKind::DocComment(..) => {
                panic!("Called finalize_tokens on doc comment attr {:?}", self)
            }
        }
    }
}

impl<T: AstLike> AstLike for Option<T> {
    fn attrs(&self) -> &[Attribute] {
        self.as_ref().map(|inner| inner.attrs()).unwrap_or(&[])
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
        if let Some(inner) = self.as_mut() {
            inner.visit_attrs(f);
        }
    }
    fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
        if let Some(inner) = self {
            inner.finalize_tokens(tokens);
        }
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
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                VecOrAttrVec::visit(&mut self.attrs, f)
            }

            fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
                if self.tokens.is_none() {
                    self.tokens = Some(tokens);
                }

            }
        }
    )* }
}

macro_rules! derive_has_attrs_no_tokens {
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            fn attrs(&self) -> &[Attribute] {
                &self.attrs
            }

            fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<Attribute>)) {
                VecOrAttrVec::visit(&mut self.attrs, f)
            }

            fn finalize_tokens(&mut self, _tokens: LazyTokenStream) {}
        }
    )* }
}

macro_rules! derive_has_tokens_no_attrs {
    ($($ty:path),*) => { $(
        impl AstLike for $ty {
            fn attrs(&self) -> &[Attribute] {
                &[]
            }

            fn visit_attrs(&mut self, _f: impl FnOnce(&mut Vec<Attribute>)) {
            }

            fn finalize_tokens(&mut self, tokens: LazyTokenStream) {
                if self.tokens.is_none() {
                    self.tokens = Some(tokens);
                }

            }
        }
    )* }
}

// These AST nodes support both inert and active
// attributes, so they also have tokens.
derive_has_tokens_and_attrs! {
    Item, Expr, Local, AssocItem, ForeignItem
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
