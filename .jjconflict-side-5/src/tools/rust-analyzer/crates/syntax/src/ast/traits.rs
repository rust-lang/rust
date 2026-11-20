//! Various traits that are implemented by ast nodes.
//!
//! The implementations are usually trivial, and live in generated.rs
use either::Either;

use crate::{
    SyntaxNode, SyntaxToken, T,
    ast::{self, AstChildren, AstNode, support},
    match_ast,
};

pub trait HasName: AstNode {
    fn name(&self) -> Option<ast::Name> {
        support::child(self.syntax())
    }
}

pub trait HasVisibility: AstNode {
    fn visibility(&self) -> Option<ast::Visibility> {
        support::child(self.syntax())
    }
}

pub trait HasLoopBody: AstNode {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        support::child(self.syntax())
    }

    fn label(&self) -> Option<ast::Label> {
        support::child(self.syntax())
    }
}

pub trait HasArgList: AstNode {
    fn arg_list(&self) -> Option<ast::ArgList> {
        support::child(self.syntax())
    }
}

pub trait HasModuleItem: AstNode {
    fn items(&self) -> AstChildren<ast::Item> {
        support::children(self.syntax())
    }
}

pub trait HasGenericParams: AstNode {
    fn generic_param_list(&self) -> Option<ast::GenericParamList> {
        support::child(self.syntax())
    }

    fn where_clause(&self) -> Option<ast::WhereClause> {
        support::child(self.syntax())
    }
}
pub trait HasGenericArgs: AstNode {
    fn generic_arg_list(&self) -> Option<ast::GenericArgList> {
        support::child(self.syntax())
    }
}

pub trait HasTypeBounds: AstNode {
    fn type_bound_list(&self) -> Option<ast::TypeBoundList> {
        support::child(self.syntax())
    }

    fn colon_token(&self) -> Option<SyntaxToken> {
        support::token(self.syntax(), T![:])
    }
}

pub trait HasAttrs: AstNode {
    fn attrs(&self) -> AstChildren<ast::Attr> {
        support::children(self.syntax())
    }

    fn doc_comments(&self) -> AstChildren<ast::DocComment> {
        support::children(self.syntax())
    }

    fn attrs_with_doc(&self) -> AstChildren<ast::AnyAttr> {
        support::children(self.syntax())
    }

    /// This may return the same node as called with (with `SourceFile`). The caller has the responsibility
    /// to avoid duplicate attributes.
    fn inner_attributes_node(&self) -> Option<SyntaxNode> {
        let syntax = self.syntax();
        Some(match_ast! {
            match syntax {
                // A `SourceFile` contains the inner attributes of itself.
                ast::SourceFile(_) => syntax.clone(),
                ast::ExternBlock(it) => it.extern_item_list()?.syntax().clone(),
                ast::Fn(it) => it.body()?.stmt_list()?.syntax().clone(),
                ast::MatchExpr(it) => it.match_arm_list()?.syntax().clone(),
                ast::Impl(it) => it.assoc_item_list()?.syntax().clone(),
                ast::Trait(it) => it.assoc_item_list()?.syntax().clone(),
                ast::Module(it) => it.item_list()?.syntax().clone(),
                ast::BlockExpr(it) => {
                    if !it.may_carry_attributes() {
                        return None;
                    }
                    syntax.clone()
                },
                _ => return None,
            }
        })
    }
}

/// Returns all attributes of this node, including inner attributes that may not be directly under this node
/// but under a child.
pub fn attrs_with_doc_including_inner(
    owner: &dyn HasAttrs,
) -> impl Iterator<Item = ast::AnyAttr> + Clone {
    owner.attrs_with_doc().filter(|attr| attr.kind().is_outer()).chain(
        owner
            .inner_attributes_node()
            .into_iter()
            .flat_map(|node| support::children::<ast::AnyAttr>(&node))
            .filter(|attr| attr.kind().is_inner()),
    )
}

pub fn attrs_including_inner(owner: &dyn HasAttrs) -> impl Iterator<Item = ast::Attr> + Clone {
    AttrsIter::new(attrs_with_doc_including_inner(owner))
}

#[derive(Clone)]
pub struct AttrsIter<I> {
    inner: I,
}

impl<I: Iterator<Item = ast::AnyAttr>> AttrsIter<I> {
    #[inline]
    pub fn new(inner: I) -> Self {
        Self { inner }
    }
}

impl<I: Iterator<Item = ast::AnyAttr>> Iterator for AttrsIter<I> {
    type Item = ast::Attr;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|attr| match attr {
            ast::AnyAttr::Attr(it) => Some(it),
            ast::AnyAttr::DocComment(_) => None,
        })
    }
}

impl<A: HasName, B: HasName> HasName for Either<A, B> {}
