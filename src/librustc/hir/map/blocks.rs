//! This module provides a simplified abstraction for working with
//! code blocks identified by their integer `NodeId`. In particular,
//! it captures a common set of attributes that all "function-like
//! things" (represented by `FnLike` instances) share. For example,
//! all `FnLike` instances have a type signature (be it explicit or
//! inferred). And all `FnLike` instances have a body, i.e., the code
//! that is run when the function-like thing it represents is invoked.
//!
//! With the above abstraction in place, one can treat the program
//! text as a collection of blocks of code (and most such blocks are
//! nested within a uniquely determined `FnLike`), and users can ask
//! for the `Code` associated with a particular NodeId.

use crate::hir as ast;
use crate::hir::map;
use crate::hir::{Expr, FnDecl, Node};
use crate::hir::intravisit::FnKind;
use syntax::ast::{Attribute, Ident};
use syntax_pos::Span;

/// An FnLikeNode is a Node that is like a fn, in that it has a decl
/// and a body (as well as a NodeId, a span, etc).
///
/// More specifically, it is one of either:
///
///   - A function item,
///   - A closure expr (i.e., an ExprKind::Closure), or
///   - The default implementation for a trait method.
///
/// To construct one, use the `Code::from_node` function.
#[derive(Copy, Clone, Debug)]
pub struct FnLikeNode<'a> { node: Node<'a> }

/// MaybeFnLike wraps a method that indicates if an object
/// corresponds to some FnLikeNode.
trait MaybeFnLike { fn is_fn_like(&self) -> bool; }

impl MaybeFnLike for ast::Item {
    fn is_fn_like(&self) -> bool {
        match self.node { ast::ItemKind::Fn(..) => true, _ => false, }
    }
}

impl MaybeFnLike for ast::ImplItem {
    fn is_fn_like(&self) -> bool {
        match self.node { ast::ImplItemKind::Method(..) => true, _ => false, }
    }
}

impl MaybeFnLike for ast::TraitItem {
    fn is_fn_like(&self) -> bool {
        match self.node {
            ast::TraitItemKind::Method(_, ast::TraitMethod::Provided(_)) => true,
            _ => false,
        }
    }
}

impl MaybeFnLike for ast::Expr {
    fn is_fn_like(&self) -> bool {
        match self.node {
            ast::ExprKind::Closure(..) => true,
            _ => false,
        }
    }
}

/// Carries either an FnLikeNode or a Expr, as these are the two
/// constructs that correspond to "code" (as in, something from which
/// we can construct a control-flow graph).
#[derive(Copy, Clone)]
pub enum Code<'a> {
    FnLike(FnLikeNode<'a>),
    Expr(&'a Expr),
}

impl<'a> Code<'a> {
    pub fn id(&self) -> ast::HirId {
        match *self {
            Code::FnLike(node) => node.id(),
            Code::Expr(block) => block.hir_id,
        }
    }

    /// Attempts to construct a Code from presumed FnLike or Expr node input.
    pub fn from_node(map: &map::Map<'a>, id: ast::HirId) -> Option<Code<'a>> {
        match map.get(id) {
            map::Node::Block(_) => {
                //  Use the parent, hopefully an expression node.
                Code::from_node(map, map.get_parent_node(id))
            }
            map::Node::Expr(expr) => Some(Code::Expr(expr)),
            node => FnLikeNode::from_node(node).map(Code::FnLike)
        }
    }
}

/// These are all the components one can extract from a fn item for
/// use when implementing FnLikeNode operations.
struct ItemFnParts<'a> {
    ident:    Ident,
    decl:     &'a ast::FnDecl,
    header:   ast::FnHeader,
    vis:      &'a ast::Visibility,
    generics: &'a ast::Generics,
    body:     ast::BodyId,
    id:       ast::HirId,
    span:     Span,
    attrs:    &'a [Attribute],
}

/// These are all the components one can extract from a closure expr
/// for use when implementing FnLikeNode operations.
struct ClosureParts<'a> {
    decl: &'a FnDecl,
    body: ast::BodyId,
    id: ast::HirId,
    span: Span,
    attrs: &'a [Attribute],
}

impl<'a> ClosureParts<'a> {
    fn new(d: &'a FnDecl, b: ast::BodyId, id: ast::HirId, s: Span, attrs: &'a [Attribute]) -> Self {
        ClosureParts {
            decl: d,
            body: b,
            id,
            span: s,
            attrs,
        }
    }
}

impl<'a> FnLikeNode<'a> {
    /// Attempts to construct a FnLikeNode from presumed FnLike node input.
    pub fn from_node(node: Node<'_>) -> Option<FnLikeNode<'_>> {
        let fn_like = match node {
            map::Node::Item(item) => item.is_fn_like(),
            map::Node::TraitItem(tm) => tm.is_fn_like(),
            map::Node::ImplItem(it) => it.is_fn_like(),
            map::Node::Expr(e) => e.is_fn_like(),
            _ => false
        };
        if fn_like {
            Some(FnLikeNode {
                node,
            })
        } else {
            None
        }
    }

    pub fn body(self) -> ast::BodyId {
        self.handle(|i: ItemFnParts<'a>| i.body,
                    |_, _, _: &'a ast::MethodSig, _, body: ast::BodyId, _, _| body,
                    |c: ClosureParts<'a>| c.body)
    }

    pub fn decl(self) -> &'a FnDecl {
        self.handle(|i: ItemFnParts<'a>| &*i.decl,
                    |_, _, sig: &'a ast::MethodSig, _, _, _, _| &sig.decl,
                    |c: ClosureParts<'a>| c.decl)
    }

    pub fn span(self) -> Span {
        self.handle(|i: ItemFnParts<'_>| i.span,
                    |_, _, _: &'a ast::MethodSig, _, _, span, _| span,
                    |c: ClosureParts<'_>| c.span)
    }

    pub fn id(self) -> ast::HirId {
        self.handle(|i: ItemFnParts<'_>| i.id,
                    |id, _, _: &'a ast::MethodSig, _, _, _, _| id,
                    |c: ClosureParts<'_>| c.id)
    }

    pub fn constness(self) -> ast::Constness {
        self.kind().header().map_or(ast::Constness::NotConst, |header| header.constness)
    }

    pub fn asyncness(self) -> ast::IsAsync {
        self.kind().header().map_or(ast::IsAsync::NotAsync, |header| header.asyncness)
    }

    pub fn unsafety(self) -> ast::Unsafety {
        self.kind().header().map_or(ast::Unsafety::Normal, |header| header.unsafety)
    }

    pub fn kind(self) -> FnKind<'a> {
        let item = |p: ItemFnParts<'a>| -> FnKind<'a> {
            FnKind::ItemFn(p.ident, p.generics, p.header, p.vis, p.attrs)
        };
        let closure = |c: ClosureParts<'a>| {
            FnKind::Closure(c.attrs)
        };
        let method = |_, ident: Ident, sig: &'a ast::MethodSig, vis, _, _, attrs| {
            FnKind::Method(ident, sig, vis, attrs)
        };
        self.handle(item, method, closure)
    }

    fn handle<A, I, M, C>(self, item_fn: I, method: M, closure: C) -> A where
        I: FnOnce(ItemFnParts<'a>) -> A,
        M: FnOnce(ast::HirId,
                  Ident,
                  &'a ast::MethodSig,
                  Option<&'a ast::Visibility>,
                  ast::BodyId,
                  Span,
                  &'a [Attribute])
                  -> A,
        C: FnOnce(ClosureParts<'a>) -> A,
    {
        match self.node {
            map::Node::Item(i) => match i.node {
                ast::ItemKind::Fn(ref decl, header, ref generics, block) =>
                    item_fn(ItemFnParts {
                        id: i.hir_id,
                        ident: i.ident,
                        decl: &decl,
                        body: block,
                        vis: &i.vis,
                        span: i.span,
                        attrs: &i.attrs,
                        header,
                        generics,
                    }),
                _ => bug!("item FnLikeNode that is not fn-like"),
            },
            map::Node::TraitItem(ti) => match ti.node {
                ast::TraitItemKind::Method(ref sig, ast::TraitMethod::Provided(body)) => {
                    method(ti.hir_id, ti.ident, sig, None, body, ti.span, &ti.attrs)
                }
                _ => bug!("trait method FnLikeNode that is not fn-like"),
            },
            map::Node::ImplItem(ii) => {
                match ii.node {
                    ast::ImplItemKind::Method(ref sig, body) => {
                        method(ii.hir_id, ii.ident, sig, Some(&ii.vis), body, ii.span, &ii.attrs)
                    }
                    _ => bug!("impl method FnLikeNode that is not fn-like")
                }
            },
            map::Node::Expr(e) => match e.node {
                ast::ExprKind::Closure(_, ref decl, block, _fn_decl_span, _gen) =>
                    closure(ClosureParts::new(&decl, block, e.hir_id, e.span, &e.attrs)),
                _ => bug!("expr FnLikeNode that is not fn-like"),
            },
            _ => bug!("other FnLikeNode that is not fn-like"),
        }
    }
}
