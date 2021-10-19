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

use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::Node;

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
pub struct FnLikeNode<'a> {
    node: Node<'a>,
}

impl<'a> FnLikeNode<'a> {
    /// Attempts to construct a FnLikeNode from presumed FnLike node input.
    pub fn from_node(node: Node<'_>) -> Option<FnLikeNode<'_>> {
        let fn_like = match node {
            Node::Item(item) => matches!(item.kind, hir::ItemKind::Fn(..)),
            Node::TraitItem(tm) => {
                matches!(tm.kind, hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(_)))
            }
            Node::ImplItem(it) => matches!(it.kind, hir::ImplItemKind::Fn(..)),
            Node::Expr(e) => matches!(e.kind, hir::ExprKind::Closure(..)),
            _ => false,
        };
        fn_like.then_some(FnLikeNode { node })
    }

    pub fn constness(self) -> hir::Constness {
        self.kind().header().map_or(hir::Constness::NotConst, |header| header.constness)
    }

    pub fn asyncness(self) -> hir::IsAsync {
        self.kind().header().map_or(hir::IsAsync::NotAsync, |header| header.asyncness)
    }

    pub fn kind(self) -> FnKind<'a> {
        match self.node {
            Node::Item(i) => match i.kind {
                hir::ItemKind::Fn(ref sig, ref generics, _) => {
                    FnKind::ItemFn(i.ident, generics, sig.header, &i.vis)
                }
                _ => bug!("item FnLikeNode that is not fn-like"),
            },
            Node::TraitItem(ti) => match ti.kind {
                hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(_)) => {
                    FnKind::Method(ti.ident, sig, None)
                }
                _ => bug!("trait method FnLikeNode that is not fn-like"),
            },
            Node::ImplItem(ii) => match ii.kind {
                hir::ImplItemKind::Fn(ref sig, _) => FnKind::Method(ii.ident, sig, Some(&ii.vis)),
                _ => bug!("impl method FnLikeNode that is not fn-like"),
            },
            Node::Expr(e) => match e.kind {
                hir::ExprKind::Closure(..) => FnKind::Closure,
                _ => bug!("expr FnLikeNode that is not fn-like"),
            },
            _ => bug!("other FnLikeNode that is not fn-like"),
        }
    }
}
