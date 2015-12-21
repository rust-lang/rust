// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides a simplified abstraction for working with
//! code blocks identified by their integer node-id.  In particular,
//! it captures a common set of attributes that all "function-like
//! things" (represented by `FnLike` instances) share.  For example,
//! all `FnLike` instances have a type signature (be it explicit or
//! inferred).  And all `FnLike` instances have a body, i.e. the code
//! that is run when the function-like thing it represents is invoked.
//!
//! With the above abstraction in place, one can treat the program
//! text as a collection of blocks of code (and most such blocks are
//! nested within a uniquely determined `FnLike`), and users can ask
//! for the `Code` associated with a particular NodeId.

pub use self::Code::*;

use front::map::{self, Node};
use syntax::abi;
use rustc_front::hir::{Block, FnDecl};
use syntax::ast::{Name, NodeId};
use rustc_front::hir as ast;
use syntax::codemap::Span;
use rustc_front::intravisit::FnKind;

/// An FnLikeNode is a Node that is like a fn, in that it has a decl
/// and a body (as well as a NodeId, a span, etc).
///
/// More specifically, it is one of either:
///   - A function item,
///   - A closure expr (i.e. an ExprClosure), or
///   - The default implementation for a trait method.
///
/// To construct one, use the `Code::from_node` function.
#[derive(Copy, Clone)]
pub struct FnLikeNode<'a> { node: map::Node<'a> }

/// MaybeFnLike wraps a method that indicates if an object
/// corresponds to some FnLikeNode.
pub trait MaybeFnLike { fn is_fn_like(&self) -> bool; }

/// Components shared by fn-like things (fn items, methods, closures).
pub struct FnParts<'a> {
    pub decl: &'a FnDecl,
    pub body: &'a Block,
    pub kind: FnKind<'a>,
    pub span: Span,
    pub id:   NodeId,
}

impl MaybeFnLike for ast::Item {
    fn is_fn_like(&self) -> bool {
        match self.node { ast::ItemFn(..) => true, _ => false, }
    }
}

impl MaybeFnLike for ast::TraitItem {
    fn is_fn_like(&self) -> bool {
        match self.node { ast::MethodTraitItem(_, Some(_)) => true, _ => false, }
    }
}

impl MaybeFnLike for ast::Expr {
    fn is_fn_like(&self) -> bool {
        match self.node {
            ast::ExprClosure(..) => true,
            _ => false,
        }
    }
}

/// Carries either an FnLikeNode or a Block, as these are the two
/// constructs that correspond to "code" (as in, something from which
/// we can construct a control-flow graph).
#[derive(Copy, Clone)]
pub enum Code<'a> {
    FnLikeCode(FnLikeNode<'a>),
    BlockCode(&'a Block),
}

impl<'a> Code<'a> {
    pub fn id(&self) -> NodeId {
        match *self {
            FnLikeCode(node) => node.id(),
            BlockCode(block) => block.id,
        }
    }

    /// Attempts to construct a Code from presumed FnLike or Block node input.
    pub fn from_node(node: Node) -> Option<Code> {
        if let map::NodeBlock(block) = node {
            Some(BlockCode(block))
        } else {
            FnLikeNode::from_node(node).map(|fn_like| FnLikeCode(fn_like))
        }
    }
}

/// These are all the components one can extract from a fn item for
/// use when implementing FnLikeNode operations.
struct ItemFnParts<'a> {
    name:     Name,
    decl:     &'a ast::FnDecl,
    unsafety: ast::Unsafety,
    constness: ast::Constness,
    abi:      abi::Abi,
    vis:      ast::Visibility,
    generics: &'a ast::Generics,
    body:     &'a Block,
    id:       NodeId,
    span:     Span
}

/// These are all the components one can extract from a closure expr
/// for use when implementing FnLikeNode operations.
struct ClosureParts<'a> {
    decl: &'a FnDecl,
    body: &'a Block,
    id: NodeId,
    span: Span
}

impl<'a> ClosureParts<'a> {
    fn new(d: &'a FnDecl, b: &'a Block, id: NodeId, s: Span) -> ClosureParts<'a> {
        ClosureParts {
            decl: d,
            body: b,
            id: id,
            span: s
        }
    }
}

impl<'a> FnLikeNode<'a> {
    /// Attempts to construct a FnLikeNode from presumed FnLike node input.
    pub fn from_node(node: Node) -> Option<FnLikeNode> {
        let fn_like = match node {
            map::NodeItem(item) => item.is_fn_like(),
            map::NodeTraitItem(tm) => tm.is_fn_like(),
            map::NodeImplItem(_) => true,
            map::NodeExpr(e) => e.is_fn_like(),
            _ => false
        };
        if fn_like {
            Some(FnLikeNode {
                node: node
            })
        } else {
            None
        }
    }

    pub fn to_fn_parts(self) -> FnParts<'a> {
        FnParts {
            decl: self.decl(),
            body: self.body(),
            kind: self.kind(),
            span: self.span(),
            id:   self.id(),
        }
    }

    pub fn body(self) -> &'a Block {
        self.handle(|i: ItemFnParts<'a>|  &*i.body,
                    |_, _, _: &'a ast::MethodSig, _, body: &'a ast::Block, _|  body,
                    |c: ClosureParts<'a>| c.body)
    }

    pub fn decl(self) -> &'a FnDecl {
        self.handle(|i: ItemFnParts<'a>|  &*i.decl,
                    |_, _, sig: &'a ast::MethodSig, _, _, _|  &sig.decl,
                    |c: ClosureParts<'a>| c.decl)
    }

    pub fn span(self) -> Span {
        self.handle(|i: ItemFnParts|     i.span,
                    |_, _, _: &'a ast::MethodSig, _, _, span| span,
                    |c: ClosureParts|    c.span)
    }

    pub fn id(self) -> NodeId {
        self.handle(|i: ItemFnParts|     i.id,
                    |id, _, _: &'a ast::MethodSig, _, _, _| id,
                    |c: ClosureParts|    c.id)
    }

    pub fn kind(self) -> FnKind<'a> {
        let item = |p: ItemFnParts<'a>| -> FnKind<'a> {
            FnKind::ItemFn(p.name, p.generics, p.unsafety, p.constness, p.abi, p.vis)
        };
        let closure = |_: ClosureParts| {
            FnKind::Closure
        };
        let method = |_, name: Name, sig: &'a ast::MethodSig, vis, _, _| {
            FnKind::Method(name, sig, vis)
        };
        self.handle(item, method, closure)
    }

    fn handle<A, I, M, C>(self, item_fn: I, method: M, closure: C) -> A where
        I: FnOnce(ItemFnParts<'a>) -> A,
        M: FnOnce(NodeId,
                  Name,
                  &'a ast::MethodSig,
                  Option<ast::Visibility>,
                  &'a ast::Block,
                  Span)
                  -> A,
        C: FnOnce(ClosureParts<'a>) -> A,
    {
        match self.node {
            map::NodeItem(i) => match i.node {
                ast::ItemFn(ref decl, unsafety, constness, abi, ref generics, ref block) =>
                    item_fn(ItemFnParts {
                        id: i.id,
                        name: i.name,
                        decl: &**decl,
                        unsafety: unsafety,
                        body: &**block,
                        generics: generics,
                        abi: abi,
                        vis: i.vis,
                        constness: constness,
                        span: i.span
                    }),
                _ => panic!("item FnLikeNode that is not fn-like"),
            },
            map::NodeTraitItem(ti) => match ti.node {
                ast::MethodTraitItem(ref sig, Some(ref body)) => {
                    method(ti.id, ti.name, sig, None, body, ti.span)
                }
                _ => panic!("trait method FnLikeNode that is not fn-like"),
            },
            map::NodeImplItem(ii) => {
                match ii.node {
                    ast::ImplItemKind::Method(ref sig, ref body) => {
                        method(ii.id, ii.name, sig, Some(ii.vis), body, ii.span)
                    }
                    _ => {
                        panic!("impl method FnLikeNode that is not fn-like")
                    }
                }
            }
            map::NodeExpr(e) => match e.node {
                ast::ExprClosure(_, ref decl, ref block) =>
                    closure(ClosureParts::new(&**decl, &**block, e.id, e.span)),
                _ => panic!("expr FnLikeNode that is not fn-like"),
            },
            _ => panic!("other FnLikeNode that is not fn-like"),
        }
    }
}
