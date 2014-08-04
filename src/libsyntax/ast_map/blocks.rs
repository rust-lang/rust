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

use abi;
use ast::{P, Block, FnDecl, NodeId};
use ast;
use ast_map::{Node};
use ast_map;
use ast_util::PostExpansionMethod;
use codemap::Span;
use visit;

/// An FnLikeNode is a Node that is like a fn, in that it has a decl
/// and a body (as well as a NodeId, a span, etc).
///
/// More specifically, it is one of either:
///   - A function item,
///   - A closure expr (i.e. an ExprFnBlock or ExprProc), or
///   - The default implementation for a trait method.
///
/// To construct one, use the `Code::from_node` function.
pub struct FnLikeNode { node: ast_map::Node }

/// MaybeFnLike wraps a method that indicates if an object
/// corresponds to some FnLikeNode.
pub trait MaybeFnLike { fn is_fn_like(&self) -> bool; }

/// Components shared by fn-like things (fn items, methods, closures).
pub struct FnParts<'a> {
    pub decl: P<FnDecl>,
    pub body: P<Block>,
    pub kind: visit::FnKind<'a>,
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
        match *self { ast::ProvidedMethod(_) => true, _ => false, }
    }
}

impl MaybeFnLike for ast::Expr {
    fn is_fn_like(&self) -> bool {
        match self.node {
            ast::ExprFnBlock(..) | ast::ExprProc(..) => true,
            _ => false,
        }
    }
}

/// Carries either an FnLikeNode or a Block, as these are the two
/// constructs that correspond to "code" (as in, something from which
/// we can construct a control-flow graph).
pub enum Code {
    FnLikeCode(FnLikeNode),
    BlockCode(P<Block>),
}

impl Code {
    pub fn id(&self) -> ast::NodeId {
        match *self {
            FnLikeCode(node) => node.id(),
            BlockCode(block) => block.id,
        }
    }

    /// Attempts to construct a Code from presumed FnLike or Block node input.
    pub fn from_node(node: Node) -> Option<Code> {
        fn new(node: Node) -> FnLikeNode { FnLikeNode { node: node } }
        match node {
            ast_map::NodeItem(item) if item.is_fn_like() =>
                Some(FnLikeCode(new(node))),
            ast_map::NodeTraitItem(tm) if tm.is_fn_like() =>
                Some(FnLikeCode(new(node))),
            ast_map::NodeImplItem(_) =>
                Some(FnLikeCode(new(node))),
            ast_map::NodeExpr(e) if e.is_fn_like() =>
                Some(FnLikeCode(new(node))),
            ast_map::NodeBlock(block) =>
                Some(BlockCode(block)),
            _ =>
                None,
        }
    }
}

/// These are all the components one can extract from a fn item for
/// use when implementing FnLikeNode operations.
struct ItemFnParts<'a> {
    ident:    ast::Ident,
    decl:     P<ast::FnDecl>,
    style:    ast::FnStyle,
    abi:      abi::Abi,
    generics: &'a ast::Generics,
    body:     P<Block>,
    id:       ast::NodeId,
    span:     Span
}

/// These are all the components one can extract from a closure expr
/// for use when implementing FnLikeNode operations.
struct ClosureParts {
    decl: P<FnDecl>,
    body: P<Block>,
    id: NodeId,
    span: Span
}

impl ClosureParts {
    fn new(d: P<FnDecl>, b: P<Block>, id: NodeId, s: Span) -> ClosureParts {
        ClosureParts { decl: d, body: b, id: id, span: s }
    }
}

impl FnLikeNode {
    pub fn to_fn_parts<'a>(&'a self) -> FnParts<'a> {
        FnParts {
            decl: self.decl(),
            body: self.body(),
            kind: self.kind(),
            span: self.span(),
            id:   self.id(),
        }
    }

    pub fn body<'a>(&'a self) -> P<Block> {
        self.handle(|i: ItemFnParts|     i.body,
                    |m: &'a ast::Method| m.pe_body(),
                    |c: ClosureParts|    c.body)
    }

    pub fn decl<'a>(&'a self) -> P<FnDecl> {
        self.handle(|i: ItemFnParts|     i.decl,
                    |m: &'a ast::Method| m.pe_fn_decl(),
                    |c: ClosureParts|    c.decl)
    }

    pub fn span<'a>(&'a self) -> Span {
        self.handle(|i: ItemFnParts|     i.span,
                    |m: &'a ast::Method| m.span,
                    |c: ClosureParts|    c.span)
    }

    pub fn id<'a>(&'a self) -> NodeId {
        self.handle(|i: ItemFnParts|     i.id,
                    |m: &'a ast::Method| m.id,
                    |c: ClosureParts|    c.id)
    }

    pub fn kind<'a>(&'a self) -> visit::FnKind<'a> {
        let item = |p: ItemFnParts<'a>| -> visit::FnKind<'a> {
            visit::FkItemFn(p.ident, p.generics, p.style, p.abi)
        };
        let closure = |_: ClosureParts| {
            visit::FkFnBlock
        };
        let method = |m: &'a ast::Method| {
            visit::FkMethod(m.pe_ident(), m.pe_generics(), m)
        };
        self.handle(item, method, closure)
    }

    fn handle<'a, A>(&'a self,
                     item_fn: |ItemFnParts<'a>| -> A,
                     method: |&'a ast::Method| -> A,
                     closure: |ClosureParts| -> A) -> A {
        match self.node {
            ast_map::NodeItem(ref i) => match i.node {
                ast::ItemFn(decl, style, abi, ref generics, block) =>
                    item_fn(ItemFnParts{
                        ident: i.ident, decl: decl, style: style, body: block,
                        generics: generics, abi: abi, id: i.id, span: i.span
                    }),
                _ => fail!("item FnLikeNode that is not fn-like"),
            },
            ast_map::NodeTraitItem(ref t) => match **t {
                ast::ProvidedMethod(ref m) => method(&**m),
                _ => fail!("trait method FnLikeNode that is not fn-like"),
            },
            ast_map::NodeImplItem(ref ii) => {
                match **ii {
                    ast::MethodImplItem(ref m) => method(&**m),
                }
            }
            ast_map::NodeExpr(ref e) => match e.node {
                ast::ExprFnBlock(_, ref decl, ref block) =>
                    closure(ClosureParts::new(*decl, *block, e.id, e.span)),
                ast::ExprProc(ref decl, ref block) =>
                    closure(ClosureParts::new(*decl, *block, e.id, e.span)),
                _ => fail!("expr FnLikeNode that is not fn-like"),
            },
            _ => fail!("other FnLikeNode that is not fn-like"),
        }
    }
}
