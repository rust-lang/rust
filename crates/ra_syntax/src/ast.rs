//! Abstract Syntax Tree, layered on top of untyped `SyntaxNode`s

mod generated;
mod traits;
mod tokens;
mod extensions;
mod expr_extensions;

use std::marker::PhantomData;

use crate::{
    syntax_node::{SyntaxNode, SyntaxNodeChildren, SyntaxToken},
    SmolStr, SyntaxKind,
};

pub use self::{
    expr_extensions::{ArrayExprKind, BinOp, ElseBranch, LiteralKind, PrefixOp},
    extensions::{FieldKind, PathSegmentKind, SelfParamKind, StructKind},
    generated::*,
    tokens::*,
    traits::*,
};

/// The main trait to go from untyped `SyntaxNode`  to a typed ast. The
/// conversion itself has zero runtime cost: ast and syntax nodes have exactly
/// the same representation: a pointer to the tree root and a pointer to the
/// node itself.
pub trait AstNode: Clone {
    fn can_cast(kind: SyntaxKind) -> bool;

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> &SyntaxNode;
}

/// Like `AstNode`, but wraps tokens rather than interior nodes.
pub trait AstToken {
    fn cast(token: SyntaxToken) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> &SyntaxToken;
    fn text(&self) -> &SmolStr {
        self.syntax().text()
    }
}

/// An iterator over `SyntaxNode` children of a particular AST type.
#[derive(Debug)]
pub struct AstChildren<N> {
    inner: SyntaxNodeChildren,
    ph: PhantomData<N>,
}

impl<N> AstChildren<N> {
    fn new(parent: &SyntaxNode) -> Self {
        AstChildren { inner: parent.children(), ph: PhantomData }
    }
}

impl<N: AstNode> Iterator for AstChildren<N> {
    type Item = N;
    fn next(&mut self) -> Option<N> {
        self.inner.by_ref().find_map(N::cast)
    }
}

fn child_opt<P: AstNode + ?Sized, C: AstNode>(parent: &P) -> Option<C> {
    children(parent).next()
}

fn children<P: AstNode + ?Sized, C: AstNode>(parent: &P) -> AstChildren<C> {
    AstChildren::new(parent.syntax())
}

#[test]
fn test_doc_comment_none() {
    let file = SourceFile::parse(
        r#"
        // non-doc
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert!(module.doc_comment_text().is_none());
}

#[test]
fn test_doc_comment_of_items() {
    let file = SourceFile::parse(
        r#"
        //! doc
        // non-doc
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("doc", module.doc_comment_text().unwrap());
}

#[test]
fn test_doc_comment_preserves_indents() {
    let file = SourceFile::parse(
        r#"
        /// doc1
        /// ```
        /// fn foo() {
        ///     // ...
        /// }
        /// ```
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("doc1\n```\nfn foo() {\n    // ...\n}\n```", module.doc_comment_text().unwrap());
}

#[test]
fn test_doc_comment_preserves_newlines() {
    let file = SourceFile::parse(
        r#"
        /// this
        /// is
        /// mod
        /// foo
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("this\nis\nmod\nfoo", module.doc_comment_text().unwrap());
}

#[test]
fn test_doc_comment_single_line_block_strips_suffix() {
    let file = SourceFile::parse(
        r#"
        /** this is mod foo*/
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("this is mod foo", module.doc_comment_text().unwrap());
}

#[test]
fn test_doc_comment_single_line_block_strips_suffix_whitespace() {
    let file = SourceFile::parse(
        r#"
        /** this is mod foo */
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("this is mod foo", module.doc_comment_text().unwrap());
}

#[test]
fn test_doc_comment_multi_line_block_strips_suffix() {
    let file = SourceFile::parse(
        r#"
        /**
        this
        is
        mod foo
        */
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("        this\n        is\n        mod foo", module.doc_comment_text().unwrap());
}

#[test]
fn test_where_predicates() {
    fn assert_bound(text: &str, bound: Option<TypeBound>) {
        assert_eq!(text, bound.unwrap().syntax().text().to_string());
    }

    let file = SourceFile::parse(
        r#"
fn foo()
where
   T: Clone + Copy + Debug + 'static,
   'a: 'b + 'c,
   Iterator::Item: 'a + Debug,
   Iterator::Item: Debug + 'a,
   <T as Iterator>::Item: Debug + 'a,
   for<'a> F: Fn(&'a str)
{}
        "#,
    )
    .ok()
    .unwrap();
    let where_clause = file.syntax().descendants().find_map(WhereClause::cast).unwrap();

    let mut predicates = where_clause.predicates();

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("T", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Clone", bounds.next());
    assert_bound("Copy", bounds.next());
    assert_bound("Debug", bounds.next());
    assert_bound("'static", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("'a", pred.lifetime_token().unwrap().text());

    assert_bound("'b", bounds.next());
    assert_bound("'c", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("Iterator::Item", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("Iterator::Item", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("<T as Iterator>::Item", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("for<'a> F", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Fn(&'a str)", bounds.next());
}
