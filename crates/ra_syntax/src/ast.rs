//! Abstract Syntax Tree, layered on top of untyped `SyntaxNode`s

mod generated;
mod traits;
mod tokens;
mod extensions;
mod expr_extensions;

use std::marker::PhantomData;

use crate::{
    syntax_node::{SyntaxNode, SyntaxNodeChildren, TreeArc, SyntaxToken},
    SmolStr,
};

pub use self::{
    generated::*,
    traits::*,
    tokens::*,
    extensions::{PathSegmentKind, StructKind,FieldKind, SelfParamKind},
    expr_extensions::{ElseBranch, PrefixOp, BinOp, LiteralKind,ArrayExprKind},
};

/// The main trait to go from untyped `SyntaxNode`  to a typed ast. The
/// conversion itself has zero runtime cost: ast and syntax nodes have exactly
/// the same representation: a pointer to the tree root and a pointer to the
/// node itself.
pub trait AstNode:
    rowan::TransparentNewType<Repr = rowan::SyntaxNode> + ToOwned<Owned = TreeArc<Self>>
{
    fn cast(syntax: &SyntaxNode) -> Option<&Self>
    where
        Self: Sized;
    fn syntax(&self) -> &SyntaxNode;
}

/// Like `AstNode`, but wraps tokens rather than interior nodes.
pub trait AstToken<'a> {
    fn cast(token: SyntaxToken<'a>) -> Option<Self>
    where
        Self: Sized;
    fn syntax(&self) -> SyntaxToken<'a>;
    fn text(&self) -> &'a SmolStr {
        self.syntax().text()
    }
}

/// An iterator over `SyntaxNode` children of a particular AST type.
#[derive(Debug)]
pub struct AstChildren<'a, N> {
    inner: SyntaxNodeChildren<'a>,
    ph: PhantomData<N>,
}

impl<'a, N> AstChildren<'a, N> {
    fn new(parent: &'a SyntaxNode) -> Self {
        AstChildren { inner: parent.children(), ph: PhantomData }
    }
}

impl<'a, N: AstNode + 'a> Iterator for AstChildren<'a, N> {
    type Item = &'a N;
    fn next(&mut self) -> Option<&'a N> {
        self.inner.by_ref().find_map(N::cast)
    }
}

fn child_opt<P: AstNode, C: AstNode>(parent: &P) -> Option<&C> {
    children(parent).next()
}

fn children<P: AstNode, C: AstNode>(parent: &P) -> AstChildren<C> {
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
fn test_where_predicates() {
    fn assert_bound(text: &str, bound: Option<&TypeBound>) {
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
