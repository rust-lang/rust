//! Abstract Syntax Tree, layered on top of untyped `SyntaxNode`s

mod generated;
mod traits;
mod token_ext;
mod node_ext;
mod expr_ext;
mod operators;
pub mod edit;
pub mod edit_in_place;
pub mod make;
pub mod prec;

use std::marker::PhantomData;

use either::Either;

use crate::{
    syntax_node::{SyntaxNode, SyntaxNodeChildren, SyntaxToken},
    SyntaxKind,
};

pub use self::{
    expr_ext::{ArrayExprKind, BlockModifier, CallableExpr, ElseBranch, LiteralKind},
    generated::{nodes::*, tokens::*},
    node_ext::{
        AttrKind, FieldKind, Macro, NameLike, NameOrNameRef, PathSegmentKind, SelfParamKind,
        SlicePatComponents, StructKind, TraitOrAlias, TypeBoundKind, TypeOrConstParam,
        VisibilityKind,
    },
    operators::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp},
    token_ext::{CommentKind, CommentPlacement, CommentShape, IsString, QuoteOffsets, Radix},
    traits::{
        AttrDocCommentIter, DocCommentIter, HasArgList, HasAttrs, HasDocComments, HasGenericParams,
        HasLoopBody, HasModuleItem, HasName, HasTypeBounds, HasVisibility,
    },
};

/// The main trait to go from untyped `SyntaxNode`  to a typed ast. The
/// conversion itself has zero runtime cost: ast and syntax nodes have exactly
/// the same representation: a pointer to the tree root and a pointer to the
/// node itself.
pub trait AstNode {
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized;

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized;

    fn syntax(&self) -> &SyntaxNode;
    fn clone_for_update(&self) -> Self
    where
        Self: Sized,
    {
        Self::cast(self.syntax().clone_for_update()).unwrap()
    }
    fn clone_subtree(&self) -> Self
    where
        Self: Sized,
    {
        Self::cast(self.syntax().clone_subtree()).unwrap()
    }
}

/// Like `AstNode`, but wraps tokens rather than interior nodes.
pub trait AstToken {
    fn can_cast(token: SyntaxKind) -> bool
    where
        Self: Sized;

    fn cast(syntax: SyntaxToken) -> Option<Self>
    where
        Self: Sized;

    fn syntax(&self) -> &SyntaxToken;

    fn text(&self) -> &str {
        self.syntax().text()
    }
}

/// An iterator over `SyntaxNode` children of a particular AST type.
#[derive(Debug, Clone)]
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
        self.inner.find_map(N::cast)
    }
}

impl<L, R> AstNode for Either<L, R>
where
    L: AstNode,
    R: AstNode,
{
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        L::can_cast(kind) || R::can_cast(kind)
    }

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized,
    {
        if L::can_cast(syntax.kind()) {
            L::cast(syntax).map(Either::Left)
        } else {
            R::cast(syntax).map(Either::Right)
        }
    }

    fn syntax(&self) -> &SyntaxNode {
        self.as_ref().either(L::syntax, R::syntax)
    }
}

impl<L, R> HasAttrs for Either<L, R>
where
    L: HasAttrs,
    R: HasAttrs,
{
}

mod support {
    use super::{AstChildren, AstNode, SyntaxKind, SyntaxNode, SyntaxToken};

    pub(super) fn child<N: AstNode>(parent: &SyntaxNode) -> Option<N> {
        parent.children().find_map(N::cast)
    }

    pub(super) fn children<N: AstNode>(parent: &SyntaxNode) -> AstChildren<N> {
        AstChildren::new(parent)
    }

    pub(super) fn token(parent: &SyntaxNode, kind: SyntaxKind) -> Option<SyntaxToken> {
        parent.children_with_tokens().filter_map(|it| it.into_token()).find(|it| it.kind() == kind)
    }
}

#[test]
fn assert_ast_is_object_safe() {
    fn _f(_: &dyn AstNode, _: &dyn HasName) {}
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
    assert!(module.doc_comments().doc_comment_text().is_none());
}

#[test]
fn test_outer_doc_comment_of_items() {
    let file = SourceFile::parse(
        r#"
        /// doc
        // non-doc
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(" doc", module.doc_comments().doc_comment_text().unwrap());
}

#[test]
fn test_inner_doc_comment_of_items() {
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
    assert!(module.doc_comments().doc_comment_text().is_none());
}

#[test]
fn test_doc_comment_of_statics() {
    let file = SourceFile::parse(
        r#"
        /// Number of levels
        static LEVELS: i32 = 0;
        "#,
    )
    .ok()
    .unwrap();
    let st = file.syntax().descendants().find_map(Static::cast).unwrap();
    assert_eq!(" Number of levels", st.doc_comments().doc_comment_text().unwrap());
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
    assert_eq!(
        " doc1\n ```\n fn foo() {\n     // ...\n }\n ```",
        module.doc_comments().doc_comment_text().unwrap()
    );
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
    assert_eq!(" this\n is\n mod\n foo", module.doc_comments().doc_comment_text().unwrap());
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
    assert_eq!(" this is mod foo", module.doc_comments().doc_comment_text().unwrap());
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
    assert_eq!(" this is mod foo ", module.doc_comments().doc_comment_text().unwrap());
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
    assert_eq!(
        "\n        this\n        is\n        mod foo\n        ",
        module.doc_comments().doc_comment_text().unwrap()
    );
}

#[test]
fn test_comments_preserve_trailing_whitespace() {
    let file = SourceFile::parse(
        "\n/// Representation of a Realm.   \n/// In the specification these are called Realm Records.\nstruct Realm {}",
    )
    .ok()
    .unwrap();
    let def = file.syntax().descendants().find_map(Struct::cast).unwrap();
    assert_eq!(
        " Representation of a Realm.   \n In the specification these are called Realm Records.",
        def.doc_comments().doc_comment_text().unwrap()
    );
}

#[test]
fn test_four_slash_line_comment() {
    let file = SourceFile::parse(
        r#"
        //// too many slashes to be a doc comment
        /// doc comment
        mod foo {}
        "#,
    )
    .ok()
    .unwrap();
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!(" doc comment", module.doc_comments().doc_comment_text().unwrap());
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

    assert!(pred.for_token().is_none());
    assert!(pred.generic_param_list().is_none());
    assert_eq!("T", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Clone", bounds.next());
    assert_bound("Copy", bounds.next());
    assert_bound("Debug", bounds.next());
    assert_bound("'static", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("'a", pred.lifetime().unwrap().lifetime_ident_token().unwrap().text());

    assert_bound("'b", bounds.next());
    assert_bound("'c", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("Iterator::Item", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("Iterator::Item", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("<T as Iterator>::Item", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert!(pred.for_token().is_some());
    assert_eq!("<'a>", pred.generic_param_list().unwrap().syntax().text().to_string());
    assert_eq!("F", pred.ty().unwrap().syntax().text().to_string());
    assert_bound("Fn(&'a str)", bounds.next());
}
