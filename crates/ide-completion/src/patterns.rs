//! Patterns telling us certain facts about current syntax element, they are used in completion context
//!
//! Most logic in this module first expands the token below the cursor to a maximum node that acts similar to the token itself.
//! This means we for example expand a NameRef token to its outermost Path node, as semantically these act in the same location
//! and the completions usually query for path specific things on the Path context instead. This simplifies some location handling.

use hir::Semantics;
use ide_db::RootDatabase;
use syntax::{
    ast::{self, HasLoopBody},
    match_ast, AstNode, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, TextSize,
};

#[cfg(test)]
use crate::tests::check_pattern_is_applicable;

/// Direct parent "thing" of what we are currently completing.
///
/// This may contain nodes of the fake file as well as the original, comments on the variants specify
/// from which file the nodes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ImmediateLocation {
    TypeBound,
    // Only set from a type arg
    /// Original file ast node
    GenericArgList(ast::GenericArgList),
}

pub(crate) fn determine_location(
    sema: &Semantics<RootDatabase>,
    original_file: &SyntaxNode,
    offset: TextSize,
    name_like: &ast::NameLike,
) -> Option<ImmediateLocation> {
    let node = match name_like {
        ast::NameLike::NameRef(name_ref) => maximize_name_ref(name_ref),
        ast::NameLike::Name(name) => name.syntax().clone(),
        ast::NameLike::Lifetime(lt) => lt.syntax().clone(),
    };

    match_ast! {
        match node {
            ast::TypeBoundList(_it) => return Some(ImmediateLocation::TypeBound),
            _ => (),
        }
    };

    let parent = match node.parent() {
        Some(parent) => match ast::MacroCall::cast(parent.clone()) {
            // When a path is being typed in an (Assoc)ItemList the parser will always emit a macro_call.
            // This is usually fine as the node expansion code above already accounts for that with
            // the ancestors call, but there is one exception to this which is that when an attribute
            // precedes it the code above will not walk the Path to the parent MacroCall as their ranges differ.
            // FIXME path expr and statement have a similar problem
            Some(call)
                if call.excl_token().is_none()
                    && call.token_tree().is_none()
                    && call.semicolon_token().is_none() =>
            {
                call.syntax().parent()?
            }
            _ => parent,
        },
        // SourceFile
        None => return None,
    };

    let res = match_ast! {
        match parent {
            ast::TypeBound(_) => ImmediateLocation::TypeBound,
            ast::TypeBoundList(_) => ImmediateLocation::TypeBound,
            ast::GenericArgList(_) => sema
                .find_node_at_offset_with_macros(original_file, offset)
                .map(ImmediateLocation::GenericArgList)?,
            _ => return None,
        }
    };
    Some(res)
}

/// Maximize a nameref to its enclosing path if its the last segment of said path.
/// That is, when completing a [`NameRef`] we actually handle it as the path it is part of when determining
/// its location.
fn maximize_name_ref(name_ref: &ast::NameRef) -> SyntaxNode {
    if let Some(segment) = name_ref.syntax().parent().and_then(ast::PathSegment::cast) {
        let p = segment.parent_path();
        if p.parent_path().is_none() {
            // Get rid of PathExpr, PathType, etc...
            let path = p
                .syntax()
                .ancestors()
                .take_while(|it| it.text_range() == p.syntax().text_range())
                .last();
            if let Some(it) = path {
                return it;
            }
        }
    }
    name_ref.syntax().clone()
}

pub(crate) fn previous_token(element: SyntaxElement) -> Option<SyntaxToken> {
    element.into_token().and_then(previous_non_trivia_token)
}

pub(crate) fn is_in_token_of_for_loop(element: SyntaxElement) -> bool {
    // oh my ...
    (|| {
        let syntax_token = element.into_token()?;
        let range = syntax_token.text_range();
        let for_expr = syntax_token.parent_ancestors().find_map(ast::ForExpr::cast)?;

        // check if the current token is the `in` token of a for loop
        if let Some(token) = for_expr.in_token() {
            return Some(syntax_token == token);
        }
        let pat = for_expr.pat()?;
        if range.end() < pat.syntax().text_range().end() {
            // if we are inside or before the pattern we can't be at the `in` token position
            return None;
        }
        let next_sibl = next_non_trivia_sibling(pat.syntax().clone().into())?;
        Some(match next_sibl {
            // the loop body is some node, if our token is at the start we are at the `in` position,
            // otherwise we could be in a recovered expression, we don't wanna ruin completions there
            syntax::NodeOrToken::Node(n) => n.text_range().start() == range.start(),
            // the loop body consists of a single token, if we are this we are certainly at the `in` token position
            syntax::NodeOrToken::Token(t) => t == syntax_token,
        })
    })()
    .unwrap_or(false)
}

#[test]
fn test_for_is_prev2() {
    check_pattern_is_applicable(r"fn __() { for i i$0 }", is_in_token_of_for_loop);
}

pub(crate) fn is_in_loop_body(node: &SyntaxNode) -> bool {
    node.ancestors()
        .take_while(|it| it.kind() != FN && it.kind() != CLOSURE_EXPR)
        .find_map(|it| {
            let loop_body = match_ast! {
                match it {
                    ast::ForExpr(it) => it.loop_body(),
                    ast::WhileExpr(it) => it.loop_body(),
                    ast::LoopExpr(it) => it.loop_body(),
                    _ => None,
                }
            };
            loop_body.filter(|it| it.syntax().text_range().contains_range(node.text_range()))
        })
        .is_some()
}

fn previous_non_trivia_token(token: SyntaxToken) -> Option<SyntaxToken> {
    let mut token = token.prev_token();
    while let Some(inner) = token {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.prev_token();
        }
    }
    None
}

fn next_non_trivia_sibling(ele: SyntaxElement) -> Option<SyntaxElement> {
    let mut e = ele.next_sibling_or_token();
    while let Some(inner) = e {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            e = inner.next_sibling_or_token();
        }
    }
    None
}
