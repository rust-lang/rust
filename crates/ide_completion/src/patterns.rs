//! Patterns telling us certain facts about current syntax element, they are used in completion context

use syntax::{
    algo::non_trivia_sibling,
    ast::{self, LoopBodyOwner},
    match_ast, AstNode, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, T,
};

#[cfg(test)]
use crate::test_utils::{check_pattern_is_applicable, check_pattern_is_not_applicable};

/// Direct parent container of the cursor position
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ImmediateLocation {
    Use,
    Impl,
    Trait,
    RecordField,
    RefExpr,
    IdentPat,
    BlockExpr,
    ItemList,
}

pub(crate) fn determine_location(tok: SyntaxToken) -> Option<ImmediateLocation> {
    // First walk the element we are completing up to its highest node that has the same text range
    // as the element so that we can check in what context it immediately lies. We only do this for
    // NameRef -> Path as that's the only thing that makes sense to being "expanded" semantically.
    // We only wanna do this if the NameRef is the last segment of the path.
    let node = match tok.parent().and_then(ast::NameLike::cast)? {
        ast::NameLike::NameRef(name_ref) => {
            if let Some(segment) = name_ref.syntax().parent().and_then(ast::PathSegment::cast) {
                let p = segment.parent_path();
                if p.parent_path().is_none() {
                    p.syntax()
                        .ancestors()
                        .take_while(|it| it.text_range() == p.syntax().text_range())
                        .last()?
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        it @ ast::NameLike::Name(_) | it @ ast::NameLike::Lifetime(_) => it.syntax().clone(),
    };
    let parent = match node.parent() {
        Some(parent) => match ast::MacroCall::cast(parent.clone()) {
            // When a path is being typed in an (Assoc)ItemList the parser will always emit a macro_call.
            // This is usually fine as the node expansion code above already accounts for that with
            // the ancestors call, but there is one exception to this which is that when an attribute
            // precedes it the code above will not walk the Path to the parent MacroCall as their ranges differ.
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
        None => {
            return match node.kind() {
                MACRO_ITEMS | SOURCE_FILE => Some(ImmediateLocation::ItemList),
                _ => None,
            }
        }
    };
    let res = match_ast! {
        match parent {
            ast::IdentPat(_it) => ImmediateLocation::IdentPat,
            ast::Use(_it) => ImmediateLocation::Use,
            ast::BlockExpr(_it) => ImmediateLocation::BlockExpr,
            ast::SourceFile(_it) => ImmediateLocation::ItemList,
            ast::ItemList(_it) => ImmediateLocation::ItemList,
            ast::RefExpr(_it) => ImmediateLocation::RefExpr,
            ast::RecordField(_it) => ImmediateLocation::RecordField,
            ast::AssocItemList(it) => match it.syntax().parent().map(|it| it.kind()) {
                Some(IMPL) => ImmediateLocation::Impl,
                Some(TRAIT) => ImmediateLocation::Trait,
                _ => return None,
            },
            _ => return None,
        }
    };
    Some(res)
}

#[cfg(test)]
fn check_location(code: &str, loc: ImmediateLocation) {
    check_pattern_is_applicable(code, |e| {
        assert_eq!(determine_location(e.into_token().expect("Expected a token")), Some(loc));
        true
    });
}

#[test]
fn test_has_trait_parent() {
    check_location(r"trait A { f$0 }", ImmediateLocation::Trait);
}

#[test]
fn test_has_use_parent() {
    check_location(r"use f$0", ImmediateLocation::Use);
}

#[test]
fn test_has_impl_parent() {
    check_location(r"impl A { f$0 }", ImmediateLocation::Impl);
}
#[test]
fn test_has_field_list_parent() {
    check_location(r"struct Foo { f$0 }", ImmediateLocation::RecordField);
    check_location(r"struct Foo { f$0 pub f: i32}", ImmediateLocation::RecordField);
}

#[test]
fn test_has_block_expr_parent() {
    check_location(r"fn my_fn() { let a = 2; f$0 }", ImmediateLocation::BlockExpr);
}

#[test]
fn test_has_ident_pat_parent() {
    check_location(r"fn my_fn(m$0) {}", ImmediateLocation::IdentPat);
    check_location(r"fn my_fn() { let m$0 }", ImmediateLocation::IdentPat);
    check_location(r"fn my_fn(&m$0) {}", ImmediateLocation::IdentPat);
    check_location(r"fn my_fn() { let &m$0 }", ImmediateLocation::IdentPat);
}

#[test]
fn test_has_ref_expr_parent() {
    check_location(r"fn my_fn() { let x = &m$0 foo; }", ImmediateLocation::RefExpr);
}

#[test]
fn test_has_item_list_or_source_file_parent() {
    check_location(r"i$0", ImmediateLocation::ItemList);
    check_location(r"mod foo { f$0 }", ImmediateLocation::ItemList);
}

pub(crate) fn inside_impl_trait_block(element: SyntaxElement) -> bool {
    // Here we search `impl` keyword up through the all ancestors, unlike in `has_impl_parent`,
    // where we only check the first parent with different text range.
    element
        .ancestors()
        .find(|it| it.kind() == IMPL)
        .map(|it| ast::Impl::cast(it).unwrap())
        .map(|it| it.trait_().is_some())
        .unwrap_or(false)
}
#[test]
fn test_inside_impl_trait_block() {
    check_pattern_is_applicable(r"impl Foo for Bar { f$0 }", inside_impl_trait_block);
    check_pattern_is_applicable(r"impl Foo for Bar { fn f$0 }", inside_impl_trait_block);
    check_pattern_is_not_applicable(r"impl A { f$0 }", inside_impl_trait_block);
    check_pattern_is_not_applicable(r"impl A { fn f$0 }", inside_impl_trait_block);
}

pub(crate) fn is_match_arm(element: SyntaxElement) -> bool {
    not_same_range_ancestor(element.clone()).filter(|it| it.kind() == MATCH_ARM).is_some()
        && previous_sibling_or_ancestor_sibling(element)
            .and_then(|it| it.into_token())
            .filter(|it| it.kind() == FAT_ARROW)
            .is_some()
}
#[test]
fn test_is_match_arm() {
    check_pattern_is_applicable(r"fn my_fn() { match () { () => m$0 } }", is_match_arm);
}

pub(crate) fn previous_token(element: SyntaxElement) -> Option<SyntaxToken> {
    element.into_token().and_then(|it| previous_non_trivia_token(it))
}

/// Check if the token previous to the previous one is `for`.
/// For example, `for _ i$0` => true.
pub(crate) fn for_is_prev2(element: SyntaxElement) -> bool {
    element
        .into_token()
        .and_then(|it| previous_non_trivia_token(it))
        .and_then(|it| previous_non_trivia_token(it))
        .filter(|it| it.kind() == T![for])
        .is_some()
}
#[test]
fn test_for_is_prev2() {
    check_pattern_is_applicable(r"for i i$0", for_is_prev2);
}

pub(crate) fn has_prev_sibling(element: SyntaxElement, kind: SyntaxKind) -> bool {
    previous_sibling_or_ancestor_sibling(element).filter(|it| it.kind() == kind).is_some()
}
#[test]
fn test_has_impl_as_prev_sibling() {
    check_pattern_is_applicable(r"impl A w$0 {}", |it| has_prev_sibling(it, IMPL));
}

pub(crate) fn is_in_loop_body(element: SyntaxElement) -> bool {
    element
        .ancestors()
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
            loop_body.filter(|it| it.syntax().text_range().contains_range(element.text_range()))
        })
        .is_some()
}

pub(crate) fn not_same_range_ancestor(element: SyntaxElement) -> Option<SyntaxNode> {
    element.ancestors().skip_while(|it| it.text_range() == element.text_range()).next()
}

fn previous_non_trivia_token(token: SyntaxToken) -> Option<SyntaxToken> {
    let mut token = token.prev_token();
    while let Some(inner) = token.clone() {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.prev_token();
        }
    }
    None
}

fn previous_sibling_or_ancestor_sibling(element: SyntaxElement) -> Option<SyntaxElement> {
    let token_sibling = non_trivia_sibling(element.clone(), Direction::Prev);
    if let Some(sibling) = token_sibling {
        Some(sibling)
    } else {
        // if not trying to find first ancestor which has such a sibling
        let range = element.text_range();
        let top_node = element.ancestors().take_while(|it| it.text_range() == range).last()?;
        let prev_sibling_node = top_node.ancestors().find(|it| {
            non_trivia_sibling(NodeOrToken::Node(it.to_owned()), Direction::Prev).is_some()
        })?;
        non_trivia_sibling(NodeOrToken::Node(prev_sibling_node), Direction::Prev)
    }
}
