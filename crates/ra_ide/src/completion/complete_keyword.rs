//! FIXME: write short doc here

use ra_syntax::{
    algo::non_trivia_sibling,
    ast::{self, LoopBodyOwner},
    match_ast, AstNode, Direction, NodeOrToken, SyntaxElement,
    SyntaxKind::*,
    SyntaxToken,
};

use crate::completion::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

pub(super) fn complete_use_tree_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    // complete keyword "crate" in use stmt
    let source_range = ctx.source_range();
    match (ctx.use_item_syntax.as_ref(), ctx.path_prefix.as_ref()) {
        (Some(_), None) => {
            CompletionItem::new(CompletionKind::Keyword, source_range, "crate")
                .kind(CompletionItemKind::Keyword)
                .insert_text("crate::")
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, source_range, "self")
                .kind(CompletionItemKind::Keyword)
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, source_range, "super")
                .kind(CompletionItemKind::Keyword)
                .insert_text("super::")
                .add_to(acc);
        }
        (Some(_), Some(_)) => {
            CompletionItem::new(CompletionKind::Keyword, source_range, "self")
                .kind(CompletionItemKind::Keyword)
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, source_range, "super")
                .kind(CompletionItemKind::Keyword)
                .insert_text("super::")
                .add_to(acc);
        }
        _ => {}
    }
}

fn keyword(ctx: &CompletionContext, kw: &str, snippet: &str) -> CompletionItem {
    let res = CompletionItem::new(CompletionKind::Keyword, ctx.source_range(), kw)
        .kind(CompletionItemKind::Keyword);

    match ctx.config.snippet_cap {
        Some(cap) => res.insert_snippet(cap, snippet),
        _ => res.insert_text(if snippet.contains('$') { kw } else { snippet }),
    }
    .build()
}

fn add_top_level_keywords(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some(token) = previous_non_triva_element(&ctx.token).and_then(|it| it.into_token()) {
        if token.kind() == UNSAFE_KW {
            acc.add(keyword(ctx, "impl", "impl $0 {}"));
            acc.add(keyword(ctx, "trait", "trait $0 {}"));
            acc.add(keyword(ctx, "fn", "fn $0() {}"));
            return;
        }
    }
    acc.add(keyword(ctx, "impl", "impl $0 {}"));
    acc.add(keyword(ctx, "enum", "enum $0 {}"));
    acc.add(keyword(ctx, "struct", "struct $0 {}"));
    acc.add(keyword(ctx, "trait", "trait $0 {}"));
    acc.add(keyword(ctx, "fn", "fn $0() {}"));
    acc.add(keyword(ctx, "unsafe", "unsafe "));
}

pub(super) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.is_new_item {
        add_top_level_keywords(acc, ctx);
        return;
    }
    if !ctx.is_trivial_path {
        return;
    }

    let fn_def = match &ctx.function_syntax {
        Some(it) => it,
        None => return,
    };
    acc.add(keyword(ctx, "if", "if $0 {}"));
    acc.add(keyword(ctx, "match", "match $0 {}"));
    acc.add(keyword(ctx, "while", "while $0 {}"));
    acc.add(keyword(ctx, "loop", "loop {$0}"));

    if ctx.after_if {
        acc.add(keyword(ctx, "else", "else {$0}"));
        acc.add(keyword(ctx, "else if", "else if $0 {}"));
    }
    if is_in_loop_body(&ctx.token) {
        if ctx.can_be_stmt {
            acc.add(keyword(ctx, "continue", "continue;"));
            acc.add(keyword(ctx, "break", "break;"));
        } else {
            acc.add(keyword(ctx, "continue", "continue"));
            acc.add(keyword(ctx, "break", "break"));
        }
    }
    acc.add_all(complete_return(ctx, &fn_def, ctx.can_be_stmt));
}

fn previous_non_triva_element(token: &SyntaxToken) -> Option<SyntaxElement> {
    // trying to get first non triva sibling if we have one
    let token_sibling = non_trivia_sibling(NodeOrToken::Token(token.to_owned()), Direction::Prev);
    let mut wrapped = if let Some(sibling) = token_sibling {
        sibling
    } else {
        // if not trying to find first ancestor which has such a sibling
        let node = token.parent();
        let range = node.text_range();
        let top_node = node.ancestors().take_while(|it| it.text_range() == range).last()?;
        let prev_sibling_node = top_node.ancestors().find(|it| {
            non_trivia_sibling(NodeOrToken::Node(it.to_owned()), Direction::Prev).is_some()
        })?;
        non_trivia_sibling(NodeOrToken::Node(prev_sibling_node), Direction::Prev)?
    };
    // traversing the tree down to get the last token or node, i.e. the closest one
    loop {
        if let Some(token) = wrapped.as_token() {
            return Some(NodeOrToken::Token(token.clone()));
        } else {
            let new = wrapped.as_node().and_then(|n| n.last_child_or_token());
            if new.is_some() {
                wrapped = new.unwrap().clone();
            } else {
                return Some(wrapped);
            }
        }
    }
}

fn is_in_loop_body(leaf: &SyntaxToken) -> bool {
    // FIXME move this to CompletionContext and make it handle macros
    for node in leaf.parent().ancestors() {
        if node.kind() == FN_DEF || node.kind() == LAMBDA_EXPR {
            break;
        }
        let loop_body = match_ast! {
            match node {
                ast::ForExpr(it) => it.loop_body(),
                ast::WhileExpr(it) => it.loop_body(),
                ast::LoopExpr(it) => it.loop_body(),
                _ => None,
            }
        };
        if let Some(body) = loop_body {
            if body.syntax().text_range().contains_range(leaf.text_range()) {
                return true;
            }
        }
    }
    false
}

fn complete_return(
    ctx: &CompletionContext,
    fn_def: &ast::FnDef,
    can_be_stmt: bool,
) -> Option<CompletionItem> {
    let snip = match (can_be_stmt, fn_def.ret_type().is_some()) {
        (true, true) => "return $0;",
        (true, false) => "return;",
        (false, true) => "return $0",
        (false, false) => "return",
    };
    Some(keyword(ctx, "return", snip))
}

#[cfg(test)]
mod tests {
    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_keyword_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Keyword)
    }

    #[test]
    fn completes_keywords_in_use_stmt() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                use <|>
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "crate",
                source_range: 21..21,
                delete: 21..21,
                insert: "crate::",
                kind: Keyword,
            },
            CompletionItem {
                label: "self",
                source_range: 21..21,
                delete: 21..21,
                insert: "self",
                kind: Keyword,
            },
            CompletionItem {
                label: "super",
                source_range: 21..21,
                delete: 21..21,
                insert: "super::",
                kind: Keyword,
            },
        ]
        "###
        );

        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                use a::<|>
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "self",
                source_range: 24..24,
                delete: 24..24,
                insert: "self",
                kind: Keyword,
            },
            CompletionItem {
                label: "super",
                source_range: 24..24,
                delete: 24..24,
                insert: "super::",
                kind: Keyword,
            },
        ]
        "###
        );

        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                use a::{b, <|>}
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "self",
                source_range: 28..28,
                delete: 28..28,
                insert: "self",
                kind: Keyword,
            },
            CompletionItem {
                label: "super",
                source_range: 28..28,
                delete: 28..28,
                insert: "super::",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_various_keywords_in_function() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() {
                    <|>
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 49..49,
                delete: 49..49,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 49..49,
                delete: 49..49,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 49..49,
                delete: 49..49,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 49..49,
                delete: 49..49,
                insert: "return;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 49..49,
                delete: 49..49,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_unsafe_context_in_item_position_with_non_empty_token() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                mod my_mod {
                    unsafe i<|>
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "fn",
                source_range: 57..58,
                delete: 57..58,
                insert: "fn $0() {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "impl",
                source_range: 57..58,
                delete: 57..58,
                insert: "impl $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "trait",
                source_range: 57..58,
                delete: 57..58,
                insert: "trait $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_unsafe_context_in_item_position_with_empty_token() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                mod my_mod {
                    unsafe <|>
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "fn",
                source_range: 57..57,
                delete: 57..57,
                insert: "fn $0() {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "impl",
                source_range: 57..57,
                delete: 57..57,
                insert: "impl $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "trait",
                source_range: 57..57,
                delete: 57..57,
                insert: "trait $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_keywords_in_item_position_with_empty_token() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                <|>
                ",
            ),
            @r###"
            [
                CompletionItem {
                    label: "enum",
                    source_range: 17..17,
                    delete: 17..17,
                    insert: "enum $0 {}",
                    kind: Keyword,
                },
                CompletionItem {
                    label: "fn",
                    source_range: 17..17,
                    delete: 17..17,
                    insert: "fn $0() {}",
                    kind: Keyword,
                },
                CompletionItem {
                    label: "impl",
                    source_range: 17..17,
                    delete: 17..17,
                    insert: "impl $0 {}",
                    kind: Keyword,
                },
                CompletionItem {
                    label: "struct",
                    source_range: 17..17,
                    delete: 17..17,
                    insert: "struct $0 {}",
                    kind: Keyword,
                },
                CompletionItem {
                    label: "trait",
                    source_range: 17..17,
                    delete: 17..17,
                    insert: "trait $0 {}",
                    kind: Keyword,
                },
                CompletionItem {
                    label: "unsafe",
                    source_range: 17..17,
                    delete: 17..17,
                    insert: "unsafe ",
                    kind: Keyword,
                },
            ]
        "###
        );
    }

    #[test]
    fn completes_else_after_if() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() {
                    if true {
                        ()
                    } <|>
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "else",
                source_range: 108..108,
                delete: 108..108,
                insert: "else {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "else if",
                source_range: 108..108,
                delete: 108..108,
                insert: "else if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "if",
                source_range: 108..108,
                delete: 108..108,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 108..108,
                delete: 108..108,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 108..108,
                delete: 108..108,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 108..108,
                delete: 108..108,
                insert: "return;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 108..108,
                delete: 108..108,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn test_completion_return_value() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() -> i32 {
                    <|>
                    92
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 56..56,
                delete: 56..56,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 56..56,
                delete: 56..56,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 56..56,
                delete: 56..56,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 56..56,
                delete: 56..56,
                insert: "return $0;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 56..56,
                delete: 56..56,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() {
                    <|>
                    92
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 49..49,
                delete: 49..49,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 49..49,
                delete: 49..49,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 49..49,
                delete: 49..49,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 49..49,
                delete: 49..49,
                insert: "return;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 49..49,
                delete: 49..49,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn dont_add_semi_after_return_if_not_a_statement() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() -> i32 {
                    match () {
                        () => <|>
                    }
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 97..97,
                delete: 97..97,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 97..97,
                delete: 97..97,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 97..97,
                delete: 97..97,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 97..97,
                delete: 97..97,
                insert: "return $0",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 97..97,
                delete: 97..97,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn last_return_in_block_has_semi() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() -> i32 {
                    if condition {
                        <|>
                    }
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 95..95,
                delete: 95..95,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 95..95,
                delete: 95..95,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 95..95,
                delete: 95..95,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 95..95,
                delete: 95..95,
                insert: "return $0;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 95..95,
                delete: 95..95,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() -> i32 {
                    if condition {
                        <|>
                    }
                    let x = 92;
                    x
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 95..95,
                delete: 95..95,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 95..95,
                delete: 95..95,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 95..95,
                delete: 95..95,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 95..95,
                delete: 95..95,
                insert: "return $0;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 95..95,
                delete: 95..95,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_break_and_continue_in_loops() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() -> i32 {
                    loop { <|> }
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "break",
                source_range: 63..63,
                delete: 63..63,
                insert: "break;",
                kind: Keyword,
            },
            CompletionItem {
                label: "continue",
                source_range: 63..63,
                delete: 63..63,
                insert: "continue;",
                kind: Keyword,
            },
            CompletionItem {
                label: "if",
                source_range: 63..63,
                delete: 63..63,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 63..63,
                delete: 63..63,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 63..63,
                delete: 63..63,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 63..63,
                delete: 63..63,
                insert: "return $0;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 63..63,
                delete: 63..63,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );

        // No completion: lambda isolates control flow
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn quux() -> i32 {
                    loop { || { <|> } }
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "if",
                source_range: 68..68,
                delete: 68..68,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 68..68,
                delete: 68..68,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 68..68,
                delete: 68..68,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 68..68,
                delete: 68..68,
                insert: "return $0;",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 68..68,
                delete: 68..68,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        );
    }

    #[test]
    fn no_semi_after_break_continue_in_expr() {
        assert_debug_snapshot!(
            do_keyword_completion(
                r"
                fn f() {
                    loop {
                        match () {
                            () => br<|>
                        }
                    }
                }
                ",
            ),
            @r###"
        [
            CompletionItem {
                label: "break",
                source_range: 122..124,
                delete: 122..124,
                insert: "break",
                kind: Keyword,
            },
            CompletionItem {
                label: "continue",
                source_range: 122..124,
                delete: 122..124,
                insert: "continue",
                kind: Keyword,
            },
            CompletionItem {
                label: "if",
                source_range: 122..124,
                delete: 122..124,
                insert: "if $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "loop",
                source_range: 122..124,
                delete: 122..124,
                insert: "loop {$0}",
                kind: Keyword,
            },
            CompletionItem {
                label: "match",
                source_range: 122..124,
                delete: 122..124,
                insert: "match $0 {}",
                kind: Keyword,
            },
            CompletionItem {
                label: "return",
                source_range: 122..124,
                delete: 122..124,
                insert: "return",
                kind: Keyword,
            },
            CompletionItem {
                label: "while",
                source_range: 122..124,
                delete: 122..124,
                insert: "while $0 {}",
                kind: Keyword,
            },
        ]
        "###
        )
    }
}
