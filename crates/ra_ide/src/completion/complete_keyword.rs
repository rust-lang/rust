//! FIXME: write short doc here

use ra_syntax::ast;

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

fn add_keyword(
    ctx: &CompletionContext,
    acc: &mut Completions,
    kw: &str,
    snippet: &str,
    should_add: bool,
) {
    if should_add {
        acc.add(keyword(ctx, kw, snippet));
    }
}

pub(super) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    add_keyword(ctx, acc, "fn", "fn $0() {}", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "type", "type ", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "use", "fn $0() {}", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "impl", "impl $0 {}", ctx.is_new_item);
    add_keyword(ctx, acc, "trait", "impl $0 {}", ctx.is_new_item);
    add_keyword(ctx, acc, "enum", "enum $0 {}", ctx.is_new_item && !ctx.unsafe_is_prev);
    add_keyword(ctx, acc, "struct", "struct $0 {}", ctx.is_new_item && !ctx.unsafe_is_prev);
    add_keyword(ctx, acc, "union", "union $0 {}", ctx.is_new_item && !ctx.unsafe_is_prev);
    add_keyword(ctx, acc, "match", "match $0 {}", ctx.block_expr_parent);
    add_keyword(ctx, acc, "loop", "loop {$0}", ctx.block_expr_parent);
    add_keyword(ctx, acc, "while", "while $0 {}", ctx.block_expr_parent);
    add_keyword(ctx, acc, "let", "let ", ctx.if_is_prev || ctx.block_expr_parent);
    add_keyword(ctx, acc, "else", "else {$0}", ctx.after_if);
    add_keyword(ctx, acc, "else if", "else if $0 {}", ctx.after_if);
    add_keyword(ctx, acc, "mod", "mod $0 {}", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "mut", "mut ", ctx.bind_pat_parent || ctx.ref_pat_parent);
    add_keyword(ctx, acc, "const", "const ", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "type", "type ", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "static", "static ", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "extern", "extern ", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "unsafe", "unsafe ", ctx.is_new_item || ctx.block_expr_parent);
    add_keyword(ctx, acc, "continue", "continue;", ctx.in_loop_body && ctx.can_be_stmt);
    add_keyword(ctx, acc, "break", "break;", ctx.in_loop_body && ctx.can_be_stmt);
    add_keyword(ctx, acc, "continue", "continue", ctx.in_loop_body && !ctx.can_be_stmt);
    add_keyword(ctx, acc, "break", "break", ctx.in_loop_body && !ctx.can_be_stmt);
    add_keyword(ctx, acc, "pub", "pub ", ctx.is_new_item && !ctx.inside_trait);
    add_keyword(ctx, acc, "where", "where ", ctx.trait_as_prev_sibling || ctx.impl_as_prev_sibling);

    let fn_def = match &ctx.function_syntax {
        Some(it) => it,
        None => return,
    };
    acc.add_all(complete_return(ctx, &fn_def, ctx.can_be_stmt));
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
    use crate::{
        completion::{
            test_utils::{do_completion, do_completion_with_position},
            CompletionItem, CompletionKind,
        },
        CompletionItemKind,
    };
    use insta::{assert_snapshot, assert_debug_snapshot};
    use rustc_hash::FxHashSet;

    fn do_keyword_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Keyword)
    }

    fn assert_completion_keyword(code: &str, keywords: &[(&str, &str)]) {
        let (position, completion_items) =
            do_completion_with_position(code, CompletionKind::Keyword);
        let mut expected_keywords = FxHashSet::<(String, String)>::default();
        for (key, value) in keywords {
            expected_keywords.insert(((*key).to_string(), (*value).to_string()));
        }
        let mut returned_keywords = FxHashSet::<(String, String)>::default();
        
        for item in completion_items {
            assert!(item.text_edit().len() == 1);
            assert!(item.kind() == Some(CompletionItemKind::Keyword));
            let atom = item.text_edit().iter().next().unwrap().clone();
            assert!(atom.delete.start() == position.offset);
            assert!(atom.delete.end() == position.offset);
            let pair = (item.label().to_string(), atom.insert);
            returned_keywords.insert(pair);
        }
        let assert_failed_message = format!("Expected keywords: {:#?}\nReceived keywords: {:#?}", expected_keywords, returned_keywords);
        debug_assert!(returned_keywords == expected_keywords, assert_failed_message);
    }

    #[test]
    fn completes_keywords_in_use_stmt_new_approach() {
        assert_completion_keyword(
            r"
        use <|>
        ",
            &[("crate", "crate::"), ("self", "self"), ("super", "super::")],
        );
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
