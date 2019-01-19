use ra_syntax::{
    algo::visit::{visitor, Visitor},
    AstNode,
    ast::{self, LoopBodyOwner},
    SyntaxKind::*, SyntaxNode,
};

use crate::completion::{CompletionContext, CompletionItem, Completions, CompletionKind, CompletionItemKind};

pub(super) fn complete_use_tree_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    // complete keyword "crate" in use stmt
    match (ctx.use_item_syntax.as_ref(), ctx.path_prefix.as_ref()) {
        (Some(_), None) => {
            CompletionItem::new(CompletionKind::Keyword, ctx, "crate")
                .kind(CompletionItemKind::Keyword)
                .insert_text("crate::")
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, ctx, "self")
                .kind(CompletionItemKind::Keyword)
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, ctx, "super")
                .kind(CompletionItemKind::Keyword)
                .insert_text("super::")
                .add_to(acc);
        }
        (Some(_), Some(_)) => {
            CompletionItem::new(CompletionKind::Keyword, ctx, "self")
                .kind(CompletionItemKind::Keyword)
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, ctx, "super")
                .kind(CompletionItemKind::Keyword)
                .insert_text("super::")
                .add_to(acc);
        }
        _ => {}
    }
}

fn keyword(ctx: &CompletionContext, kw: &str, snippet: &str) -> CompletionItem {
    CompletionItem::new(CompletionKind::Keyword, ctx, kw)
        .kind(CompletionItemKind::Keyword)
        .snippet(snippet)
        .build()
}

pub(super) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_trivial_path {
        return;
    }

    let fn_def = match ctx.function_syntax {
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
    if is_in_loop_body(ctx.leaf) {
        if ctx.can_be_stmt {
            acc.add(keyword(ctx, "continue", "continue;"));
            acc.add(keyword(ctx, "break", "break;"));
        } else {
            acc.add(keyword(ctx, "continue", "continue"));
            acc.add(keyword(ctx, "break", "break"));
        }
    }
    acc.add_all(complete_return(ctx, fn_def, ctx.can_be_stmt));
}

fn is_in_loop_body(leaf: &SyntaxNode) -> bool {
    for node in leaf.ancestors() {
        if node.kind() == FN_DEF || node.kind() == LAMBDA_EXPR {
            break;
        }
        let loop_body = visitor()
            .visit::<ast::ForExpr, _>(LoopBodyOwner::loop_body)
            .visit::<ast::WhileExpr, _>(LoopBodyOwner::loop_body)
            .visit::<ast::LoopExpr, _>(LoopBodyOwner::loop_body)
            .accept(node);
        if let Some(Some(body)) = loop_body {
            if leaf.range().is_subrange(&body.syntax().range()) {
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
    use crate::completion::CompletionKind;
    use crate::completion::completion_item::check_completion;

    fn check_keyword_completion(name: &str, code: &str) {
        check_completion(name, code, CompletionKind::Keyword);
    }

    #[test]
    fn completes_keywords_in_use_stmt() {
        check_keyword_completion(
            "keywords_in_use_stmt1",
            r"
            use <|>
            ",
        );

        check_keyword_completion(
            "keywords_in_use_stmt2",
            r"
            use a::<|>
            ",
        );

        check_keyword_completion(
            "keywords_in_use_stmt3",
            r"
            use a::{b, <|>}
            ",
        );
    }

    #[test]
    fn completes_various_keywords_in_function() {
        check_keyword_completion(
            "keywords_in_function1",
            r"
            fn quux() {
                <|>
            }
            ",
        );
    }

    #[test]
    fn completes_else_after_if() {
        check_keyword_completion(
            "keywords_in_function2",
            r"
            fn quux() {
                if true {
                    ()
                } <|>
            }
            ",
        );
    }

    #[test]
    fn test_completion_return_value() {
        check_keyword_completion(
            "keywords_in_function3",
            r"
            fn quux() -> i32 {
                <|>
                92
            }
            ",
        );
        check_keyword_completion(
            "keywords_in_function4",
            r"
            fn quux() {
                <|>
                92
            }
            ",
        );
    }

    #[test]
    fn dont_add_semi_after_return_if_not_a_statement() {
        check_keyword_completion(
            "dont_add_semi_after_return_if_not_a_statement",
            r"
            fn quux() -> i32 {
                match () {
                    () => <|>
                }
            }
            ",
        );
    }

    #[test]
    fn last_return_in_block_has_semi() {
        check_keyword_completion(
            "last_return_in_block_has_semi1",
            r"
            fn quux() -> i32 {
                if condition {
                    <|>
                }
            }
            ",
        );
        check_keyword_completion(
            "last_return_in_block_has_semi2",
            r"
            fn quux() -> i32 {
                if condition {
                    <|>
                }
                let x = 92;
                x
            }
            ",
        );
    }

    #[test]
    fn completes_break_and_continue_in_loops() {
        check_keyword_completion(
            "completes_break_and_continue_in_loops1",
            r"
            fn quux() -> i32 {
                loop { <|> }
            }
            ",
        );

        // No completion: lambda isolates control flow
        check_keyword_completion(
            "completes_break_and_continue_in_loops2",
            r"
            fn quux() -> i32 {
                loop { || { <|> } }
            }
            ",
        );
    }

    #[test]
    fn no_semi_after_break_continue_in_expr() {
        check_keyword_completion(
            "no_semi_after_break_continue_in_expr",
            r"
            fn f() {
                loop {
                    match () {
                        () => br<|>
                    }
                }
            }
            ",
        )
    }
}
