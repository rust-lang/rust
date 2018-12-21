use ra_syntax::{
    algo::visit::{visitor, Visitor},
    AstNode,
    ast::{self, LoopBodyOwner},
    SyntaxKind::*, SyntaxNodeRef,
};

use crate::completion::{CompletionContext, CompletionItem, Completions, CompletionKind};

fn keyword(kw: &str, snippet: &str) -> CompletionItem {
    CompletionItem::new(CompletionKind::Keyword, kw)
        .snippet(snippet)
        .build()
}

pub(super) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_trivial_path {
        return;
    }
    let fn_def = match ctx.enclosing_fn {
        Some(it) => it,
        None => return,
    };
    acc.add(keyword("if", "if $0 {}"));
    acc.add(keyword("match", "match $0 {}"));
    acc.add(keyword("while", "while $0 {}"));
    acc.add(keyword("loop", "loop {$0}"));

    if ctx.after_if {
        acc.add(keyword("else", "else {$0}"));
        acc.add(keyword("else if", "else if $0 {}"));
    }
    if is_in_loop_body(ctx.leaf) {
        acc.add(keyword("continue", "continue"));
        acc.add(keyword("break", "break"));
    }
    acc.add_all(complete_return(fn_def, ctx.is_stmt));
}

fn is_in_loop_body(leaf: SyntaxNodeRef) -> bool {
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

fn complete_return(fn_def: ast::FnDef, is_stmt: bool) -> Option<CompletionItem> {
    let snip = match (is_stmt, fn_def.ret_type().is_some()) {
        (true, true) => "return $0;",
        (true, false) => "return;",
        (false, true) => "return $0",
        (false, false) => "return",
    };
    Some(keyword("return", snip))
}

#[cfg(test)]
mod tests {
    use crate::completion::{CompletionKind, check_completion};
    fn check_keyword_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Keyword);
    }

    #[test]
    fn test_completion_kewords() {
        check_keyword_completion(
            r"
            fn quux() {
                <|>
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return"
            "#,
        );
    }

    #[test]
    fn test_completion_else() {
        check_keyword_completion(
            r"
            fn quux() {
                if true {
                    ()
                } <|>
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            else "else {$0}"
            else if "else if $0 {}"
            return "return"
            "#,
        );
    }

    #[test]
    fn test_completion_return_value() {
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                <|>
                92
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return $0;"
            "#,
        );
        check_keyword_completion(
            r"
            fn quux() {
                <|>
                92
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return;"
            "#,
        );
    }

    #[test]
    fn test_completion_return_no_stmt() {
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                match () {
                    () => <|>
                }
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return $0"
            "#,
        );
    }

    #[test]
    fn test_continue_break_completion() {
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                loop { <|> }
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            continue "continue"
            break "break"
            return "return $0"
            "#,
        );
        check_keyword_completion(
            r"
            fn quux() -> i32 {
                loop { || { <|> } }
            }
            ",
            r#"
            if "if $0 {}"
            match "match $0 {}"
            while "while $0 {}"
            loop "loop {$0}"
            return "return $0"
            "#,
        );
    }
}
