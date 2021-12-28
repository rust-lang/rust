//! Completes keywords, except:
//! - `self`, `super` and `crate`, as these are considered part of path completions.
//! - `await`, as this is a postfix completion we handle this in the postfix completions.

use syntax::{SyntaxKind, T};

use crate::{
    context::{PathCompletionContext, PathKind},
    patterns::ImmediateLocation,
    CompletionContext, CompletionItem, CompletionItemKind, Completions,
};

pub(crate) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.token.kind() == SyntaxKind::COMMENT {
        cov_mark::hit!(no_keyword_completion_in_comments);
        return;
    }
    if matches!(ctx.completion_location, Some(ImmediateLocation::RecordExpr(_))) {
        cov_mark::hit!(no_keyword_completion_in_record_lit);
        return;
    }
    if ctx.fake_attribute_under_caret.is_some() {
        cov_mark::hit!(no_keyword_completion_in_attr_of_expr);
        return;
    }
    if ctx.is_non_trivial_path() {
        cov_mark::hit!(no_keyword_completion_in_non_trivial_path);
        return;
    }

    let mut add_keyword = |kw, snippet| add_keyword(acc, ctx, kw, snippet);

    let expects_assoc_item = ctx.expects_assoc_item();
    let has_block_expr_parent = ctx.has_block_expr_parent();
    let expects_item = ctx.expects_item();

    if let Some(PathKind::Vis { has_in_token }) = ctx.path_kind() {
        if !has_in_token {
            cov_mark::hit!(kw_completion_in);
            add_keyword("in", "in");
        }
        return;
    }
    if ctx.has_impl_or_trait_prev_sibling() {
        add_keyword("where", "where");
        if ctx.has_impl_prev_sibling() {
            add_keyword("for", "for");
        }
        return;
    }
    if ctx.previous_token_is(T![unsafe]) {
        if expects_item || expects_assoc_item || has_block_expr_parent {
            add_keyword("fn", "fn $1($2) {\n    $0\n}")
        }

        if expects_item || has_block_expr_parent {
            add_keyword("trait", "trait $1 {\n    $0\n}");
            add_keyword("impl", "impl $1 {\n    $0\n}");
        }

        return;
    }

    if !ctx.has_visibility_prev_sibling()
        && (expects_item || ctx.expects_non_trait_assoc_item() || ctx.expect_field())
    {
        add_keyword("pub(crate)", "pub(crate)");
        add_keyword("pub(super)", "pub(super)");
        add_keyword("pub", "pub");
    }

    if expects_item || expects_assoc_item || has_block_expr_parent {
        add_keyword("unsafe", "unsafe");
        add_keyword("fn", "fn $1($2) {\n    $0\n}");
        add_keyword("const", "const $0");
        add_keyword("type", "type $0");
    }

    if expects_item || has_block_expr_parent {
        if !ctx.has_visibility_prev_sibling() {
            add_keyword("impl", "impl $1 {\n    $0\n}");
            add_keyword("extern", "extern $0");
        }
        add_keyword("use", "use $0");
        add_keyword("trait", "trait $1 {\n    $0\n}");
        add_keyword("static", "static $0");
        add_keyword("mod", "mod $0");
    }

    if expects_item {
        add_keyword("enum", "enum $1 {\n    $0\n}");
        add_keyword("struct", "struct $0");
        add_keyword("union", "union $1 {\n    $0\n}");
    }

    if ctx.expects_type() {
        return;
    }

    if ctx.expects_expression() {
        if !has_block_expr_parent {
            add_keyword("unsafe", "unsafe {\n    $0\n}");
        }
        add_keyword("match", "match $1 {\n    $0\n}");
        add_keyword("while", "while $1 {\n    $0\n}");
        add_keyword("while let", "while let $1 = $2 {\n    $0\n}");
        add_keyword("loop", "loop {\n    $0\n}");
        add_keyword("if", "if $1 {\n    $0\n}");
        add_keyword("if let", "if let $1 = $2 {\n    $0\n}");
        add_keyword("for", "for $1 in $2 {\n    $0\n}");
        add_keyword("true", "true");
        add_keyword("false", "false");
    }

    if ctx.previous_token_is(T![if]) || ctx.previous_token_is(T![while]) || has_block_expr_parent {
        add_keyword("let", "let");
    }

    if ctx.after_if() {
        add_keyword("else", "else {\n    $0\n}");
        add_keyword("else if", "else if $1 {\n    $0\n}");
    }

    if ctx.expects_ident_pat_or_ref_expr() {
        add_keyword("mut", "mut ");
    }

    let (can_be_stmt, in_loop_body) = match ctx.path_context {
        Some(PathCompletionContext {
            is_trivial_path: true, can_be_stmt, in_loop_body, ..
        }) => (can_be_stmt, in_loop_body),
        _ => return,
    };

    if in_loop_body {
        if can_be_stmt {
            add_keyword("continue", "continue;");
            add_keyword("break", "break;");
        } else {
            add_keyword("continue", "continue");
            add_keyword("break", "break");
        }
    }

    let fn_def = match &ctx.function_def {
        Some(it) => it,
        None => return,
    };

    add_keyword(
        "return",
        match (can_be_stmt, fn_def.ret_type().is_some()) {
            (true, true) => "return $0;",
            (true, false) => "return;",
            (false, true) => "return $0",
            (false, false) => "return",
        },
    )
}

fn add_keyword(acc: &mut Completions, ctx: &CompletionContext, kw: &str, snippet: &str) {
    let mut item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), kw);

    match ctx.config.snippet_cap {
        Some(cap) => {
            if snippet.ends_with('}') && ctx.incomplete_let {
                cov_mark::hit!(let_semi);
                item.insert_snippet(cap, format!("{};", snippet));
            } else {
                item.insert_snippet(cap, snippet);
            }
        }
        None => {
            item.insert_text(if snippet.contains('$') { kw } else { snippet });
        }
    };
    item.add_to(acc);
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::{check_edit, completion_list};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture);
        expect.assert_eq(&actual)
    }

    #[test]
    fn test_else_edit_after_if() {
        check_edit(
            "else",
            r#"fn quux() { if true { () } $0 }"#,
            r#"fn quux() { if true { () } else {
    $0
} }"#,
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_block_expr() {
        check(
            r"fn my_fn() { unsafe $0 }",
            expect![[r#"
                kw fn
                kw trait
                kw impl
                sn pd
                sn ppd
            "#]],
        );
    }

    #[test]
    fn test_completion_await_impls_future() {
        check(
            r#"
//- minicore: future
use core::future::*;
struct A {}
impl Future for A {}
fn foo(a: A) { a.$0 }
"#,
            expect![[r#"
                kw await expr.await
                sn ref   &expr
                sn refm  &mut expr
                sn match match expr {}
                sn box   Box::new(expr)
                sn dbg   dbg!(expr)
                sn dbgr  dbg!(&expr)
                sn call  function(expr)
                sn let   let
                sn letm  let mut
            "#]],
        );

        check(
            r#"
//- minicore: future
use std::future::*;
fn foo() {
    let a = async {};
    a.$0
}
"#,
            expect![[r#"
                kw await expr.await
                sn ref   &expr
                sn refm  &mut expr
                sn match match expr {}
                sn box   Box::new(expr)
                sn dbg   dbg!(expr)
                sn dbgr  dbg!(&expr)
                sn call  function(expr)
                sn let   let
                sn letm  let mut
            "#]],
        )
    }

    #[test]
    fn let_semi() {
        cov_mark::check!(let_semi);
        check_edit(
            "match",
            r#"
fn main() { let x = $0 }
"#,
            r#"
fn main() { let x = match $1 {
    $0
}; }
"#,
        );

        check_edit(
            "if",
            r#"
fn main() {
    let x = $0
    let y = 92;
}
"#,
            r#"
fn main() {
    let x = if $1 {
    $0
};
    let y = 92;
}
"#,
        );

        check_edit(
            "loop",
            r#"
fn main() {
    let x = $0
    bar();
}
"#,
            r#"
fn main() {
    let x = loop {
    $0
};
    bar();
}
"#,
        );
    }
}
