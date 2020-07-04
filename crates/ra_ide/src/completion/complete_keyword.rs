//! FIXME: write short doc here

use ra_syntax::{ast, SyntaxKind};
use test_utils::mark;

use crate::completion::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

pub(super) fn complete_use_tree_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    // complete keyword "crate" in use stmt
    let source_range = ctx.source_range();
    match (ctx.use_item_syntax.as_ref(), ctx.path_prefix.as_ref()) {
        (Some(_), None) => {
            CompletionItem::new(CompletionKind::Keyword, source_range, "crate::")
                .kind(CompletionItemKind::Keyword)
                .insert_text("crate::")
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, source_range, "self")
                .kind(CompletionItemKind::Keyword)
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, source_range, "super::")
                .kind(CompletionItemKind::Keyword)
                .insert_text("super::")
                .add_to(acc);
        }
        (Some(_), Some(_)) => {
            CompletionItem::new(CompletionKind::Keyword, source_range, "self")
                .kind(CompletionItemKind::Keyword)
                .add_to(acc);
            CompletionItem::new(CompletionKind::Keyword, source_range, "super::")
                .kind(CompletionItemKind::Keyword)
                .insert_text("super::")
                .add_to(acc);
        }
        _ => {}
    }

    // Suggest .await syntax for types that implement Future trait
    if let Some(receiver) = &ctx.dot_receiver {
        if let Some(ty) = ctx.sema.type_of_expr(receiver) {
            if ty.impls_future(ctx.db) {
                CompletionItem::new(CompletionKind::Keyword, ctx.source_range(), "await")
                    .kind(CompletionItemKind::Keyword)
                    .detail("expr.await")
                    .insert_text("await")
                    .add_to(acc);
            }
        };
    }
}

pub(super) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.token.kind() == SyntaxKind::COMMENT {
        mark::hit!(no_keyword_completion_in_comments);
        return;
    }

    let has_trait_or_impl_parent = ctx.has_impl_parent || ctx.has_trait_parent;
    if ctx.trait_as_prev_sibling || ctx.impl_as_prev_sibling {
        add_keyword(ctx, acc, "where", "where ");
        return;
    }
    if ctx.unsafe_is_prev {
        if ctx.has_item_list_or_source_file_parent || ctx.block_expr_parent {
            add_keyword(ctx, acc, "fn", "fn $0() {}")
        }

        if (ctx.has_item_list_or_source_file_parent && !has_trait_or_impl_parent)
            || ctx.block_expr_parent
        {
            add_keyword(ctx, acc, "trait", "trait $0 {}");
            add_keyword(ctx, acc, "impl", "impl $0 {}");
        }

        return;
    }
    if ctx.has_item_list_or_source_file_parent || ctx.block_expr_parent {
        add_keyword(ctx, acc, "fn", "fn $0() {}");
    }
    if (ctx.has_item_list_or_source_file_parent && !has_trait_or_impl_parent)
        || ctx.block_expr_parent
    {
        add_keyword(ctx, acc, "use", "use ");
        add_keyword(ctx, acc, "impl", "impl $0 {}");
        add_keyword(ctx, acc, "trait", "trait $0 {}");
    }

    if ctx.has_item_list_or_source_file_parent && !has_trait_or_impl_parent {
        add_keyword(ctx, acc, "enum", "enum $0 {}");
        add_keyword(ctx, acc, "struct", "struct $0 {}");
        add_keyword(ctx, acc, "union", "union $0 {}");
    }

    if ctx.block_expr_parent || ctx.is_match_arm {
        add_keyword(ctx, acc, "match", "match $0 {}");
        add_keyword(ctx, acc, "loop", "loop {$0}");
    }
    if ctx.block_expr_parent {
        add_keyword(ctx, acc, "while", "while $0 {}");
    }
    if ctx.if_is_prev || ctx.block_expr_parent {
        add_keyword(ctx, acc, "let", "let ");
    }
    if ctx.if_is_prev || ctx.block_expr_parent || ctx.is_match_arm {
        add_keyword(ctx, acc, "if", "if ");
        add_keyword(ctx, acc, "if let", "if let ");
    }
    if ctx.after_if {
        add_keyword(ctx, acc, "else", "else {$0}");
        add_keyword(ctx, acc, "else if", "else if $0 {}");
    }
    if (ctx.has_item_list_or_source_file_parent && !has_trait_or_impl_parent)
        || ctx.block_expr_parent
    {
        add_keyword(ctx, acc, "mod", "mod $0 {}");
    }
    if ctx.bind_pat_parent || ctx.ref_pat_parent {
        add_keyword(ctx, acc, "mut", "mut ");
    }
    if ctx.has_item_list_or_source_file_parent || ctx.block_expr_parent {
        add_keyword(ctx, acc, "const", "const ");
        add_keyword(ctx, acc, "type", "type ");
    }
    if (ctx.has_item_list_or_source_file_parent && !has_trait_or_impl_parent)
        || ctx.block_expr_parent
    {
        add_keyword(ctx, acc, "static", "static ");
    };
    if (ctx.has_item_list_or_source_file_parent && !has_trait_or_impl_parent)
        || ctx.block_expr_parent
    {
        add_keyword(ctx, acc, "extern", "extern ");
    }
    if ctx.has_item_list_or_source_file_parent || ctx.block_expr_parent || ctx.is_match_arm {
        add_keyword(ctx, acc, "unsafe", "unsafe ");
    }
    if ctx.in_loop_body {
        if ctx.can_be_stmt {
            add_keyword(ctx, acc, "continue", "continue;");
            add_keyword(ctx, acc, "break", "break;");
        } else {
            add_keyword(ctx, acc, "continue", "continue");
            add_keyword(ctx, acc, "break", "break");
        }
    }
    if ctx.has_item_list_or_source_file_parent && !ctx.has_trait_parent {
        add_keyword(ctx, acc, "pub", "pub ")
    }

    if !ctx.is_trivial_path {
        return;
    }
    let fn_def = match &ctx.function_syntax {
        Some(it) => it,
        None => return,
    };
    acc.add_all(complete_return(ctx, &fn_def, ctx.can_be_stmt));
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

fn add_keyword(ctx: &CompletionContext, acc: &mut Completions, kw: &str, snippet: &str) {
    acc.add(keyword(ctx, kw, snippet));
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
    use expect::{expect, Expect};

    use crate::completion::{
        test_utils::{check_edit, completion_list},
        CompletionKind,
    };
    use test_utils::mark;

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Keyword);
        expect.assert_eq(&actual)
    }

    #[test]
    fn test_keywords_in_use_stmt() {
        check(
            r"use <|>",
            expect![[r#"
                kw crate::
                kw self
                kw super::
            "#]],
        );

        check(
            r"use a::<|>",
            expect![[r#"
                kw self
                kw super::
            "#]],
        );

        check(
            r"use a::{b, <|>}",
            expect![[r#"
                kw self
                kw super::
            "#]],
        );
    }

    #[test]
    fn test_keywords_at_source_file_level() {
        check(
            r"m<|>",
            expect![[r#"
                kw const
                kw enum
                kw extern
                kw fn
                kw impl
                kw mod
                kw pub
                kw static
                kw struct
                kw trait
                kw type
                kw union
                kw unsafe
                kw use
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_function() {
        check(
            r"fn quux() { <|> }",
            expect![[r#"
                kw const
                kw extern
                kw fn
                kw if
                kw if let
                kw impl
                kw let
                kw loop
                kw match
                kw mod
                kw return
                kw static
                kw trait
                kw type
                kw unsafe
                kw use
                kw while
            "#]],
        );
    }

    #[test]
    fn test_keywords_inside_block() {
        check(
            r"fn quux() { if true { <|> } }",
            expect![[r#"
                kw const
                kw extern
                kw fn
                kw if
                kw if let
                kw impl
                kw let
                kw loop
                kw match
                kw mod
                kw return
                kw static
                kw trait
                kw type
                kw unsafe
                kw use
                kw while
            "#]],
        );
    }

    #[test]
    fn test_keywords_after_if() {
        check(
            r#"fn quux() { if true { () } <|> }"#,
            expect![[r#"
                kw const
                kw else
                kw else if
                kw extern
                kw fn
                kw if
                kw if let
                kw impl
                kw let
                kw loop
                kw match
                kw mod
                kw return
                kw static
                kw trait
                kw type
                kw unsafe
                kw use
                kw while
            "#]],
        );
        check_edit(
            "else",
            r#"fn quux() { if true { () } <|> }"#,
            r#"fn quux() { if true { () } else {$0} }"#,
        );
    }

    #[test]
    fn test_keywords_in_match_arm() {
        check(
            r#"
fn quux() -> i32 {
    match () {
        () => <|>
    }
}
"#,
            expect![[r#"
                kw if
                kw if let
                kw loop
                kw match
                kw return
                kw unsafe
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_trait_def() {
        check(
            r"trait My { <|> }",
            expect![[r#"
                kw const
                kw fn
                kw type
                kw unsafe
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_impl_def() {
        check(
            r"impl My { <|> }",
            expect![[r#"
                kw const
                kw fn
                kw pub
                kw type
                kw unsafe
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_loop() {
        check(
            r"fn my() { loop { <|> } }",
            expect![[r#"
                kw break
                kw const
                kw continue
                kw extern
                kw fn
                kw if
                kw if let
                kw impl
                kw let
                kw loop
                kw match
                kw mod
                kw return
                kw static
                kw trait
                kw type
                kw unsafe
                kw use
                kw while
            "#]],
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_item_list() {
        check(
            r"unsafe <|>",
            expect![[r#"
                kw fn
                kw impl
                kw trait
            "#]],
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_block_expr() {
        check(
            r"fn my_fn() { unsafe <|> }",
            expect![[r#"
                kw fn
                kw impl
                kw trait
            "#]],
        );
    }

    #[test]
    fn test_mut_in_ref_and_in_fn_parameters_list() {
        check(
            r"fn my_fn(&<|>) {}",
            expect![[r#"
                kw mut
            "#]],
        );
        check(
            r"fn my_fn(<|>) {}",
            expect![[r#"
                kw mut
            "#]],
        );
        check(
            r"fn my_fn() { let &<|> }",
            expect![[r#"
                kw mut
            "#]],
        );
    }

    #[test]
    fn test_where_keyword() {
        check(
            r"trait A <|>",
            expect![[r#"
                kw where
            "#]],
        );
        check(
            r"impl A <|>",
            expect![[r#"
                kw where
            "#]],
        );
    }

    #[test]
    fn no_keyword_completion_in_comments() {
        mark::check!(no_keyword_completion_in_comments);
        check(
            r#"
fn test() {
    let x = 2; // A comment<|>
}
"#,
            expect![[""]],
        );
        check(
            r#"
/*
Some multi-line comment<|>
*/
"#,
            expect![[""]],
        );
        check(
            r#"
/// Some doc comment
/// let test<|> = 1
"#,
            expect![[""]],
        );
    }

    #[test]
    fn test_completion_await_impls_future() {
        check(
            r#"
//- /main.rs
use std::future::*;
struct A {}
impl Future for A {}
fn foo(a: A) { a.<|> }

//- /std/lib.rs
pub mod future {
    #[lang = "future_trait"]
    pub trait Future {}
}
"#,
            expect![[r#"
                kw await expr.await
            "#]],
        )
    }
}
