//! Completes keywords.

use syntax::SyntaxKind;
use test_utils::mark;

use crate::{CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions};

pub(crate) fn complete_use_tree_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    // complete keyword "crate" in use stmt
    let source_range = ctx.source_range();

    if ctx.use_item_syntax.is_some() {
        if ctx.path_qual.is_none() {
            CompletionItem::new(CompletionKind::Keyword, source_range, "crate::")
                .kind(CompletionItemKind::Keyword)
                .insert_text("crate::")
                .add_to(acc);
        }
        CompletionItem::new(CompletionKind::Keyword, source_range, "self")
            .kind(CompletionItemKind::Keyword)
            .add_to(acc);
        CompletionItem::new(CompletionKind::Keyword, source_range, "super::")
            .kind(CompletionItemKind::Keyword)
            .insert_text("super::")
            .add_to(acc);
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

pub(crate) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.token.kind() == SyntaxKind::COMMENT {
        mark::hit!(no_keyword_completion_in_comments);
        return;
    }
    if ctx.record_lit_syntax.is_some() {
        mark::hit!(no_keyword_completion_in_record_lit);
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

        if (ctx.has_item_list_or_source_file_parent) || ctx.block_expr_parent {
            add_keyword(ctx, acc, "trait", "trait $0 {}");
            add_keyword(ctx, acc, "impl", "impl $0 {}");
        }

        return;
    }
    if ctx.has_item_list_or_source_file_parent || has_trait_or_impl_parent || ctx.block_expr_parent
    {
        add_keyword(ctx, acc, "fn", "fn $0() {}");
    }
    if (ctx.has_item_list_or_source_file_parent) || ctx.block_expr_parent {
        add_keyword(ctx, acc, "use", "use ");
        add_keyword(ctx, acc, "impl", "impl $0 {}");
        add_keyword(ctx, acc, "trait", "trait $0 {}");
    }

    if ctx.has_item_list_or_source_file_parent {
        add_keyword(ctx, acc, "enum", "enum $0 {}");
        add_keyword(ctx, acc, "struct", "struct $0");
        add_keyword(ctx, acc, "union", "union $0 {}");
    }

    if ctx.is_expr {
        add_keyword(ctx, acc, "match", "match $0 {}");
        add_keyword(ctx, acc, "while", "while $0 {}");
        add_keyword(ctx, acc, "loop", "loop {$0}");
        add_keyword(ctx, acc, "if", "if $0 {}");
        add_keyword(ctx, acc, "if let", "if let $1 = $0 {}");
    }

    if ctx.if_is_prev || ctx.block_expr_parent {
        add_keyword(ctx, acc, "let", "let ");
    }

    if ctx.after_if {
        add_keyword(ctx, acc, "else", "else {$0}");
        add_keyword(ctx, acc, "else if", "else if $0 {}");
    }
    if (ctx.has_item_list_or_source_file_parent) || ctx.block_expr_parent {
        add_keyword(ctx, acc, "mod", "mod $0");
    }
    if ctx.bind_pat_parent || ctx.ref_pat_parent {
        add_keyword(ctx, acc, "mut", "mut ");
    }
    if ctx.has_item_list_or_source_file_parent || has_trait_or_impl_parent || ctx.block_expr_parent
    {
        add_keyword(ctx, acc, "const", "const ");
        add_keyword(ctx, acc, "type", "type ");
    }
    if (ctx.has_item_list_or_source_file_parent) || ctx.block_expr_parent {
        add_keyword(ctx, acc, "static", "static ");
    };
    if (ctx.has_item_list_or_source_file_parent) || ctx.block_expr_parent {
        add_keyword(ctx, acc, "extern", "extern ");
    }
    if ctx.has_item_list_or_source_file_parent
        || has_trait_or_impl_parent
        || ctx.block_expr_parent
        || ctx.is_match_arm
    {
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
    if ctx.has_item_list_or_source_file_parent || ctx.has_impl_parent | ctx.has_field_list_parent {
        add_keyword(ctx, acc, "pub(crate)", "pub(crate) ");
        add_keyword(ctx, acc, "pub", "pub ");
    }

    if !ctx.is_trivial_path {
        return;
    }
    let fn_def = match &ctx.function_syntax {
        Some(it) => it,
        None => return,
    };

    add_keyword(
        ctx,
        acc,
        "return",
        match (ctx.can_be_stmt, fn_def.ret_type().is_some()) {
            (true, true) => "return $0;",
            (true, false) => "return;",
            (false, true) => "return $0",
            (false, false) => "return",
        },
    )
}

fn add_keyword(ctx: &CompletionContext, acc: &mut Completions, kw: &str, snippet: &str) {
    let builder = CompletionItem::new(CompletionKind::Keyword, ctx.source_range(), kw)
        .kind(CompletionItemKind::Keyword);
    let builder = match ctx.config.snippet_cap {
        Some(cap) => {
            let tmp;
            let snippet = if snippet.ends_with('}') && ctx.incomplete_let {
                mark::hit!(let_semi);
                tmp = format!("{};", snippet);
                &tmp
            } else {
                snippet
            };
            builder.insert_snippet(cap, snippet)
        }
        None => builder.insert_text(if snippet.contains('$') { kw } else { snippet }),
    };
    acc.add(builder.build());
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use test_utils::mark;

    use crate::{
        test_utils::{check_edit, completion_list},
        CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Keyword);
        expect.assert_eq(&actual)
    }

    #[test]
    fn test_keywords_in_use_stmt() {
        check(
            r"use $0",
            expect![[r#"
                kw crate::
                kw self
                kw super::
            "#]],
        );

        check(
            r"use a::$0",
            expect![[r#"
                kw self
                kw super::
            "#]],
        );

        check(
            r"use a::{b, $0}",
            expect![[r#"
                kw self
                kw super::
            "#]],
        );
    }

    #[test]
    fn test_keywords_at_source_file_level() {
        check(
            r"m$0",
            expect![[r#"
                kw fn
                kw use
                kw impl
                kw trait
                kw enum
                kw struct
                kw union
                kw mod
                kw const
                kw type
                kw static
                kw extern
                kw unsafe
                kw pub(crate)
                kw pub
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_function() {
        check(
            r"fn quux() { $0 }",
            expect![[r#"
                kw fn
                kw use
                kw impl
                kw trait
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw let
                kw mod
                kw const
                kw type
                kw static
                kw extern
                kw unsafe
                kw return
            "#]],
        );
    }

    #[test]
    fn test_keywords_inside_block() {
        check(
            r"fn quux() { if true { $0 } }",
            expect![[r#"
                kw fn
                kw use
                kw impl
                kw trait
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw let
                kw mod
                kw const
                kw type
                kw static
                kw extern
                kw unsafe
                kw return
            "#]],
        );
    }

    #[test]
    fn test_keywords_after_if() {
        check(
            r#"fn quux() { if true { () } $0 }"#,
            expect![[r#"
                kw fn
                kw use
                kw impl
                kw trait
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw let
                kw else
                kw else if
                kw mod
                kw const
                kw type
                kw static
                kw extern
                kw unsafe
                kw return
            "#]],
        );
        check_edit(
            "else",
            r#"fn quux() { if true { () } $0 }"#,
            r#"fn quux() { if true { () } else {$0} }"#,
        );
    }

    #[test]
    fn test_keywords_in_match_arm() {
        check(
            r#"
fn quux() -> i32 {
    match () { () => $0 }
}
"#,
            expect![[r#"
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw unsafe
                kw return
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_trait_def() {
        check(
            r"trait My { $0 }",
            expect![[r#"
                kw fn
                kw const
                kw type
                kw unsafe
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_impl_def() {
        check(
            r"impl My { $0 }",
            expect![[r#"
                kw fn
                kw const
                kw type
                kw unsafe
                kw pub(crate)
                kw pub
            "#]],
        );
    }

    #[test]
    fn test_keywords_in_loop() {
        check(
            r"fn my() { loop { $0 } }",
            expect![[r#"
                kw fn
                kw use
                kw impl
                kw trait
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw let
                kw mod
                kw const
                kw type
                kw static
                kw extern
                kw unsafe
                kw continue
                kw break
                kw return
            "#]],
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_item_list() {
        check(
            r"unsafe $0",
            expect![[r#"
                kw fn
                kw trait
                kw impl
            "#]],
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
            "#]],
        );
    }

    #[test]
    fn test_mut_in_ref_and_in_fn_parameters_list() {
        check(
            r"fn my_fn(&$0) {}",
            expect![[r#"
                kw mut
            "#]],
        );
        check(
            r"fn my_fn($0) {}",
            expect![[r#"
                kw mut
            "#]],
        );
        check(
            r"fn my_fn() { let &$0 }",
            expect![[r#"
                kw mut
            "#]],
        );
    }

    #[test]
    fn test_where_keyword() {
        check(
            r"trait A $0",
            expect![[r#"
                kw where
            "#]],
        );
        check(
            r"impl A $0",
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
    let x = 2; // A comment$0
}
"#,
            expect![[""]],
        );
        check(
            r#"
/*
Some multi-line comment$0
*/
"#,
            expect![[""]],
        );
        check(
            r#"
/// Some doc comment
/// let test$0 = 1
"#,
            expect![[""]],
        );
    }

    #[test]
    fn test_completion_await_impls_future() {
        check(
            r#"
//- /main.rs crate:main deps:std
use std::future::*;
struct A {}
impl Future for A {}
fn foo(a: A) { a.$0 }

//- /std/lib.rs crate:std
pub mod future {
    #[lang = "future_trait"]
    pub trait Future {}
}
"#,
            expect![[r#"
                kw await expr.await
            "#]],
        );

        check(
            r#"
//- /main.rs crate:main deps:std
use std::future::*;
fn foo() {
    let a = async {};
    a.$0
}

//- /std/lib.rs crate:std
pub mod future {
    #[lang = "future_trait"]
    pub trait Future {
        type Output;
    }
}
"#,
            expect![[r#"
                kw await expr.await
            "#]],
        )
    }

    #[test]
    fn after_let() {
        check(
            r#"fn main() { let _ = $0 }"#,
            expect![[r#"
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw return
            "#]],
        )
    }

    #[test]
    fn before_field() {
        check(
            r#"
struct Foo {
    $0
    pub f: i32,
}
"#,
            expect![[r#"
                kw pub(crate)
                kw pub
            "#]],
        )
    }

    #[test]
    fn skip_struct_initializer() {
        mark::check!(no_keyword_completion_in_record_lit);
        check(
            r#"
struct Foo {
    pub f: i32,
}
fn foo() {
    Foo {
        $0
    }
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn struct_initializer_field_expr() {
        check(
            r#"
struct Foo {
    pub f: i32,
}
fn foo() {
    Foo {
        f: $0
    }
}
"#,
            expect![[r#"
                kw match
                kw while
                kw loop
                kw if
                kw if let
                kw return
            "#]],
        );
    }

    #[test]
    fn let_semi() {
        mark::check!(let_semi);
        check_edit(
            "match",
            r#"
fn main() { let x = $0 }
"#,
            r#"
fn main() { let x = match $0 {}; }
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
    let x = if $0 {};
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
    let x = loop {$0};
    bar();
}
"#,
        );
    }
}
