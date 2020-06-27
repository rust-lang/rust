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
}

pub(super) fn complete_expr_keyword(acc: &mut Completions, ctx: &CompletionContext) {
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
    use crate::completion::{test_utils::completion_list, CompletionKind};
    use insta::assert_snapshot;

    fn get_keyword_completions(code: &str) -> String {
        completion_list(code, CompletionKind::Keyword)
    }

    #[test]
    fn test_keywords_in_use_stmt() {
        assert_snapshot!(
            get_keyword_completions(r"use <|>"),
            @r###"
            kw crate::
            kw self
            kw super::
        "###
        );

        assert_snapshot!(
            get_keyword_completions(r"use a::<|>"),
            @r###"
            kw self
            kw super::
        "###
        );

        assert_snapshot!(
            get_keyword_completions(r"use a::{b, <|>}"),
            @r###"
            kw self
            kw super::
        "###
        );
    }

    #[test]
    fn test_keywords_at_source_file_level() {
        assert_snapshot!(
            get_keyword_completions(r"m<|>"),
            @r###"
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
        "###
        );
    }

    #[test]
    fn test_keywords_in_function() {
        assert_snapshot!(
            get_keyword_completions(r"fn quux() { <|> }"),
            @r###"
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
        "###
        );
    }

    #[test]
    fn test_keywords_inside_block() {
        assert_snapshot!(
            get_keyword_completions(r"fn quux() { if true { <|> } }"),
            @r###"
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
        "###
        );
    }

    #[test]
    fn test_keywords_after_if() {
        assert_snapshot!(
            get_keyword_completions(
                r"
                fn quux() {
                    if true {
                        ()
                    } <|>
                }
                ",
            ),
            @r###"
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
        "###
        );
    }

    #[test]
    fn test_keywords_in_match_arm() {
        assert_snapshot!(
            get_keyword_completions(
                r"
                fn quux() -> i32 {
                    match () {
                        () => <|>
                    }
                }
                ",
            ),
            @r###"
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw unsafe
        "###
        );
    }

    #[test]
    fn test_keywords_in_trait_def() {
        assert_snapshot!(
            get_keyword_completions(r"trait My { <|> }"),
            @r###"
            kw const
            kw fn
            kw type
            kw unsafe
        "###
        );
    }

    #[test]
    fn test_keywords_in_impl_def() {
        assert_snapshot!(
            get_keyword_completions(r"impl My { <|> }"),
            @r###"
            kw const
            kw fn
            kw pub
            kw type
            kw unsafe
        "###
        );
    }

    #[test]
    fn test_keywords_in_loop() {
        assert_snapshot!(
            get_keyword_completions(r"fn my() { loop { <|> } }"),
            @r###"
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
        "###
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_item_list() {
        assert_snapshot!(
            get_keyword_completions(r"unsafe <|>"),
            @r###"
            kw fn
            kw impl
            kw trait
        "###
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_block_expr() {
        assert_snapshot!(
            get_keyword_completions(r"fn my_fn() { unsafe <|> }"),
            @r###"
            kw fn
            kw impl
            kw trait
        "###
        );
    }

    #[test]
    fn test_mut_in_ref_and_in_fn_parameters_list() {
        assert_snapshot!(
            get_keyword_completions(r"fn my_fn(&<|>) {}"),
            @r###"
            kw mut
        "###
        );
        assert_snapshot!(
            get_keyword_completions(r"fn my_fn(<|>) {}"),
            @r###"
            kw mut
        "###
        );
        assert_snapshot!(
            get_keyword_completions(r"fn my_fn() { let &<|> }"),
            @r###"
            kw mut
        "###
        );
    }

    #[test]
    fn test_where_keyword() {
        assert_snapshot!(
            get_keyword_completions(r"trait A <|>"),
            @r###"
            kw where
        "###
        );
        assert_snapshot!(
            get_keyword_completions(r"impl A <|>"),
            @r###"
            kw where
        "###
        );
    }
}
