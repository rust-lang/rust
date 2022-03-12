//! Postfix completions, like `Ok(10).ifl$0` => `if let Ok() = Ok(10) { $0 }`.

mod format_like;

use hir::{Documentation, HasAttrs};
use ide_db::{imports::insert_use::ImportScope, ty_filter::TryEnum, SnippetCap};
use syntax::{
    ast::{self, AstNode, AstToken},
    SyntaxKind::{EXPR_STMT, STMT_LIST},
    TextRange, TextSize,
};
use text_edit::TextEdit;

use crate::{
    completions::postfix::format_like::add_format_like_completions, context::CompletionContext,
    item::Builder, patterns::ImmediateLocation, CompletionItem, CompletionItemKind,
    CompletionRelevance, Completions, SnippetScope,
};

pub(crate) fn complete_postfix(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.config.enable_postfix_completions {
        return;
    }

    let (dot_receiver, receiver_is_ambiguous_float_literal) = match &ctx.completion_location {
        Some(ImmediateLocation::MethodCall { receiver: Some(it), .. }) => (it, false),
        Some(ImmediateLocation::FieldAccess {
            receiver: Some(it),
            receiver_is_ambiguous_float_literal,
        }) => (it, *receiver_is_ambiguous_float_literal),
        _ => return,
    };

    let receiver_text = get_receiver_text(dot_receiver, receiver_is_ambiguous_float_literal);

    let receiver_ty = match ctx.sema.type_of_expr(dot_receiver) {
        Some(it) => it.original,
        None => return,
    };

    // Suggest .await syntax for types that implement Future trait
    if receiver_ty.impls_future(ctx.db) {
        let mut item =
            CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), "await");
        item.detail("expr.await");
        item.add_to(acc);
    }

    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, dot_receiver) {
        Some(it) => it,
        None => return,
    };

    if let Some(drop_trait) = ctx.famous_defs().core_ops_Drop() {
        if receiver_ty.impls_trait(ctx.db, drop_trait, &[]) {
            if let &[hir::AssocItem::Function(drop_fn)] = &*drop_trait.items(ctx.db) {
                cov_mark::hit!(postfix_drop_completion);
                // FIXME: check that `drop` is in scope, use fully qualified path if it isn't/if shadowed
                let mut item = postfix_snippet(
                    "drop",
                    "fn drop(&mut self)",
                    &format!("drop($0{})", receiver_text),
                );
                item.set_documentation(drop_fn.docs(ctx.db));
                item.add_to(acc);
            }
        }
    }

    if !ctx.config.snippets.is_empty() {
        add_custom_postfix_completions(acc, ctx, &postfix_snippet, &receiver_text);
    }

    let try_enum = TryEnum::from_ty(&ctx.sema, &receiver_ty.strip_references());
    if let Some(try_enum) = &try_enum {
        match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    "ifl",
                    "if let Ok {}",
                    &format!("if let Ok($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);

                postfix_snippet(
                    "while",
                    "while let Ok {}",
                    &format!("while let Ok($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);
            }
            TryEnum::Option => {
                postfix_snippet(
                    "ifl",
                    "if let Some {}",
                    &format!("if let Some($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);

                postfix_snippet(
                    "while",
                    "while let Some {}",
                    &format!("while let Some($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);
            }
        }
    } else if receiver_ty.is_bool() || receiver_ty.is_unknown() {
        postfix_snippet("if", "if expr {}", &format!("if {} {{\n    $0\n}}", receiver_text))
            .add_to(acc);
        postfix_snippet(
            "while",
            "while expr {}",
            &format!("while {} {{\n    $0\n}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet("not", "!expr", &format!("!{}", receiver_text)).add_to(acc);
    } else if let Some(trait_) = ctx.famous_defs().core_iter_IntoIterator() {
        if receiver_ty.impls_trait(ctx.db, trait_, &[]) {
            postfix_snippet(
                "for",
                "for ele in expr {}",
                &format!("for ele in {} {{\n    $0\n}}", receiver_text),
            )
            .add_to(acc);
        }
    }

    postfix_snippet("ref", "&expr", &format!("&{}", receiver_text)).add_to(acc);
    postfix_snippet("refm", "&mut expr", &format!("&mut {}", receiver_text)).add_to(acc);

    // The rest of the postfix completions create an expression that moves an argument,
    // so it's better to consider references now to avoid breaking the compilation
    let dot_receiver = include_references(dot_receiver);
    let receiver_text = get_receiver_text(&dot_receiver, receiver_is_ambiguous_float_literal);
    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, &dot_receiver) {
        Some(it) => it,
        None => return,
    };

    match try_enum {
        Some(try_enum) => match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    "match",
                    "match expr {}",
                    &format!("match {} {{\n    Ok(${{1:_}}) => {{$2}},\n    Err(${{3:_}}) => {{$0}},\n}}", receiver_text),
                )
                .add_to(acc);
            }
            TryEnum::Option => {
                postfix_snippet(
                    "match",
                    "match expr {}",
                    &format!(
                        "match {} {{\n    Some(${{1:_}}) => {{$2}},\n    None => {{$0}},\n}}",
                        receiver_text
                    ),
                )
                .add_to(acc);
            }
        },
        None => {
            postfix_snippet(
                "match",
                "match expr {}",
                &format!("match {} {{\n    ${{1:_}} => {{$0}},\n}}", receiver_text),
            )
            .add_to(acc);
        }
    }

    postfix_snippet("box", "Box::new(expr)", &format!("Box::new({})", receiver_text)).add_to(acc);
    postfix_snippet("dbg", "dbg!(expr)", &format!("dbg!({})", receiver_text)).add_to(acc); // fixme
    postfix_snippet("dbgr", "dbg!(&expr)", &format!("dbg!(&{})", receiver_text)).add_to(acc);
    postfix_snippet("call", "function(expr)", &format!("${{1}}({})", receiver_text)).add_to(acc);

    if let Some(parent) = dot_receiver.syntax().parent().and_then(|p| p.parent()) {
        if matches!(parent.kind(), STMT_LIST | EXPR_STMT) {
            postfix_snippet("let", "let", &format!("let $0 = {};", receiver_text)).add_to(acc);
            postfix_snippet("letm", "let mut", &format!("let mut $0 = {};", receiver_text))
                .add_to(acc);
        }
    }

    if let ast::Expr::Literal(literal) = dot_receiver.clone() {
        if let Some(literal_text) = ast::String::cast(literal.token()) {
            add_format_like_completions(acc, ctx, &dot_receiver, cap, &literal_text);
        }
    }
}

fn get_receiver_text(receiver: &ast::Expr, receiver_is_ambiguous_float_literal: bool) -> String {
    if receiver_is_ambiguous_float_literal {
        let text = receiver.syntax().text();
        let without_dot = ..text.len() - TextSize::of('.');
        text.slice(without_dot).to_string()
    } else {
        receiver.to_string()
    }
}

fn include_references(initial_element: &ast::Expr) -> ast::Expr {
    let mut resulting_element = initial_element.clone();
    while let Some(parent_ref_element) =
        resulting_element.syntax().parent().and_then(ast::RefExpr::cast)
    {
        resulting_element = ast::Expr::from(parent_ref_element);
    }
    resulting_element
}

fn build_postfix_snippet_builder<'ctx>(
    ctx: &'ctx CompletionContext,
    cap: SnippetCap,
    receiver: &'ctx ast::Expr,
) -> Option<impl Fn(&str, &str, &str) -> Builder + 'ctx> {
    let receiver_syntax = receiver.syntax();
    let receiver_range = ctx.sema.original_range_opt(receiver_syntax)?.range;
    if ctx.source_range().end() < receiver_range.start() {
        // This shouldn't happen, yet it does. I assume this might be due to an incorrect token mapping.
        return None;
    }
    let delete_range = TextRange::new(receiver_range.start(), ctx.source_range().end());

    // Wrapping impl Fn in an option ruins lifetime inference for the parameters in a way that
    // can't be annotated for the closure, hence fix it by constructing it without the Option first
    fn build<'ctx>(
        ctx: &'ctx CompletionContext,
        cap: SnippetCap,
        delete_range: TextRange,
    ) -> impl Fn(&str, &str, &str) -> Builder + 'ctx {
        move |label, detail, snippet| {
            let edit = TextEdit::replace(delete_range, snippet.to_string());
            let mut item =
                CompletionItem::new(CompletionItemKind::Snippet, ctx.source_range(), label);
            item.detail(detail).snippet_edit(cap, edit);
            if ctx.original_token.text() == label {
                let relevance =
                    CompletionRelevance { exact_postfix_snippet_match: true, ..Default::default() };
                item.set_relevance(relevance);
            }

            item
        }
    }
    Some(build(ctx, cap, delete_range))
}

fn add_custom_postfix_completions(
    acc: &mut Completions,
    ctx: &CompletionContext,
    postfix_snippet: impl Fn(&str, &str, &str) -> Builder,
    receiver_text: &str,
) -> Option<()> {
    let import_scope = ImportScope::find_insert_use_container(&ctx.token.parent()?, &ctx.sema)?;
    ctx.config.postfix_snippets().filter(|(_, snip)| snip.scope == SnippetScope::Expr).for_each(
        |(trigger, snippet)| {
            let imports = match snippet.imports(ctx, &import_scope) {
                Some(imports) => imports,
                None => return,
            };
            let body = snippet.postfix_snippet(receiver_text);
            let mut builder =
                postfix_snippet(trigger, snippet.description.as_deref().unwrap_or_default(), &body);
            builder.documentation(Documentation::new(format!("```rust\n{}\n```", body)));
            for import in imports.into_iter() {
                builder.add_import(import);
            }
            builder.add_to(acc);
        },
    );
    None
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        tests::{check_edit, check_edit_with_config, completion_list, TEST_CONFIG},
        CompletionConfig, Snippet,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture);
        expect.assert_eq(&actual)
    }

    #[test]
    fn postfix_completion_works_for_trivial_path_expression() {
        check(
            r#"
fn main() {
    let bar = true;
    bar.$0
}
"#,
            expect![[r#"
                sn if    if expr {}
                sn while while expr {}
                sn not   !expr
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
    }

    #[test]
    fn postfix_completion_works_for_function_calln() {
        check(
            r#"
fn foo(elt: bool) -> bool {
    !elt
}

fn main() {
    let bar = true;
    foo(bar.$0)
}
"#,
            expect![[r#"
                sn if    if expr {}
                sn while while expr {}
                sn not   !expr
                sn ref   &expr
                sn refm  &mut expr
                sn match match expr {}
                sn box   Box::new(expr)
                sn dbg   dbg!(expr)
                sn dbgr  dbg!(&expr)
                sn call  function(expr)
            "#]],
        );
    }

    #[test]
    fn postfix_type_filtering() {
        check(
            r#"
fn main() {
    let bar: u8 = 12;
    bar.$0
}
"#,
            expect![[r#"
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
    fn let_middle_block() {
        check(
            r#"
fn main() {
    baz.l$0
    res
}
"#,
            expect![[r#"
                sn if    if expr {}
                sn while while expr {}
                sn not   !expr
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
    }

    #[test]
    fn option_iflet() {
        check_edit(
            "ifl",
            r#"
//- minicore: option
fn main() {
    let bar = Some(true);
    bar.$0
}
"#,
            r#"
fn main() {
    let bar = Some(true);
    if let Some($1) = bar {
    $0
}
}
"#,
        );
    }

    #[test]
    fn result_match() {
        check_edit(
            "match",
            r#"
//- minicore: result
fn main() {
    let bar = Ok(true);
    bar.$0
}
"#,
            r#"
fn main() {
    let bar = Ok(true);
    match bar {
    Ok(${1:_}) => {$2},
    Err(${3:_}) => {$0},
}
}
"#,
        );
    }

    #[test]
    fn postfix_completion_works_for_ambiguous_float_literal() {
        check_edit("refm", r#"fn main() { 42.$0 }"#, r#"fn main() { &mut 42 }"#)
    }

    #[test]
    fn works_in_simple_macro() {
        check_edit(
            "dbg",
            r#"
macro_rules! m { ($e:expr) => { $e } }
fn main() {
    let bar: u8 = 12;
    m!(bar.d$0)
}
"#,
            r#"
macro_rules! m { ($e:expr) => { $e } }
fn main() {
    let bar: u8 = 12;
    m!(dbg!(bar))
}
"#,
        );
    }

    #[test]
    fn postfix_completion_for_references() {
        check_edit("dbg", r#"fn main() { &&42.$0 }"#, r#"fn main() { dbg!(&&42) }"#);
        check_edit("refm", r#"fn main() { &&42.$0 }"#, r#"fn main() { &&&mut 42 }"#);
        check_edit(
            "ifl",
            r#"
//- minicore: option
fn main() {
    let bar = &Some(true);
    bar.$0
}
"#,
            r#"
fn main() {
    let bar = &Some(true);
    if let Some($1) = bar {
    $0
}
}
"#,
        )
    }

    #[test]
    fn custom_postfix_completion() {
        check_edit_with_config(
            CompletionConfig {
                snippets: vec![Snippet::new(
                    &[],
                    &["break".into()],
                    &["ControlFlow::Break(${receiver})".into()],
                    "",
                    &["core::ops::ControlFlow".into()],
                    crate::SnippetScope::Expr,
                )
                .unwrap()],
                ..TEST_CONFIG
            },
            "break",
            r#"
//- minicore: try
fn main() { 42.$0 }
"#,
            r#"
use core::ops::ControlFlow;

fn main() { ControlFlow::Break(42) }
"#,
        );
    }

    #[test]
    fn postfix_completion_for_format_like_strings() {
        check_edit(
            "format",
            r#"fn main() { "{some_var:?}".$0 }"#,
            r#"fn main() { format!("{:?}", some_var) }"#,
        );
        check_edit(
            "panic",
            r#"fn main() { "Panic with {a}".$0 }"#,
            r#"fn main() { panic!("Panic with {}", a) }"#,
        );
        check_edit(
            "println",
            r#"fn main() { "{ 2+2 } { SomeStruct { val: 1, other: 32 } :?}".$0 }"#,
            r#"fn main() { println!("{} {:?}", 2+2, SomeStruct { val: 1, other: 32 }) }"#,
        );
        check_edit(
            "loge",
            r#"fn main() { "{2+2}".$0 }"#,
            r#"fn main() { log::error!("{}", 2+2) }"#,
        );
        check_edit(
            "logt",
            r#"fn main() { "{2+2}".$0 }"#,
            r#"fn main() { log::trace!("{}", 2+2) }"#,
        );
        check_edit(
            "logd",
            r#"fn main() { "{2+2}".$0 }"#,
            r#"fn main() { log::debug!("{}", 2+2) }"#,
        );
        check_edit("logi", r#"fn main() { "{2+2}".$0 }"#, r#"fn main() { log::info!("{}", 2+2) }"#);
        check_edit("logw", r#"fn main() { "{2+2}".$0 }"#, r#"fn main() { log::warn!("{}", 2+2) }"#);
        check_edit(
            "loge",
            r#"fn main() { "{2+2}".$0 }"#,
            r#"fn main() { log::error!("{}", 2+2) }"#,
        );
    }
}
