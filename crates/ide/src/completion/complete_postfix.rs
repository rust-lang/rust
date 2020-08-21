//! FIXME: write short doc here
use assists::utils::TryEnum;
use syntax::{
    ast::{self, AstNode},
    TextRange, TextSize,
};
use text_edit::TextEdit;

use crate::{
    completion::{
        completion_config::SnippetCap,
        completion_context::CompletionContext,
        completion_item::{Builder, CompletionKind, Completions},
    },
    CompletionItem, CompletionItemKind,
};

pub(super) fn complete_postfix(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.config.enable_postfix_completions {
        return;
    }

    let dot_receiver = match &ctx.dot_receiver {
        Some(it) => it,
        None => return,
    };

    let receiver_text =
        get_receiver_text(dot_receiver, ctx.dot_receiver_is_ambiguous_float_literal);

    let receiver_ty = match ctx.sema.type_of_expr(&dot_receiver) {
        Some(it) => it,
        None => return,
    };

    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };
    let try_enum = TryEnum::from_ty(&ctx.sema, &receiver_ty);
    if let Some(try_enum) = &try_enum {
        match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    ctx,
                    cap,
                    &dot_receiver,
                    "ifl",
                    "if let Ok {}",
                    &format!("if let Ok($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);

                postfix_snippet(
                    ctx,
                    cap,
                    &dot_receiver,
                    "while",
                    "while let Ok {}",
                    &format!("while let Ok($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);
            }
            TryEnum::Option => {
                postfix_snippet(
                    ctx,
                    cap,
                    &dot_receiver,
                    "ifl",
                    "if let Some {}",
                    &format!("if let Some($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);

                postfix_snippet(
                    ctx,
                    cap,
                    &dot_receiver,
                    "while",
                    "while let Some {}",
                    &format!("while let Some($1) = {} {{\n    $0\n}}", receiver_text),
                )
                .add_to(acc);
            }
        }
    } else if receiver_ty.is_bool() || receiver_ty.is_unknown() {
        postfix_snippet(
            ctx,
            cap,
            &dot_receiver,
            "if",
            "if expr {}",
            &format!("if {} {{\n    $0\n}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet(
            ctx,
            cap,
            &dot_receiver,
            "while",
            "while expr {}",
            &format!("while {} {{\n    $0\n}}", receiver_text),
        )
        .add_to(acc);
        postfix_snippet(ctx, cap, &dot_receiver, "not", "!expr", &format!("!{}", receiver_text))
            .add_to(acc);
    }

    postfix_snippet(ctx, cap, &dot_receiver, "ref", "&expr", &format!("&{}", receiver_text))
        .add_to(acc);
    postfix_snippet(
        ctx,
        cap,
        &dot_receiver,
        "refm",
        "&mut expr",
        &format!("&mut {}", receiver_text),
    )
    .add_to(acc);

    // The rest of the postfix completions create an expression that moves an argument,
    // so it's better to consider references now to avoid breaking the compilation
    let dot_receiver = include_references(dot_receiver);
    let receiver_text =
        get_receiver_text(&dot_receiver, ctx.dot_receiver_is_ambiguous_float_literal);

    match try_enum {
        Some(try_enum) => match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    ctx,
                    cap,
                    &dot_receiver,
                    "match",
                    "match expr {}",
                    &format!("match {} {{\n    Ok(${{1:_}}) => {{$2}},\n    Err(${{3:_}}) => {{$0}},\n}}", receiver_text),
                )
                .add_to(acc);
            }
            TryEnum::Option => {
                postfix_snippet(
                    ctx,
                    cap,
                    &dot_receiver,
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
                ctx,
                cap,
                &dot_receiver,
                "match",
                "match expr {}",
                &format!("match {} {{\n    ${{1:_}} => {{$0}},\n}}", receiver_text),
            )
            .add_to(acc);
        }
    }

    postfix_snippet(
        ctx,
        cap,
        &dot_receiver,
        "box",
        "Box::new(expr)",
        &format!("Box::new({})", receiver_text),
    )
    .add_to(acc);

    postfix_snippet(
        ctx,
        cap,
        &dot_receiver,
        "dbg",
        "dbg!(expr)",
        &format!("dbg!({})", receiver_text),
    )
    .add_to(acc);

    postfix_snippet(
        ctx,
        cap,
        &dot_receiver,
        "call",
        "function(expr)",
        &format!("${{1}}({})", receiver_text),
    )
    .add_to(acc);
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

fn postfix_snippet(
    ctx: &CompletionContext,
    cap: SnippetCap,
    receiver: &ast::Expr,
    label: &str,
    detail: &str,
    snippet: &str,
) -> Builder {
    let edit = {
        let receiver_syntax = receiver.syntax();
        let receiver_range = ctx.sema.original_range(receiver_syntax).range;
        let delete_range = TextRange::new(receiver_range.start(), ctx.source_range().end());
        TextEdit::replace(delete_range, snippet.to_string())
    };
    CompletionItem::new(CompletionKind::Postfix, ctx.source_range(), label)
        .detail(detail)
        .kind(CompletionItemKind::Snippet)
        .snippet_edit(cap, edit)
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::completion::{
        test_utils::{check_edit, completion_list},
        CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Postfix);
        expect.assert_eq(&actual)
    }

    #[test]
    fn postfix_completion_works_for_trivial_path_expression() {
        check(
            r#"
fn main() {
    let bar = true;
    bar.<|>
}
"#,
            expect![[r#"
                sn box   Box::new(expr)
                sn call  function(expr)
                sn dbg   dbg!(expr)
                sn if    if expr {}
                sn match match expr {}
                sn not   !expr
                sn ref   &expr
                sn refm  &mut expr
                sn while while expr {}
            "#]],
        );
    }

    #[test]
    fn postfix_type_filtering() {
        check(
            r#"
fn main() {
    let bar: u8 = 12;
    bar.<|>
}
"#,
            expect![[r#"
                sn box   Box::new(expr)
                sn call  function(expr)
                sn dbg   dbg!(expr)
                sn match match expr {}
                sn ref   &expr
                sn refm  &mut expr
            "#]],
        )
    }

    #[test]
    fn option_iflet() {
        check_edit(
            "ifl",
            r#"
enum Option<T> { Some(T), None }

fn main() {
    let bar = Option::Some(true);
    bar.<|>
}
"#,
            r#"
enum Option<T> { Some(T), None }

fn main() {
    let bar = Option::Some(true);
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
enum Result<T, E> { Ok(T), Err(E) }

fn main() {
    let bar = Result::Ok(true);
    bar.<|>
}
"#,
            r#"
enum Result<T, E> { Ok(T), Err(E) }

fn main() {
    let bar = Result::Ok(true);
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
        check_edit("refm", r#"fn main() { 42.<|> }"#, r#"fn main() { &mut 42 }"#)
    }

    #[test]
    fn works_in_simple_macro() {
        check_edit(
            "dbg",
            r#"
macro_rules! m { ($e:expr) => { $e } }
fn main() {
    let bar: u8 = 12;
    m!(bar.d<|>)
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
        check_edit("dbg", r#"fn main() { &&42.<|> }"#, r#"fn main() { dbg!(&&42) }"#);
        check_edit("refm", r#"fn main() { &&42.<|> }"#, r#"fn main() { &&&mut 42 }"#);
    }
}
