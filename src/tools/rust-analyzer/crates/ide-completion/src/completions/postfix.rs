//! Postfix completions, like `Ok(10).ifl$0` => `if let Ok() = Ok(10) { $0 }`.

mod format_like;

use base_db::SourceDatabase;
use hir::{ItemInNs, Semantics};
use ide_db::{
    RootDatabase, SnippetCap,
    documentation::{Documentation, HasDocs},
    imports::insert_use::ImportScope,
    text_edit::TextEdit,
    ty_filter::TryEnum,
};
use stdx::never;
use syntax::{
    SyntaxKind::{BLOCK_EXPR, EXPR_STMT, FOR_EXPR, IF_EXPR, LOOP_EXPR, STMT_LIST, WHILE_EXPR},
    TextRange, TextSize,
    ast::{self, AstNode, AstToken},
};

use crate::{
    CompletionItem, CompletionItemKind, CompletionRelevance, Completions, SnippetScope,
    completions::postfix::format_like::add_format_like_completions,
    context::{BreakableKind, CompletionContext, DotAccess, DotAccessKind},
    item::{Builder, CompletionRelevancePostfixMatch},
};

pub(crate) fn complete_postfix(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_access: &DotAccess,
) {
    if !ctx.config.enable_postfix_completions {
        return;
    }

    let (dot_receiver, receiver_ty, receiver_is_ambiguous_float_literal) = match dot_access {
        DotAccess { receiver_ty: Some(ty), receiver: Some(it), kind, .. } => (
            it,
            &ty.original,
            match *kind {
                DotAccessKind::Field { receiver_is_ambiguous_float_literal } => {
                    receiver_is_ambiguous_float_literal
                }
                DotAccessKind::Method { .. } => false,
            },
        ),
        _ => return,
    };
    let expr_ctx = &dot_access.ctx;

    let receiver_text =
        get_receiver_text(&ctx.sema, dot_receiver, receiver_is_ambiguous_float_literal);

    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, dot_receiver) {
        Some(it) => it,
        None => return,
    };

    let cfg = ctx.config.import_path_config(ctx.is_nightly);

    if let Some(drop_trait) = ctx.famous_defs().core_ops_Drop() {
        if receiver_ty.impls_trait(ctx.db, drop_trait, &[]) {
            if let Some(drop_fn) = ctx.famous_defs().core_mem_drop() {
                if let Some(path) =
                    ctx.module.find_path(ctx.db, ItemInNs::Values(drop_fn.into()), cfg)
                {
                    cov_mark::hit!(postfix_drop_completion);
                    let mut item = postfix_snippet(
                        "drop",
                        "fn drop(&mut self)",
                        &format!(
                            "{path}($0{receiver_text})",
                            path = path.display(ctx.db, ctx.edition)
                        ),
                    );
                    item.set_documentation(drop_fn.docs(ctx.db));
                    item.add_to(acc, ctx.db);
                }
            }
        }
    }

    let try_enum = TryEnum::from_ty(&ctx.sema, &receiver_ty.strip_references());
    if let Some(try_enum) = &try_enum {
        match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    "ifl",
                    "if let Ok {}",
                    &format!("if let Ok($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "lete",
                    "let Ok else {}",
                    &format!("let Ok($1) = {receiver_text} else {{\n    $2\n}};\n$0"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "while",
                    "while let Ok {}",
                    &format!("while let Ok($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);
            }
            TryEnum::Option => {
                postfix_snippet(
                    "ifl",
                    "if let Some {}",
                    &format!("if let Some($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "lete",
                    "let Some else {}",
                    &format!("let Some($1) = {receiver_text} else {{\n    $2\n}};\n$0"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "while",
                    "while let Some {}",
                    &format!("while let Some($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);
            }
        }
    } else if receiver_ty.is_bool() || receiver_ty.is_unknown() {
        postfix_snippet("if", "if expr {}", &format!("if {receiver_text} {{\n    $0\n}}"))
            .add_to(acc, ctx.db);
        postfix_snippet("while", "while expr {}", &format!("while {receiver_text} {{\n    $0\n}}"))
            .add_to(acc, ctx.db);
        postfix_snippet("not", "!expr", &format!("!{receiver_text}")).add_to(acc, ctx.db);
    } else if let Some(trait_) = ctx.famous_defs().core_iter_IntoIterator() {
        if receiver_ty.impls_trait(ctx.db, trait_, &[]) {
            postfix_snippet(
                "for",
                "for ele in expr {}",
                &format!("for ele in {receiver_text} {{\n    $0\n}}"),
            )
            .add_to(acc, ctx.db);
        }
    }

    postfix_snippet("ref", "&expr", &format!("&{receiver_text}")).add_to(acc, ctx.db);
    postfix_snippet("refm", "&mut expr", &format!("&mut {receiver_text}")).add_to(acc, ctx.db);
    postfix_snippet("deref", "*expr", &format!("*{receiver_text}")).add_to(acc, ctx.db);

    let mut block_should_be_wrapped = true;
    if dot_receiver.syntax().kind() == BLOCK_EXPR {
        block_should_be_wrapped = false;
        if let Some(parent) = dot_receiver.syntax().parent() {
            if matches!(parent.kind(), IF_EXPR | WHILE_EXPR | LOOP_EXPR | FOR_EXPR) {
                block_should_be_wrapped = true;
            }
        }
    };
    let unsafe_completion_string = if block_should_be_wrapped {
        format!("unsafe {{ {receiver_text} }}")
    } else {
        format!("unsafe {receiver_text}")
    };
    postfix_snippet("unsafe", "unsafe {}", &unsafe_completion_string).add_to(acc, ctx.db);

    let const_completion_string = if block_should_be_wrapped {
        format!("const {{ {receiver_text} }}")
    } else {
        format!("const {receiver_text}")
    };
    postfix_snippet("const", "const {}", &const_completion_string).add_to(acc, ctx.db);

    // The rest of the postfix completions create an expression that moves an argument,
    // so it's better to consider references now to avoid breaking the compilation

    let (dot_receiver_including_refs, prefix) = include_references(dot_receiver);
    let mut receiver_text =
        get_receiver_text(&ctx.sema, dot_receiver, receiver_is_ambiguous_float_literal);
    receiver_text.insert_str(0, &prefix);
    let postfix_snippet =
        match build_postfix_snippet_builder(ctx, cap, &dot_receiver_including_refs) {
            Some(it) => it,
            None => return,
        };

    if !ctx.config.snippets.is_empty() {
        add_custom_postfix_completions(acc, ctx, &postfix_snippet, &receiver_text);
    }

    match try_enum {
        Some(try_enum) => match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    "match",
                    "match expr {}",
                    &format!("match {receiver_text} {{\n    Ok(${{1:_}}) => {{$2}},\n    Err(${{3:_}}) => {{$0}},\n}}"),
                )
                .add_to(acc, ctx.db);
            }
            TryEnum::Option => {
                postfix_snippet(
                    "match",
                    "match expr {}",
                    &format!(
                        "match {receiver_text} {{\n    Some(${{1:_}}) => {{$2}},\n    None => {{$0}},\n}}"
                    ),
                )
                .add_to(acc, ctx.db);
            }
        },
        None => {
            postfix_snippet(
                "match",
                "match expr {}",
                &format!("match {receiver_text} {{\n    ${{1:_}} => {{$0}},\n}}"),
            )
            .add_to(acc, ctx.db);
        }
    }

    postfix_snippet("box", "Box::new(expr)", &format!("Box::new({receiver_text})"))
        .add_to(acc, ctx.db);
    postfix_snippet("dbg", "dbg!(expr)", &format!("dbg!({receiver_text})")).add_to(acc, ctx.db); // fixme
    postfix_snippet("dbgr", "dbg!(&expr)", &format!("dbg!(&{receiver_text})")).add_to(acc, ctx.db);
    postfix_snippet("call", "function(expr)", &format!("${{1}}({receiver_text})"))
        .add_to(acc, ctx.db);

    if let Some(parent) = dot_receiver_including_refs.syntax().parent().and_then(|p| p.parent()) {
        if matches!(parent.kind(), STMT_LIST | EXPR_STMT) {
            postfix_snippet("let", "let", &format!("let $0 = {receiver_text};"))
                .add_to(acc, ctx.db);
            postfix_snippet("letm", "let mut", &format!("let mut $0 = {receiver_text};"))
                .add_to(acc, ctx.db);
        }
    }

    if let ast::Expr::Literal(literal) = dot_receiver_including_refs.clone() {
        if let Some(literal_text) = ast::String::cast(literal.token()) {
            add_format_like_completions(acc, ctx, &dot_receiver_including_refs, cap, &literal_text);
        }
    }

    postfix_snippet(
        "return",
        "return expr",
        &format!(
            "return {receiver_text}{semi}",
            semi = if expr_ctx.in_block_expr { ";" } else { "" }
        ),
    )
    .add_to(acc, ctx.db);

    if let BreakableKind::Block | BreakableKind::Loop = expr_ctx.in_breakable {
        postfix_snippet(
            "break",
            "break expr",
            &format!(
                "break {receiver_text}{semi}",
                semi = if expr_ctx.in_block_expr { ";" } else { "" }
            ),
        )
        .add_to(acc, ctx.db);
    }
}

fn get_receiver_text(
    sema: &Semantics<'_, RootDatabase>,
    receiver: &ast::Expr,
    receiver_is_ambiguous_float_literal: bool,
) -> String {
    // Do not just call `receiver.to_string()`, as that will mess up whitespaces inside macros.
    let Some(mut range) = sema.original_range_opt(receiver.syntax()) else {
        return receiver.to_string();
    };
    if receiver_is_ambiguous_float_literal {
        range.range = TextRange::at(range.range.start(), range.range.len() - TextSize::of('.'))
    }
    let file_text = sema.db.file_text(range.file_id.file_id(sema.db));
    let mut text = file_text.text(sema.db)[range.range].to_owned();

    // The receiver texts should be interpreted as-is, as they are expected to be
    // normal Rust expressions.
    escape_snippet_bits(&mut text);
    text
}

/// Escapes `\` and `$` so that they don't get interpreted as snippet-specific constructs.
///
/// Note that we don't need to escape the other characters that can be escaped,
/// because they wouldn't be treated as snippet-specific constructs without '$'.
fn escape_snippet_bits(text: &mut String) {
    stdx::replace(text, '\\', "\\\\");
    stdx::replace(text, '$', "\\$");
}

fn include_references(initial_element: &ast::Expr) -> (ast::Expr, String) {
    let mut resulting_element = initial_element.clone();

    while let Some(field_expr) = resulting_element.syntax().parent().and_then(ast::FieldExpr::cast)
    {
        resulting_element = ast::Expr::from(field_expr);
    }

    let mut prefix = String::new();

    let mut found_ref_or_deref = false;

    while let Some(parent_deref_element) =
        resulting_element.syntax().parent().and_then(ast::PrefixExpr::cast)
    {
        if parent_deref_element.op_kind() != Some(ast::UnaryOp::Deref) {
            break;
        }

        found_ref_or_deref = true;
        resulting_element = ast::Expr::from(parent_deref_element);

        prefix.insert(0, '*');
    }

    while let Some(parent_ref_element) =
        resulting_element.syntax().parent().and_then(ast::RefExpr::cast)
    {
        found_ref_or_deref = true;
        let exclusive = parent_ref_element.mut_token().is_some();
        resulting_element = ast::Expr::from(parent_ref_element);

        prefix.insert_str(0, if exclusive { "&mut " } else { "&" });
    }

    if !found_ref_or_deref {
        // If we do not find any ref/deref expressions, restore
        // all the progress of tree climbing
        prefix.clear();
        resulting_element = initial_element.clone();
    }

    (resulting_element, prefix)
}

fn build_postfix_snippet_builder<'ctx>(
    ctx: &'ctx CompletionContext<'_>,
    cap: SnippetCap,
    receiver: &'ctx ast::Expr,
) -> Option<impl Fn(&str, &str, &str) -> Builder + 'ctx> {
    let receiver_range = ctx.sema.original_range_opt(receiver.syntax())?.range;
    if ctx.source_range().end() < receiver_range.start() {
        // This shouldn't happen, yet it does. I assume this might be due to an incorrect token
        // mapping.
        never!();
        return None;
    }
    let delete_range = TextRange::new(receiver_range.start(), ctx.source_range().end());

    // Wrapping impl Fn in an option ruins lifetime inference for the parameters in a way that
    // can't be annotated for the closure, hence fix it by constructing it without the Option first
    fn build<'ctx>(
        ctx: &'ctx CompletionContext<'_>,
        cap: SnippetCap,
        delete_range: TextRange,
    ) -> impl Fn(&str, &str, &str) -> Builder + 'ctx {
        move |label, detail, snippet| {
            let edit = TextEdit::replace(delete_range, snippet.to_owned());
            let mut item = CompletionItem::new(
                CompletionItemKind::Snippet,
                ctx.source_range(),
                label,
                ctx.edition,
            );
            item.detail(detail).snippet_edit(cap, edit);
            let postfix_match = if ctx.original_token.text() == label {
                cov_mark::hit!(postfix_exact_match_is_high_priority);
                Some(CompletionRelevancePostfixMatch::Exact)
            } else {
                cov_mark::hit!(postfix_inexact_match_is_low_priority);
                Some(CompletionRelevancePostfixMatch::NonExact)
            };
            let relevance = CompletionRelevance { postfix_match, ..Default::default() };
            item.set_relevance(relevance);
            item
        }
    }
    Some(build(ctx, cap, delete_range))
}

fn add_custom_postfix_completions(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    postfix_snippet: impl Fn(&str, &str, &str) -> Builder,
    receiver_text: &str,
) -> Option<()> {
    ImportScope::find_insert_use_container(&ctx.token.parent()?, &ctx.sema)?;
    ctx.config.postfix_snippets().filter(|(_, snip)| snip.scope == SnippetScope::Expr).for_each(
        |(trigger, snippet)| {
            let imports = match snippet.imports(ctx) {
                Some(imports) => imports,
                None => return,
            };
            let body = snippet.postfix_snippet(receiver_text);
            let mut builder =
                postfix_snippet(trigger, snippet.description.as_deref().unwrap_or_default(), &body);
            builder.documentation(Documentation::new(format!("```rust\n{body}\n```")));
            for import in imports.into_iter() {
                builder.add_import(import);
            }
            builder.add_to(acc, ctx.db);
        },
    );
    None
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::{
        CompletionConfig, Snippet,
        tests::{TEST_CONFIG, check, check_edit, check_edit_with_config},
    };

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
                sn box  Box::new(expr)
                sn call function(expr)
                sn const      const {}
                sn dbg      dbg!(expr)
                sn dbgr    dbg!(&expr)
                sn deref         *expr
                sn if       if expr {}
                sn let             let
                sn letm        let mut
                sn match match expr {}
                sn not           !expr
                sn ref           &expr
                sn refm      &mut expr
                sn return  return expr
                sn unsafe    unsafe {}
                sn while while expr {}
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
                sn box  Box::new(expr)
                sn call function(expr)
                sn const      const {}
                sn dbg      dbg!(expr)
                sn dbgr    dbg!(&expr)
                sn deref         *expr
                sn if       if expr {}
                sn match match expr {}
                sn not           !expr
                sn ref           &expr
                sn refm      &mut expr
                sn return  return expr
                sn unsafe    unsafe {}
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
    bar.$0
}
"#,
            expect![[r#"
                sn box  Box::new(expr)
                sn call function(expr)
                sn const      const {}
                sn dbg      dbg!(expr)
                sn dbgr    dbg!(&expr)
                sn deref         *expr
                sn let             let
                sn letm        let mut
                sn match match expr {}
                sn ref           &expr
                sn refm      &mut expr
                sn return  return expr
                sn unsafe    unsafe {}
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
                sn box  Box::new(expr)
                sn call function(expr)
                sn const      const {}
                sn dbg      dbg!(expr)
                sn dbgr    dbg!(&expr)
                sn deref         *expr
                sn if       if expr {}
                sn let             let
                sn letm        let mut
                sn match match expr {}
                sn not           !expr
                sn ref           &expr
                sn refm      &mut expr
                sn return  return expr
                sn unsafe    unsafe {}
                sn while while expr {}
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
    fn option_letelse() {
        check_edit(
            "lete",
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
    let Some($1) = bar else {
    $2
};
$0
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
    fn postfix_completion_for_unsafe() {
        postfix_completion_for_block("unsafe");
    }

    #[test]
    fn postfix_completion_for_const() {
        postfix_completion_for_block("const");
    }

    fn postfix_completion_for_block(kind: &str) {
        check_edit(kind, r#"fn main() { foo.$0 }"#, &format!("fn main() {{ {kind} {{ foo }} }}"));
        check_edit(
            kind,
            r#"fn main() { { foo }.$0 }"#,
            &format!("fn main() {{ {kind} {{ foo }} }}"),
        );
        check_edit(
            kind,
            r#"fn main() { if x { foo }.$0 }"#,
            &format!("fn main() {{ {kind} {{ if x {{ foo }} }} }}"),
        );
        check_edit(
            kind,
            r#"fn main() { loop { foo }.$0 }"#,
            &format!("fn main() {{ {kind} {{ loop {{ foo }} }} }}"),
        );
        check_edit(
            kind,
            r#"fn main() { if true {}.$0 }"#,
            &format!("fn main() {{ {kind} {{ if true {{}} }} }}"),
        );
        check_edit(
            kind,
            r#"fn main() { while true {}.$0 }"#,
            &format!("fn main() {{ {kind} {{ while true {{}} }} }}"),
        );
        check_edit(
            kind,
            r#"fn main() { for i in 0..10 {}.$0 }"#,
            &format!("fn main() {{ {kind} {{ for i in 0..10 {{}} }} }}"),
        );
        check_edit(
            kind,
            r#"fn main() { let x = if true {1} else {2}.$0 }"#,
            &format!("fn main() {{ let x = {kind} {{ if true {{1}} else {{2}} }} }}"),
        );

        // completion will not be triggered
        check_edit(
            kind,
            r#"fn main() { let x = true else {panic!()}.$0}"#,
            &format!("fn main() {{ let x = true else {{panic!()}}.{kind} $0}}"),
        );
    }

    #[test]
    fn custom_postfix_completion() {
        let config = CompletionConfig {
            snippets: vec![
                Snippet::new(
                    &[],
                    &["break".into()],
                    &["ControlFlow::Break(${receiver})".into()],
                    "",
                    &["core::ops::ControlFlow".into()],
                    crate::SnippetScope::Expr,
                )
                .unwrap(),
            ],
            ..TEST_CONFIG
        };

        check_edit_with_config(
            config.clone(),
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

        // The receiver texts should be escaped, see comments in `get_receiver_text()`
        // for detail.
        //
        // Note that the last argument is what *lsp clients would see* rather than
        // what users would see. Unescaping happens thereafter.
        check_edit_with_config(
            config.clone(),
            "break",
            r#"
//- minicore: try
fn main() { '\\'.$0 }
"#,
            r#"
use core::ops::ControlFlow;

fn main() { ControlFlow::Break('\\\\') }
"#,
        );

        check_edit_with_config(
            config,
            "break",
            r#"
//- minicore: try
fn main() {
    match true {
        true => "${1:placeholder}",
        false => "\$",
    }.$0
}
"#,
            r#"
use core::ops::ControlFlow;

fn main() {
    ControlFlow::Break(match true {
        true => "\${1:placeholder}",
        false => "\\\$",
    })
}
"#,
        );
    }

    #[test]
    fn postfix_completion_for_format_like_strings() {
        check_edit(
            "format",
            r#"fn main() { "{some_var:?}".$0 }"#,
            r#"fn main() { format!("{some_var:?}") }"#,
        );
        check_edit(
            "panic",
            r#"fn main() { "Panic with {a}".$0 }"#,
            r#"fn main() { panic!("Panic with {a}") }"#,
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

    #[test]
    fn postfix_custom_snippets_completion_for_references() {
        // https://github.com/rust-lang/rust-analyzer/issues/7929

        let snippet = Snippet::new(
            &[],
            &["ok".into()],
            &["Ok(${receiver})".into()],
            "",
            &[],
            crate::SnippetScope::Expr,
        )
        .unwrap();

        check_edit_with_config(
            CompletionConfig { snippets: vec![snippet.clone()], ..TEST_CONFIG },
            "ok",
            r#"fn main() { &&42.o$0 }"#,
            r#"fn main() { Ok(&&42) }"#,
        );

        check_edit_with_config(
            CompletionConfig { snippets: vec![snippet.clone()], ..TEST_CONFIG },
            "ok",
            r#"fn main() { &&42.$0 }"#,
            r#"fn main() { Ok(&&42) }"#,
        );

        check_edit_with_config(
            CompletionConfig { snippets: vec![snippet], ..TEST_CONFIG },
            "ok",
            r#"
struct A {
    a: i32,
}

fn main() {
    let a = A {a :1};
    &a.a.$0
}
            "#,
            r#"
struct A {
    a: i32,
}

fn main() {
    let a = A {a :1};
    Ok(&a.a)
}
            "#,
        );
    }

    #[test]
    fn no_postfix_completions_in_if_block_that_has_an_else() {
        check(
            r#"
fn test() {
    if true {}.$0 else {}
}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn mut_ref_consuming() {
        check_edit(
            "call",
            r#"
fn main() {
    let mut x = &mut 2;
    &mut x.$0;
}
"#,
            r#"
fn main() {
    let mut x = &mut 2;
    ${1}(&mut x);
}
"#,
        );
    }

    #[test]
    fn deref_consuming() {
        check_edit(
            "call",
            r#"
fn main() {
    let mut x = &mut 2;
    &mut *x.$0;
}
"#,
            r#"
fn main() {
    let mut x = &mut 2;
    ${1}(&mut *x);
}
"#,
        );
    }

    #[test]
    fn inside_macro() {
        check_edit(
            "box",
            r#"
macro_rules! assert {
    ( $it:expr $(,)? ) => { $it };
}

fn foo() {
    let a = true;
    assert!(if a == false { true } else { false }.$0);
}
        "#,
            r#"
macro_rules! assert {
    ( $it:expr $(,)? ) => { $it };
}

fn foo() {
    let a = true;
    assert!(Box::new(if a == false { true } else { false }));
}
        "#,
        );
    }
}
