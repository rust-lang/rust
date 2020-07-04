//! Completion for associated items in a trait implementation.
//!
//! This module adds the completion items related to implementing associated
//! items within a `impl Trait for Struct` block. The current context node
//! must be within either a `FN_DEF`, `TYPE_ALIAS_DEF`, or `CONST_DEF` node
//! and an direct child of an `IMPL_DEF`.
//!
//! # Examples
//!
//! Considering the following trait `impl`:
//!
//! ```ignore
//! trait SomeTrait {
//!     fn foo();
//! }
//!
//! impl SomeTrait for () {
//!     fn f<|>
//! }
//! ```
//!
//! may result in the completion of the following method:
//!
//! ```ignore
//! # trait SomeTrait {
//! #    fn foo();
//! # }
//!
//! impl SomeTrait for () {
//!     fn foo() {}<|>
//! }
//! ```

use hir::{self, Docs, HasSource};
use ra_assists::utils::get_missing_assoc_items;
use ra_syntax::{
    ast::{self, edit, ImplDef},
    AstNode, SyntaxKind, SyntaxNode, TextRange, T,
};
use ra_text_edit::TextEdit;

use crate::{
    completion::{
        CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
    },
    display::FunctionSignature,
};

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some((trigger, impl_def)) = completion_match(ctx) {
        match trigger.kind() {
            SyntaxKind::NAME_REF => get_missing_assoc_items(&ctx.sema, &impl_def)
                .into_iter()
                .for_each(|item| match item {
                    hir::AssocItem::Function(fn_item) => {
                        add_function_impl(&trigger, acc, ctx, fn_item)
                    }
                    hir::AssocItem::TypeAlias(type_item) => {
                        add_type_alias_impl(&trigger, acc, ctx, type_item)
                    }
                    hir::AssocItem::Const(const_item) => {
                        add_const_impl(&trigger, acc, ctx, const_item)
                    }
                }),

            SyntaxKind::FN_DEF => {
                for missing_fn in get_missing_assoc_items(&ctx.sema, &impl_def)
                    .into_iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::Function(fn_item) => Some(fn_item),
                        _ => None,
                    })
                {
                    add_function_impl(&trigger, acc, ctx, missing_fn);
                }
            }

            SyntaxKind::TYPE_ALIAS_DEF => {
                for missing_fn in get_missing_assoc_items(&ctx.sema, &impl_def)
                    .into_iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::TypeAlias(type_item) => Some(type_item),
                        _ => None,
                    })
                {
                    add_type_alias_impl(&trigger, acc, ctx, missing_fn);
                }
            }

            SyntaxKind::CONST_DEF => {
                for missing_fn in get_missing_assoc_items(&ctx.sema, &impl_def)
                    .into_iter()
                    .filter_map(|item| match item {
                        hir::AssocItem::Const(const_item) => Some(const_item),
                        _ => None,
                    })
                {
                    add_const_impl(&trigger, acc, ctx, missing_fn);
                }
            }

            _ => {}
        }
    }
}

fn completion_match(ctx: &CompletionContext) -> Option<(SyntaxNode, ImplDef)> {
    let (trigger, impl_def_offset) = ctx.token.ancestors().find_map(|p| match p.kind() {
        SyntaxKind::FN_DEF
        | SyntaxKind::TYPE_ALIAS_DEF
        | SyntaxKind::CONST_DEF
        | SyntaxKind::BLOCK_EXPR => Some((p, 2)),
        SyntaxKind::NAME_REF => Some((p, 5)),
        _ => None,
    })?;
    let impl_def = (0..impl_def_offset - 1)
        .try_fold(trigger.parent()?, |t, _| t.parent())
        .and_then(ast::ImplDef::cast)?;
    Some((trigger, impl_def))
}

fn add_function_impl(
    fn_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    func: hir::Function,
) {
    let signature = FunctionSignature::from_hir(ctx.db, func);

    let fn_name = func.name(ctx.db).to_string();

    let label = if !func.params(ctx.db).is_empty() {
        format!("fn {}(..)", fn_name)
    } else {
        format!("fn {}()", fn_name)
    };

    let builder = CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label)
        .lookup_by(fn_name)
        .set_documentation(func.docs(ctx.db));

    let completion_kind = if func.has_self_param(ctx.db) {
        CompletionItemKind::Method
    } else {
        CompletionItemKind::Function
    };
    let range = TextRange::new(fn_def_node.text_range().start(), ctx.source_range().end());

    match ctx.config.snippet_cap {
        Some(cap) => {
            let snippet = format!("{} {{\n    $0\n}}", signature);
            builder.snippet_edit(cap, TextEdit::replace(range, snippet))
        }
        None => {
            let header = format!("{} {{", signature);
            builder.text_edit(TextEdit::replace(range, header))
        }
    }
    .kind(completion_kind)
    .add_to(acc);
}

fn add_type_alias_impl(
    type_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    type_alias: hir::TypeAlias,
) {
    let alias_name = type_alias.name(ctx.db).to_string();

    let snippet = format!("type {} = ", alias_name);

    let range = TextRange::new(type_def_node.text_range().start(), ctx.source_range().end());

    CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
        .text_edit(TextEdit::replace(range, snippet))
        .lookup_by(alias_name)
        .kind(CompletionItemKind::TypeAlias)
        .set_documentation(type_alias.docs(ctx.db))
        .add_to(acc);
}

fn add_const_impl(
    const_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    const_: hir::Const,
) {
    let const_name = const_.name(ctx.db).map(|n| n.to_string());

    if let Some(const_name) = const_name {
        let snippet = make_const_compl_syntax(&const_.source(ctx.db).value);

        let range = TextRange::new(const_def_node.text_range().start(), ctx.source_range().end());

        CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
            .text_edit(TextEdit::replace(range, snippet))
            .lookup_by(const_name)
            .kind(CompletionItemKind::Const)
            .set_documentation(const_.docs(ctx.db))
            .add_to(acc);
    }
}

fn make_const_compl_syntax(const_: &ast::ConstDef) -> String {
    let const_ = edit::remove_attrs_and_docs(const_);

    let const_start = const_.syntax().text_range().start();
    let const_end = const_.syntax().text_range().end();

    let start =
        const_.syntax().first_child_or_token().map_or(const_start, |f| f.text_range().start());

    let end = const_
        .syntax()
        .children_with_tokens()
        .find(|s| s.kind() == T![;] || s.kind() == T![=])
        .map_or(const_end, |f| f.text_range().start());

    let len = end - start;
    let range = TextRange::new(0.into(), len);

    let syntax = const_.syntax().text().slice(range).to_string();

    format!("{} = ", syntax.trim_end())
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};

    use crate::completion::{
        test_utils::{check_edit, completion_list},
        CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Magic);
        expect.assert_eq(&actual)
    }

    #[test]
    fn name_ref_function_type_const() {
        check(
            r#"
trait Test {
    type TestType;
    const TEST_CONST: u16;
    fn test();
}
struct T;

impl Test for T {
    t<|>
}
"#,
            expect![["
ct const TEST_CONST: u16 = \n\
fn fn test()
ta type TestType = \n\
            "]],
        );
    }

    #[test]
    fn no_nested_fn_completions() {
        check(
            r"
trait Test {
    fn test();
    fn test2();
}
struct T;

impl Test for T {
    fn test() {
        t<|>
    }
}
",
            expect![[""]],
        );
    }

    #[test]
    fn name_ref_single_function() {
        check_edit(
            "test",
            r#"
trait Test {
    fn test();
}
struct T;

impl Test for T {
    t<|>
}
"#,
            r#"
trait Test {
    fn test();
}
struct T;

impl Test for T {
    fn test() {
    $0
}
}
"#,
        );
    }

    #[test]
    fn single_function() {
        check_edit(
            "test",
            r#"
trait Test {
    fn test();
}
struct T;

impl Test for T {
    fn t<|>
}
"#,
            r#"
trait Test {
    fn test();
}
struct T;

impl Test for T {
    fn test() {
    $0
}
}
"#,
        );
    }

    #[test]
    fn hide_implemented_fn() {
        check(
            r#"
trait Test {
    fn foo();
    fn foo_bar();
}
struct T;

impl Test for T {
    fn foo() {}
    fn f<|>
}
"#,
            expect![[r#"
                fn fn foo_bar()
            "#]],
        );
    }

    #[test]
    fn generic_fn() {
        check_edit(
            "foo",
            r#"
trait Test {
    fn foo<T>();
}
struct T;

impl Test for T {
    fn f<|>
}
"#,
            r#"
trait Test {
    fn foo<T>();
}
struct T;

impl Test for T {
    fn foo<T>() {
    $0
}
}
"#,
        );
        check_edit(
            "foo",
            r#"
trait Test {
    fn foo<T>() where T: Into<String>;
}
struct T;

impl Test for T {
    fn f<|>
}
"#,
            r#"
trait Test {
    fn foo<T>() where T: Into<String>;
}
struct T;

impl Test for T {
    fn foo<T>()
where T: Into<String> {
    $0
}
}
"#,
        );
    }

    #[test]
    fn associated_type() {
        check_edit(
            "SomeType",
            r#"
trait Test {
    type SomeType;
}

impl Test for () {
    type S<|>
}
"#,
            "
trait Test {
    type SomeType;
}

impl Test for () {
    type SomeType = \n\
}
",
        );
    }

    #[test]
    fn associated_const() {
        check_edit(
            "SOME_CONST",
            r#"
trait Test {
    const SOME_CONST: u16;
}

impl Test for () {
    const S<|>
}
"#,
            "
trait Test {
    const SOME_CONST: u16;
}

impl Test for () {
    const SOME_CONST: u16 = \n\
}
",
        );

        check_edit(
            "SOME_CONST",
            r#"
trait Test {
    const SOME_CONST: u16 = 92;
}

impl Test for () {
    const S<|>
}
"#,
            "
trait Test {
    const SOME_CONST: u16 = 92;
}

impl Test for () {
    const SOME_CONST: u16 = \n\
}
",
        );
    }
}
