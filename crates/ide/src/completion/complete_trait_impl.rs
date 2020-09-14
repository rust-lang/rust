//! Completion for associated items in a trait implementation.
//!
//! This module adds the completion items related to implementing associated
//! items within a `impl Trait for Struct` block. The current context node
//! must be within either a `FN`, `TYPE_ALIAS`, or `CONST` node
//! and an direct child of an `IMPL`.
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

use assists::utils::get_missing_assoc_items;
use hir::{self, HasAttrs, HasSource};
use syntax::{
    ast::{self, edit, Impl},
    AstNode, SyntaxKind, SyntaxNode, TextRange, T,
};
use text_edit::TextEdit;

use crate::{
    completion::{
        CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
    },
    display::function_declaration,
};

#[derive(Debug, PartialEq, Eq)]
enum ImplCompletionKind {
    All,
    Fn,
    TypeAlias,
    Const,
}

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some((kind, trigger, impl_def)) = completion_match(ctx) {
        get_missing_assoc_items(&ctx.sema, &impl_def).into_iter().for_each(|item| match item {
            hir::AssocItem::Function(fn_item)
                if kind == ImplCompletionKind::All || kind == ImplCompletionKind::Fn =>
            {
                add_function_impl(&trigger, acc, ctx, fn_item)
            }
            hir::AssocItem::TypeAlias(type_item)
                if kind == ImplCompletionKind::All || kind == ImplCompletionKind::TypeAlias =>
            {
                add_type_alias_impl(&trigger, acc, ctx, type_item)
            }
            hir::AssocItem::Const(const_item)
                if kind == ImplCompletionKind::All || kind == ImplCompletionKind::Const =>
            {
                add_const_impl(&trigger, acc, ctx, const_item)
            }
            _ => {}
        });
    }
}

fn completion_match(ctx: &CompletionContext) -> Option<(ImplCompletionKind, SyntaxNode, Impl)> {
    let mut token = ctx.token.clone();
    // For keywork without name like `impl .. { fn <|> }`, the current position is inside
    // the whitespace token, which is outside `FN` syntax node.
    // We need to follow the previous token in this case.
    if token.kind() == SyntaxKind::WHITESPACE {
        token = token.prev_token()?;
    }

    let (kind, trigger, impl_def_offset) = token.ancestors().find_map(|p| match p.kind() {
        // `const` can be a modifier of an item, so the `const` token may be inside another item syntax node.
        // Eg. `impl .. { const <|> fn bar() .. }`
        SyntaxKind::FN | SyntaxKind::TYPE_ALIAS | SyntaxKind::CONST
            if token.kind() == SyntaxKind::CONST_KW =>
        {
            Some((ImplCompletionKind::Const, p, 2))
        }
        SyntaxKind::FN => Some((ImplCompletionKind::Fn, p, 2)),
        SyntaxKind::TYPE_ALIAS => Some((ImplCompletionKind::TypeAlias, p, 2)),
        SyntaxKind::CONST => Some((ImplCompletionKind::Const, p, 2)),
        // `impl .. { const <|> }` is parsed as:
        // IMPL
        //   ASSOC_ITEM_LIST
        //     ERROR
        //       CONST_KW <- token
        //     WHITESPACE <- ctx.token
        SyntaxKind::ERROR
            if p.first_token().map_or(false, |t| t.kind() == SyntaxKind::CONST_KW) =>
        {
            Some((ImplCompletionKind::Const, p, 2))
        }
        SyntaxKind::NAME_REF => Some((ImplCompletionKind::All, p, 5)),
        _ => None,
    })?;

    let impl_def = (0..impl_def_offset - 1)
        .try_fold(trigger.parent()?, |t, _| t.parent())
        .and_then(ast::Impl::cast)?;
    Some((kind, trigger, impl_def))
}

fn add_function_impl(
    fn_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    func: hir::Function,
) {
    let fn_name = func.name(ctx.db).to_string();

    let label = if func.params(ctx.db).is_empty() {
        format!("fn {}()", fn_name)
    } else {
        format!("fn {}(..)", fn_name)
    };

    let builder = CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label)
        .lookup_by(fn_name)
        .set_documentation(func.docs(ctx.db));

    let completion_kind = if func.self_param(ctx.db).is_some() {
        CompletionItemKind::Method
    } else {
        CompletionItemKind::Function
    };
    let range = TextRange::new(fn_def_node.text_range().start(), ctx.source_range().end());

    let function_decl = function_declaration(&func.source(ctx.db).value);
    match ctx.config.snippet_cap {
        Some(cap) => {
            let snippet = format!("{} {{\n    $0\n}}", function_decl);
            builder.snippet_edit(cap, TextEdit::replace(range, snippet))
        }
        None => {
            let header = format!("{} {{", function_decl);
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

fn make_const_compl_syntax(const_: &ast::Const) -> String {
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
    use expect_test::{expect, Expect};

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

    #[test]
    fn complete_without_name() {
        let test = |completion: &str, hint: &str, completed: &str, next_sibling: &str| {
            println!(
                "completion='{}', hint='{}', next_sibling='{}'",
                completion, hint, next_sibling
            );

            check_edit(
                completion,
                &format!(
                    r#"
trait Test {{
    type Foo;
    const CONST: u16;
    fn bar();
}}
struct T;

impl Test for T {{
    {}
    {}
}}
"#,
                    hint, next_sibling
                ),
                &format!(
                    r#"
trait Test {{
    type Foo;
    const CONST: u16;
    fn bar();
}}
struct T;

impl Test for T {{
    {}
    {}
}}
"#,
                    completed, next_sibling
                ),
            )
        };

        // Enumerate some possible next siblings.
        for next_sibling in &[
            "",
            "fn other_fn() {}", // `const <|> fn` -> `const fn`
            "type OtherType = i32;",
            "const OTHER_CONST: i32 = 0;",
            "async fn other_fn() {}",
            "unsafe fn other_fn() {}",
            "default fn other_fn() {}",
            "default type OtherType = i32;",
            "default const OTHER_CONST: i32 = 0;",
        ] {
            test("bar", "fn <|>", "fn bar() {\n    $0\n}", next_sibling);
            test("Foo", "type <|>", "type Foo = ", next_sibling);
            test("CONST", "const <|>", "const CONST: u16 = ", next_sibling);
        }
    }
}
