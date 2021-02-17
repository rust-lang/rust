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
//!     fn f$0
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
//!     fn foo() {}$0
//! }
//! ```

use hir::{self, HasAttrs, HasSource};
use ide_db::{traits::get_missing_assoc_items, SymbolKind};
use syntax::{
    ast::{self, edit, Impl},
    display::function_declaration,
    AstNode, SyntaxKind, SyntaxNode, TextRange, T,
};
use text_edit::TextEdit;

use crate::{
    CompletionContext,
    CompletionItem,
    CompletionItemKind,
    CompletionKind,
    Completions,
    // display::function_declaration,
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
    // For keywork without name like `impl .. { fn $0 }`, the current position is inside
    // the whitespace token, which is outside `FN` syntax node.
    // We need to follow the previous token in this case.
    if token.kind() == SyntaxKind::WHITESPACE {
        token = token.prev_token()?;
    }

    let impl_item_offset = match token.kind() {
        // `impl .. { const $0 }`
        // ERROR      0
        //   CONST_KW <- *
        T![const] => 0,
        // `impl .. { fn/type $0 }`
        // FN/TYPE_ALIAS  0
        //   FN_KW        <- *
        T![fn] | T![type] => 0,
        // `impl .. { fn/type/const foo$0 }`
        // FN/TYPE_ALIAS/CONST  1
        //  NAME                0
        //    IDENT             <- *
        SyntaxKind::IDENT if token.parent().kind() == SyntaxKind::NAME => 1,
        // `impl .. { foo$0 }`
        // MACRO_CALL       3
        //  PATH            2
        //    PATH_SEGMENT  1
        //      NAME_REF    0
        //        IDENT     <- *
        SyntaxKind::IDENT if token.parent().kind() == SyntaxKind::NAME_REF => 3,
        _ => return None,
    };

    let impl_item = token.ancestors().nth(impl_item_offset)?;
    // Must directly belong to an impl block.
    // IMPL
    //   ASSOC_ITEM_LIST
    //     <item>
    let impl_def = ast::Impl::cast(impl_item.parent()?.parent()?)?;
    let kind = match impl_item.kind() {
        // `impl ... { const $0 fn/type/const }`
        _ if token.kind() == T![const] => ImplCompletionKind::Const,
        SyntaxKind::CONST | SyntaxKind::ERROR => ImplCompletionKind::Const,
        SyntaxKind::TYPE_ALIAS => ImplCompletionKind::TypeAlias,
        SyntaxKind::FN => ImplCompletionKind::Fn,
        SyntaxKind::MACRO_CALL => ImplCompletionKind::All,
        _ => return None,
    };
    Some((kind, impl_item, impl_def))
}

fn add_function_impl(
    fn_def_node: &SyntaxNode,
    acc: &mut Completions,
    ctx: &CompletionContext,
    func: hir::Function,
) {
    let fn_name = func.name(ctx.db).to_string();

    let label = if func.assoc_fn_params(ctx.db).is_empty() {
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
        CompletionItemKind::SymbolKind(SymbolKind::Function)
    };
    let range = TextRange::new(fn_def_node.text_range().start(), ctx.source_range().end());

    if let Some(src) = func.source(ctx.db) {
        let function_decl = function_declaration(&src.value);
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
        .kind(SymbolKind::TypeAlias)
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
        if let Some(source) = const_.source(ctx.db) {
            let snippet = make_const_compl_syntax(&source.value);

            let range =
                TextRange::new(const_def_node.text_range().start(), ctx.source_range().end());

            CompletionItem::new(CompletionKind::Magic, ctx.source_range(), snippet.clone())
                .text_edit(TextEdit::replace(range, snippet))
                .lookup_by(const_name)
                .kind(SymbolKind::Const)
                .set_documentation(const_.docs(ctx.db))
                .add_to(acc);
        }
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

    use crate::{
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
    t$0
}
"#,
            expect![["
ta type TestType = \n\
ct const TEST_CONST: u16 = \n\
fn fn test()
"]],
        );
    }

    #[test]
    fn no_completion_inside_fn() {
        check(
            r"
trait Test { fn test(); fn test2(); }
struct T;

impl Test for T {
    fn test() {
        t$0
    }
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { fn test(); fn test2(); }
struct T;

impl Test for T {
    fn test() {
        fn t$0
    }
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { fn test(); fn test2(); }
struct T;

impl Test for T {
    fn test() {
        fn $0
    }
}
",
            expect![[""]],
        );

        // https://github.com/rust-analyzer/rust-analyzer/pull/5976#issuecomment-692332191
        check(
            r"
trait Test { fn test(); fn test2(); }
struct T;

impl Test for T {
    fn test() {
        foo.$0
    }
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { fn test(_: i32); fn test2(); }
struct T;

impl Test for T {
    fn test(t$0)
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { fn test(_: fn()); fn test2(); }
struct T;

impl Test for T {
    fn test(f: fn $0)
}
",
            expect![[""]],
        );
    }

    #[test]
    fn no_completion_inside_const() {
        check(
            r"
trait Test { const TEST: fn(); const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: fn $0
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: T$0
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: u32 = f$0
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: u32 = {
        t$0
    };
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: u32 = {
        fn $0
    };
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: u32 = {
        fn t$0
    };
}
",
            expect![[""]],
        );
    }

    #[test]
    fn no_completion_inside_type() {
        check(
            r"
trait Test { type Test; type Test2; fn test(); }
struct T;

impl Test for T {
    type Test = T$0;
}
",
            expect![[""]],
        );

        check(
            r"
trait Test { type Test; type Test2; fn test(); }
struct T;

impl Test for T {
    type Test = fn $0;
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
    t$0
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
    fn t$0
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
    fn f$0
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
    fn f$0
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
    fn f$0
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
    type S$0
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
    const S$0
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
    const S$0
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
            "fn other_fn() {}", // `const $0 fn` -> `const fn`
            "type OtherType = i32;",
            "const OTHER_CONST: i32 = 0;",
            "async fn other_fn() {}",
            "unsafe fn other_fn() {}",
            "default fn other_fn() {}",
            "default type OtherType = i32;",
            "default const OTHER_CONST: i32 = 0;",
        ] {
            test("bar", "fn $0", "fn bar() {\n    $0\n}", next_sibling);
            test("Foo", "type $0", "type Foo = ", next_sibling);
            test("CONST", "const $0", "const CONST: u16 = ", next_sibling);
        }
    }
}
