//! Completion for associated items in a trait implementation.
//!
//! This module adds the completion items related to implementing associated
//! items within an `impl Trait for Struct` block. The current context node
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

use hir::{self, HasAttrs};
use ide_db::{path_transform::PathTransform, traits::get_missing_assoc_items, SymbolKind};
use syntax::{
    ast::{self, edit_in_place::AttrsOwnerEdit},
    display::function_declaration,
    AstNode, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, TextRange, T,
};
use text_edit::TextEdit;

use crate::{CompletionContext, CompletionItem, CompletionItemKind, Completions};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ImplCompletionKind {
    All,
    Fn,
    TypeAlias,
    Const,
}

pub(crate) fn complete_trait_impl(acc: &mut Completions, ctx: &CompletionContext) {
    if let Some((kind, trigger, impl_def)) = completion_match(ctx.token.clone()) {
        if let Some(hir_impl) = ctx.sema.to_def(&impl_def) {
            get_missing_assoc_items(&ctx.sema, &impl_def).into_iter().for_each(|item| {
                match (item, kind) {
                    (
                        hir::AssocItem::Function(fn_item),
                        ImplCompletionKind::All | ImplCompletionKind::Fn,
                    ) => add_function_impl(acc, ctx, &trigger, fn_item, hir_impl),
                    (
                        hir::AssocItem::TypeAlias(type_item),
                        ImplCompletionKind::All | ImplCompletionKind::TypeAlias,
                    ) => add_type_alias_impl(acc, ctx, &trigger, type_item),
                    (
                        hir::AssocItem::Const(const_item),
                        ImplCompletionKind::All | ImplCompletionKind::Const,
                    ) => add_const_impl(acc, ctx, &trigger, const_item, hir_impl),
                    _ => {}
                }
            });
        }
    }
}

fn completion_match(mut token: SyntaxToken) -> Option<(ImplCompletionKind, SyntaxNode, ast::Impl)> {
    // For keyword without name like `impl .. { fn $0 }`, the current position is inside
    // the whitespace token, which is outside `FN` syntax node.
    // We need to follow the previous token in this case.
    if token.kind() == SyntaxKind::WHITESPACE {
        token = token.prev_token()?;
    }

    let parent_kind = token.parent().map_or(SyntaxKind::EOF, |it| it.kind());
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
        SyntaxKind::IDENT if parent_kind == SyntaxKind::NAME => 1,
        // `impl .. { foo$0 }`
        // MACRO_CALL       3
        //  PATH            2
        //    PATH_SEGMENT  1
        //      NAME_REF    0
        //        IDENT     <- *
        SyntaxKind::IDENT if parent_kind == SyntaxKind::NAME_REF => 3,
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
    acc: &mut Completions,
    ctx: &CompletionContext,
    fn_def_node: &SyntaxNode,
    func: hir::Function,
    impl_def: hir::Impl,
) {
    let fn_name = func.name(ctx.db).to_smol_str();

    let label = if func.assoc_fn_params(ctx.db).is_empty() {
        format!("fn {}()", fn_name)
    } else {
        format!("fn {}(..)", fn_name)
    };

    let completion_kind = if func.self_param(ctx.db).is_some() {
        CompletionItemKind::Method
    } else {
        CompletionItemKind::SymbolKind(SymbolKind::Function)
    };

    let range = replacement_range(ctx, fn_def_node);
    let mut item = CompletionItem::new(completion_kind, range, label);
    item.lookup_by(fn_name).set_documentation(func.docs(ctx.db));

    if let Some(source) = ctx.sema.source(func) {
        let assoc_item = ast::AssocItem::Fn(source.value);
        if let Some(transformed_item) = get_transformed_assoc_item(ctx, assoc_item, impl_def) {
            let transformed_fn = match transformed_item {
                ast::AssocItem::Fn(func) => func,
                _ => unreachable!(),
            };

            let function_decl = function_declaration(&transformed_fn);
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let snippet = format!("{} {{\n    $0\n}}", function_decl);
                    item.snippet_edit(cap, TextEdit::replace(range, snippet));
                }
                None => {
                    let header = format!("{} {{", function_decl);
                    item.text_edit(TextEdit::replace(range, header));
                }
            };
            item.add_to(acc);
        }
    }
}

/// Transform a relevant associated item to inline generics from the impl, remove attrs and docs, etc.
fn get_transformed_assoc_item(
    ctx: &CompletionContext,
    assoc_item: ast::AssocItem,
    impl_def: hir::Impl,
) -> Option<ast::AssocItem> {
    let assoc_item = assoc_item.clone_for_update();
    let trait_ = impl_def.trait_(ctx.db)?;
    let source_scope = &ctx.sema.scope_for_def(trait_);
    let target_scope = &ctx.sema.scope(ctx.sema.source(impl_def)?.syntax().value);
    let transform = PathTransform::trait_impl(
        target_scope,
        source_scope,
        trait_,
        ctx.sema.source(impl_def)?.value,
    );

    transform.apply(assoc_item.syntax());
    if let ast::AssocItem::Fn(func) = &assoc_item {
        func.remove_attrs_and_docs();
    }
    Some(assoc_item)
}

fn add_type_alias_impl(
    acc: &mut Completions,
    ctx: &CompletionContext,
    type_def_node: &SyntaxNode,
    type_alias: hir::TypeAlias,
) {
    let alias_name = type_alias.name(ctx.db).to_smol_str();

    let snippet = format!("type {} = ", alias_name);

    let range = replacement_range(ctx, type_def_node);
    let mut item = CompletionItem::new(SymbolKind::TypeAlias, range, &snippet);
    item.text_edit(TextEdit::replace(range, snippet))
        .lookup_by(alias_name)
        .set_documentation(type_alias.docs(ctx.db));
    item.add_to(acc);
}

fn add_const_impl(
    acc: &mut Completions,
    ctx: &CompletionContext,
    const_def_node: &SyntaxNode,
    const_: hir::Const,
    impl_def: hir::Impl,
) {
    let const_name = const_.name(ctx.db).map(|n| n.to_smol_str());

    if let Some(const_name) = const_name {
        if let Some(source) = ctx.sema.source(const_) {
            let assoc_item = ast::AssocItem::Const(source.value);
            if let Some(transformed_item) = get_transformed_assoc_item(ctx, assoc_item, impl_def) {
                let transformed_const = match transformed_item {
                    ast::AssocItem::Const(const_) => const_,
                    _ => unreachable!(),
                };

                let snippet = make_const_compl_syntax(&transformed_const);

                let range = replacement_range(ctx, const_def_node);
                let mut item = CompletionItem::new(SymbolKind::Const, range, &snippet);
                item.text_edit(TextEdit::replace(range, snippet))
                    .lookup_by(const_name)
                    .set_documentation(const_.docs(ctx.db));
                item.add_to(acc);
            }
        }
    }
}

fn make_const_compl_syntax(const_: &ast::Const) -> String {
    const_.remove_attrs_and_docs();

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

fn replacement_range(ctx: &CompletionContext, item: &SyntaxNode) -> TextRange {
    let first_child = item
        .children_with_tokens()
        .find(|child| {
            !matches!(child.kind(), SyntaxKind::COMMENT | SyntaxKind::WHITESPACE | SyntaxKind::ATTR)
        })
        .unwrap_or_else(|| SyntaxElement::Node(item.clone()));

    TextRange::new(first_child.text_range().start(), ctx.source_range().end())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::{check_edit, completion_list_no_kw};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list_no_kw(ra_fixture);
        expect.assert_eq(&actual)
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
            expect![[r#"
                sp Self
                tt Test
                st T
                bt u32
            "#]],
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
            expect![[r#""#]],
        );

        check(
            r"
trait Test { fn test(_: i32); fn test2(); }
struct T;

impl Test for T {
    fn test(t$0)
}
",
            expect![[r#"
                sp Self
                st T
            "#]],
        );

        check(
            r"
trait Test { fn test(_: fn()); fn test2(); }
struct T;

impl Test for T {
    fn test(f: fn $0)
}
",
            expect![[r#"
                sp Self
                st T
            "#]],
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
            expect![[r#""#]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: T$0
}
",
            expect![[r#"
                sp Self
                tt Test
                st T
                bt u32
            "#]],
        );

        check(
            r"
trait Test { const TEST: u32; const TEST2: u32; type Test; fn test(); }
struct T;

impl Test for T {
    const TEST: u32 = f$0
}
",
            expect![[r#"
                sp Self
                tt Test
                st T
                bt u32
            "#]],
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
            expect![[r#"
                sp Self
                tt Test
                st T
                bt u32
            "#]],
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
            expect![[r#"
                sp Self
                tt Test
                st T
                bt u32
            "#]],
        );

        check(
            r"
trait Test { type Test; type Test2; fn test(); }
struct T;

impl Test for T {
    type Test = fn $0;
}
",
            expect![[r#"
                sp Self
                tt Test
                st T
                bt u32
            "#]],
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

    #[test]
    fn snippet_does_not_overwrite_comment_or_attr() {
        let test = |completion: &str, hint: &str, completed: &str| {
            check_edit(
                completion,
                &format!(
                    r#"
trait Foo {{
    type Type;
    fn function();
    const CONST: i32 = 0;
}}
struct T;

impl Foo for T {{
    // Comment
    #[bar]
    {}
}}
"#,
                    hint
                ),
                &format!(
                    r#"
trait Foo {{
    type Type;
    fn function();
    const CONST: i32 = 0;
}}
struct T;

impl Foo for T {{
    // Comment
    #[bar]
    {}
}}
"#,
                    completed
                ),
            )
        };
        test("function", "fn f$0", "fn function() {\n    $0\n}");
        test("Type", "type T$0", "type Type = ");
        test("CONST", "const C$0", "const CONST: i32 = ");
    }

    #[test]
    fn generics_are_inlined_in_return_type() {
        check_edit(
            "function",
            r#"
trait Foo<T> {
    fn function() -> T;
}
struct Bar;

impl Foo<u32> for Bar {
    fn f$0
}
"#,
            r#"
trait Foo<T> {
    fn function() -> T;
}
struct Bar;

impl Foo<u32> for Bar {
    fn function() -> u32 {
    $0
}
}
"#,
        )
    }

    #[test]
    fn generics_are_inlined_in_parameter() {
        check_edit(
            "function",
            r#"
trait Foo<T> {
    fn function(bar: T);
}
struct Bar;

impl Foo<u32> for Bar {
    fn f$0
}
"#,
            r#"
trait Foo<T> {
    fn function(bar: T);
}
struct Bar;

impl Foo<u32> for Bar {
    fn function(bar: u32) {
    $0
}
}
"#,
        )
    }

    #[test]
    fn generics_are_inlined_when_part_of_other_types() {
        check_edit(
            "function",
            r#"
trait Foo<T> {
    fn function(bar: Vec<T>);
}
struct Bar;

impl Foo<u32> for Bar {
    fn f$0
}
"#,
            r#"
trait Foo<T> {
    fn function(bar: Vec<T>);
}
struct Bar;

impl Foo<u32> for Bar {
    fn function(bar: Vec<u32>) {
    $0
}
}
"#,
        )
    }

    #[test]
    fn generics_are_inlined_complex() {
        check_edit(
            "function",
            r#"
trait Foo<T, U, V> {
    fn function(bar: Vec<T>, baz: U) -> Arc<Vec<V>>;
}
struct Bar;

impl Foo<u32, Vec<usize>, u8> for Bar {
    fn f$0
}
"#,
            r#"
trait Foo<T, U, V> {
    fn function(bar: Vec<T>, baz: U) -> Arc<Vec<V>>;
}
struct Bar;

impl Foo<u32, Vec<usize>, u8> for Bar {
    fn function(bar: Vec<u32>, baz: Vec<usize>) -> Arc<Vec<u8>> {
    $0
}
}
"#,
        )
    }

    #[test]
    fn generics_are_inlined_in_associated_const() {
        check_edit(
            "BAR",
            r#"
trait Foo<T> {
    const BAR: T;
}
struct Bar;

impl Foo<u32> for Bar {
    const B$0;
}
"#,
            r#"
trait Foo<T> {
    const BAR: T;
}
struct Bar;

impl Foo<u32> for Bar {
    const BAR: u32 = ;
}
"#,
        )
    }

    #[test]
    fn generics_are_inlined_in_where_clause() {
        check_edit(
            "function",
            r#"
trait SomeTrait<T> {}

trait Foo<T> {
    fn function()
    where Self: SomeTrait<T>;
}
struct Bar;

impl Foo<u32> for Bar {
    fn f$0
}
"#,
            r#"
trait SomeTrait<T> {}

trait Foo<T> {
    fn function()
    where Self: SomeTrait<T>;
}
struct Bar;

impl Foo<u32> for Bar {
    fn function()
where Self: SomeTrait<u32> {
    $0
}
}
"#,
        )
    }
}
