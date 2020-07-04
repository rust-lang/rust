//! Completion of names from the current scope, e.g. locals and imported items.

use hir::ScopeDef;
use test_utils::mark;

use crate::completion::{CompletionContext, Completions};
use hir::{Adt, ModuleDef, Type};
use ra_syntax::AstNode;

pub(super) fn complete_unqualified_path(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_trivial_path || ctx.is_pat_binding_or_const) {
        return;
    }
    if ctx.record_lit_syntax.is_some()
        || ctx.record_pat_syntax.is_some()
        || ctx.attribute_under_caret.is_some()
    {
        return;
    }

    if let Some(ty) = &ctx.expected_type {
        complete_enum_variants(acc, ctx, ty);
    }

    if ctx.is_pat_binding_or_const {
        return;
    }

    ctx.scope().process_all_names(&mut |name, res| {
        if ctx.use_item_syntax.is_some() {
            if let (ScopeDef::Unknown, Some(name_ref)) = (&res, &ctx.name_ref_syntax) {
                if name_ref.syntax().text() == name.to_string().as_str() {
                    mark::hit!(self_fulfilling_completion);
                    return;
                }
            }
        }
        acc.add_resolution(ctx, name.to_string(), &res)
    });
}

fn complete_enum_variants(acc: &mut Completions, ctx: &CompletionContext, ty: &Type) {
    if let Some(Adt::Enum(enum_data)) = ty.as_adt() {
        let variants = enum_data.variants(ctx.db);

        let module = if let Some(module) = ctx.scope().module() {
            // Compute path from the completion site if available.
            module
        } else {
            // Otherwise fall back to the enum's definition site.
            enum_data.module(ctx.db)
        };

        for variant in variants {
            if let Some(path) = module.find_use_path(ctx.db, ModuleDef::from(variant)) {
                // Variants with trivial paths are already added by the existing completion logic,
                // so we should avoid adding these twice
                if path.segments.len() > 1 {
                    acc.add_qualified_enum_variant(ctx, variant, path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use test_utils::mark;

    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};

    fn do_reference_completion(ra_fixture: &str) -> Vec<CompletionItem> {
        do_completion(ra_fixture, CompletionKind::Reference)
    }

    #[test]
    fn self_fulfilling_completion() {
        mark::check!(self_fulfilling_completion);
        assert_debug_snapshot!(
            do_reference_completion(
                r#"
                use foo<|>
                use std::collections;
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "collections",
                source_range: 4..7,
                delete: 4..7,
                insert: "collections",
            },
        ]
        "###
        );
    }

    #[test]
    fn bind_pat_and_path_ignore_at() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Enum {
                    A,
                    B,
                }
                fn quux(x: Option<Enum>) {
                    match x {
                        None => (),
                        Some(en<|> @ Enum::A) => (),
                    }
                }
                "
            ),
            @"[]"
        );
    }

    #[test]
    fn bind_pat_and_path_ignore_ref() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Enum {
                    A,
                    B,
                }
                fn quux(x: Option<Enum>) {
                    match x {
                        None => (),
                        Some(ref en<|>) => (),
                    }
                }
                "
            ),
            @r###"[]"###
        );
    }

    #[test]
    fn bind_pat_and_path() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Enum {
                    A,
                    B,
                }
                fn quux(x: Option<Enum>) {
                    match x {
                        None => (),
                        Some(En<|>) => (),
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Enum",
                source_range: 102..104,
                delete: 102..104,
                insert: "Enum",
                kind: Enum,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_bindings_from_let() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn quux(x: i32) {
                    let y = 92;
                    1 + <|>;
                    let z = ();
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "quux(…)",
                source_range: 42..42,
                delete: 42..42,
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: 42..42,
                delete: 42..42,
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: 42..42,
                delete: 42..42,
                insert: "y",
                kind: Binding,
                detail: "i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_bindings_from_if_let() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn quux() {
                    if let Some(x) = foo() {
                        let y = 92;
                    };
                    if let Some(a) = bar() {
                        let b = 62;
                        1 + <|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "a",
                source_range: 129..129,
                delete: 129..129,
                insert: "a",
                kind: Binding,
            },
            CompletionItem {
                label: "b",
                source_range: 129..129,
                delete: 129..129,
                insert: "b",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "quux()",
                source_range: 129..129,
                delete: 129..129,
                insert: "quux()$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_bindings_from_for() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn quux() {
                    for x in &[1, 2, 3] {
                        <|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "quux()",
                source_range: 46..46,
                delete: 46..46,
                insert: "quux()$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux()",
            },
            CompletionItem {
                label: "x",
                source_range: 46..46,
                delete: 46..46,
                insert: "x",
                kind: Binding,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_bindings_from_for_with_in_prefix() {
        mark::check!(completes_bindings_from_for_with_in_prefix);
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn test() {
                    for index in &[1, 2, 3] {
                        let t = in<|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "index",
                source_range: 58..58,
                delete: 58..58,
                insert: "index",
                kind: Binding,
            },
            CompletionItem {
                label: "test()",
                source_range: 58..58,
                delete: 58..58,
                insert: "test()$0",
                kind: Function,
                lookup: "test",
                detail: "fn test()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_generic_params() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn quux<T>() {
                    <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "T",
                source_range: 19..19,
                delete: 19..19,
                insert: "T",
                kind: TypeParam,
            },
            CompletionItem {
                label: "quux()",
                source_range: 19..19,
                delete: 19..19,
                insert: "quux()$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux<T>()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_generic_params_in_struct() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct X<T> {
                    x: <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Self",
                source_range: 21..21,
                delete: 21..21,
                insert: "Self",
                kind: TypeParam,
            },
            CompletionItem {
                label: "T",
                source_range: 21..21,
                delete: 21..21,
                insert: "T",
                kind: TypeParam,
            },
            CompletionItem {
                label: "X<…>",
                source_range: 21..21,
                delete: 21..21,
                insert: "X<$0>",
                kind: Struct,
                lookup: "X",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_self_in_enum() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum X {
                    Y(<|>)
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Self",
                source_range: 15..15,
                delete: 15..15,
                insert: "Self",
                kind: TypeParam,
            },
            CompletionItem {
                label: "X",
                source_range: 15..15,
                delete: 15..15,
                insert: "X",
                kind: Enum,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_module_items() {
        assert_debug_snapshot!(
        do_reference_completion(
            r"
                struct Foo;
                enum Baz {}
                fn quux() {
                    <|>
                }
                "
        ),
        @r###"
        [
            CompletionItem {
                label: "Baz",
                source_range: 40..40,
                delete: 40..40,
                insert: "Baz",
                kind: Enum,
            },
            CompletionItem {
                label: "Foo",
                source_range: 40..40,
                delete: 40..40,
                insert: "Foo",
                kind: Struct,
            },
            CompletionItem {
                label: "quux()",
                source_range: 40..40,
                delete: 40..40,
                insert: "quux()$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux()",
            },
        ]
        "###
            );
    }

    #[test]
    fn completes_extern_prelude() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                //- /lib.rs
                use <|>;

                //- /other_crate/lib.rs
                // nothing here
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "other_crate",
                source_range: 4..4,
                delete: 4..4,
                insert: "other_crate",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_module_items_in_nested_modules() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct Foo;
                mod m {
                    struct Bar;
                    fn quux() { <|> }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Bar",
                source_range: 52..52,
                delete: 52..52,
                insert: "Bar",
                kind: Struct,
            },
            CompletionItem {
                label: "quux()",
                source_range: 52..52,
                delete: 52..52,
                insert: "quux()$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_return_type() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct Foo;
                fn x() -> <|>
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 22..22,
                delete: 22..22,
                insert: "Foo",
                kind: Struct,
            },
            CompletionItem {
                label: "x()",
                source_range: 22..22,
                delete: 22..22,
                insert: "x()$0",
                kind: Function,
                lookup: "x",
                detail: "fn x()",
            },
        ]
        "###
        );
    }

    #[test]
    fn dont_show_both_completions_for_shadowing() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn foo() {
                    let bar = 92;
                    {
                        let bar = 62;
                        <|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "bar",
                source_range: 65..65,
                delete: 65..65,
                insert: "bar",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "foo()",
                source_range: 65..65,
                delete: 65..65,
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_self_in_methods() {
        assert_debug_snapshot!(
            do_reference_completion(r"impl S { fn foo(&self) { <|> } }"),
            @r###"
        [
            CompletionItem {
                label: "Self",
                source_range: 25..25,
                delete: 25..25,
                insert: "Self",
                kind: TypeParam,
            },
            CompletionItem {
                label: "self",
                source_range: 25..25,
                delete: 25..25,
                insert: "self",
                kind: Binding,
                detail: "&{unknown}",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_prelude() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                fn foo() { let x: <|> }

                //- /std/lib.rs
                #[prelude_import]
                use prelude::*;

                mod prelude {
                    struct Option;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Option",
                source_range: 18..18,
                delete: 18..18,
                insert: "Option",
                kind: Struct,
            },
            CompletionItem {
                label: "foo()",
                source_range: 18..18,
                delete: 18..18,
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
            CompletionItem {
                label: "std",
                source_range: 18..18,
                delete: 18..18,
                insert: "std",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_std_prelude_if_core_is_defined() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                fn foo() { let x: <|> }

                //- /core/lib.rs
                #[prelude_import]
                use prelude::*;

                mod prelude {
                    struct Option;
                }

                //- /std/lib.rs
                #[prelude_import]
                use prelude::*;

                mod prelude {
                    struct String;
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "String",
                source_range: 18..18,
                delete: 18..18,
                insert: "String",
                kind: Struct,
            },
            CompletionItem {
                label: "core",
                source_range: 18..18,
                delete: 18..18,
                insert: "core",
                kind: Module,
            },
            CompletionItem {
                label: "foo()",
                source_range: 18..18,
                delete: 18..18,
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
            CompletionItem {
                label: "std",
                source_range: 18..18,
                delete: 18..18,
                insert: "std",
                kind: Module,
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_macros_as_value() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                macro_rules! foo {
                    () => {}
                }

                #[macro_use]
                mod m1 {
                    macro_rules! bar {
                        () => {}
                    }
                }

                mod m2 {
                    macro_rules! nope {
                        () => {}
                    }

                    #[macro_export]
                    macro_rules! baz {
                        () => {}
                    }
                }

                fn main() {
                    let v = <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "bar!(…)",
                source_range: 256..256,
                delete: 256..256,
                insert: "bar!($0)",
                kind: Macro,
                lookup: "bar!",
                detail: "macro_rules! bar",
            },
            CompletionItem {
                label: "baz!(…)",
                source_range: 256..256,
                delete: 256..256,
                insert: "baz!($0)",
                kind: Macro,
                lookup: "baz!",
                detail: "#[macro_export]\nmacro_rules! baz",
            },
            CompletionItem {
                label: "foo!(…)",
                source_range: 256..256,
                delete: 256..256,
                insert: "foo!($0)",
                kind: Macro,
                lookup: "foo!",
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "m1",
                source_range: 256..256,
                delete: 256..256,
                insert: "m1",
                kind: Module,
            },
            CompletionItem {
                label: "m2",
                source_range: 256..256,
                delete: 256..256,
                insert: "m2",
                kind: Module,
            },
            CompletionItem {
                label: "main()",
                source_range: 256..256,
                delete: 256..256,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_both_macro_and_value() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                macro_rules! foo {
                    () => {}
                }

                fn foo() {
                    <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo!(…)",
                source_range: 50..50,
                delete: 50..50,
                insert: "foo!($0)",
                kind: Macro,
                lookup: "foo!",
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "foo()",
                source_range: 50..50,
                delete: 50..50,
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_macros_as_type() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                macro_rules! foo {
                    () => {}
                }

                fn main() {
                    let x: <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo!(…)",
                source_range: 58..58,
                delete: 58..58,
                insert: "foo!($0)",
                kind: Macro,
                lookup: "foo!",
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "main()",
                source_range: 58..58,
                delete: 58..58,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_macros_as_stmt() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                macro_rules! foo {
                    () => {}
                }

                fn main() {
                    <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo!(…)",
                source_range: 51..51,
                delete: 51..51,
                insert: "foo!($0)",
                kind: Macro,
                lookup: "foo!",
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "main()",
                source_range: 51..51,
                delete: 51..51,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_local_item() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                fn main() {
                    return f<|>;
                    fn frobnicate() {}
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "frobnicate()",
                source_range: 23..24,
                delete: 23..24,
                insert: "frobnicate()$0",
                kind: Function,
                lookup: "frobnicate",
                detail: "fn frobnicate()",
            },
            CompletionItem {
                label: "main()",
                source_range: 23..24,
                delete: 23..24,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        )
    }

    #[test]
    fn completes_in_simple_macro_1() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                macro_rules! m { ($e:expr) => { $e } }
                fn quux(x: i32) {
                    let y = 92;
                    m!(<|>);
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m!(…)",
                source_range: 80..80,
                delete: 80..80,
                insert: "m!($0)",
                kind: Macro,
                lookup: "m!",
                detail: "macro_rules! m",
            },
            CompletionItem {
                label: "quux(…)",
                source_range: 80..80,
                delete: 80..80,
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: 80..80,
                delete: 80..80,
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: 80..80,
                delete: 80..80,
                insert: "y",
                kind: Binding,
                detail: "i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_in_simple_macro_2() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                macro_rules! m { ($e:expr) => { $e } }
                fn quux(x: i32) {
                    let y = 92;
                    m!(x<|>);
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "m!(…)",
                source_range: 80..81,
                delete: 80..81,
                insert: "m!($0)",
                kind: Macro,
                lookup: "m!",
                detail: "macro_rules! m",
            },
            CompletionItem {
                label: "quux(…)",
                source_range: 80..81,
                delete: 80..81,
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: 80..81,
                delete: 80..81,
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: 80..81,
                delete: 80..81,
                insert: "y",
                kind: Binding,
                detail: "i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_in_simple_macro_without_closing_parens() {
        assert_debug_snapshot!(
                    do_reference_completion(
                        r"
                macro_rules! m { ($e:expr) => { $e } }
                fn quux(x: i32) {
                    let y = 92;
                    m!(x<|>
                }
                "
                    ),
                    @r###"
        [
            CompletionItem {
                label: "m!(…)",
                source_range: 80..81,
                delete: 80..81,
                insert: "m!($0)",
                kind: Macro,
                lookup: "m!",
                detail: "macro_rules! m",
            },
            CompletionItem {
                label: "quux(…)",
                source_range: 80..81,
                delete: 80..81,
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: 80..81,
                delete: 80..81,
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: 80..81,
                delete: 80..81,
                insert: "y",
                kind: Binding,
                detail: "i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn completes_unresolved_uses() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                use spam::Quux;

                fn main() {
                    <|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Quux",
                source_range: 33..33,
                delete: 33..33,
                insert: "Quux",
            },
            CompletionItem {
                label: "main()",
                source_range: 33..33,
                delete: 33..33,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }
    #[test]
    fn completes_enum_variant_matcharm() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Foo {
                    Bar,
                    Baz,
                    Quux
                }

                fn main() {
                    let foo = Foo::Quux;

                    match foo {
                        Qu<|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 103..105,
                delete: 103..105,
                insert: "Foo",
                kind: Enum,
            },
            CompletionItem {
                label: "Foo::Bar",
                source_range: 103..105,
                delete: 103..105,
                insert: "Foo::Bar",
                kind: EnumVariant,
                lookup: "Bar",
                detail: "()",
            },
            CompletionItem {
                label: "Foo::Baz",
                source_range: 103..105,
                delete: 103..105,
                insert: "Foo::Baz",
                kind: EnumVariant,
                lookup: "Baz",
                detail: "()",
            },
            CompletionItem {
                label: "Foo::Quux",
                source_range: 103..105,
                delete: 103..105,
                insert: "Foo::Quux",
                kind: EnumVariant,
                lookup: "Quux",
                detail: "()",
            },
        ]
        "###
        )
    }

    #[test]
    fn completes_enum_variant_iflet() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Foo {
                    Bar,
                    Baz,
                    Quux
                }

                fn main() {
                    let foo = Foo::Quux;

                    if let Qu<|> = foo {

                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 90..92,
                delete: 90..92,
                insert: "Foo",
                kind: Enum,
            },
            CompletionItem {
                label: "Foo::Bar",
                source_range: 90..92,
                delete: 90..92,
                insert: "Foo::Bar",
                kind: EnumVariant,
                lookup: "Bar",
                detail: "()",
            },
            CompletionItem {
                label: "Foo::Baz",
                source_range: 90..92,
                delete: 90..92,
                insert: "Foo::Baz",
                kind: EnumVariant,
                lookup: "Baz",
                detail: "()",
            },
            CompletionItem {
                label: "Foo::Quux",
                source_range: 90..92,
                delete: 90..92,
                insert: "Foo::Quux",
                kind: EnumVariant,
                lookup: "Quux",
                detail: "()",
            },
        ]
        "###
        )
    }

    #[test]
    fn completes_enum_variant_basic_expr() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Foo {
                    Bar,
                    Baz,
                    Quux
                }

                fn main() {
                    let foo: Foo = Q<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 72..73,
                delete: 72..73,
                insert: "Foo",
                kind: Enum,
            },
            CompletionItem {
                label: "Foo::Bar",
                source_range: 72..73,
                delete: 72..73,
                insert: "Foo::Bar",
                kind: EnumVariant,
                lookup: "Bar",
                detail: "()",
            },
            CompletionItem {
                label: "Foo::Baz",
                source_range: 72..73,
                delete: 72..73,
                insert: "Foo::Baz",
                kind: EnumVariant,
                lookup: "Baz",
                detail: "()",
            },
            CompletionItem {
                label: "Foo::Quux",
                source_range: 72..73,
                delete: 72..73,
                insert: "Foo::Quux",
                kind: EnumVariant,
                lookup: "Quux",
                detail: "()",
            },
            CompletionItem {
                label: "main()",
                source_range: 72..73,
                delete: 72..73,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        )
    }

    #[test]
    fn completes_enum_variant_from_module() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                mod m { pub enum E { V } }

                fn f() -> m::E {
                    V<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "f()",
                source_range: 49..50,
                delete: 49..50,
                insert: "f()$0",
                kind: Function,
                lookup: "f",
                detail: "fn f() -> m::E",
            },
            CompletionItem {
                label: "m",
                source_range: 49..50,
                delete: 49..50,
                insert: "m",
                kind: Module,
            },
            CompletionItem {
                label: "m::E::V",
                source_range: 49..50,
                delete: 49..50,
                insert: "m::E::V",
                kind: EnumVariant,
                lookup: "V",
                detail: "()",
            },
        ]
        "###
        )
    }

    #[test]
    fn dont_complete_attr() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct Foo;
                #[<|>]
                fn f() {}
                "
            ),
            @r###"[]"###
        )
    }
}
