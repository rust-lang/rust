//! Completion of names from the current scope, e.g. locals and imported items.

use crate::completion::{CompletionContext, Completions};
use hir::{ModuleDef, ScopeDef};

pub(super) fn complete_scope(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_trivial_path && !ctx.is_pat_binding_and_path {
        return;
    }

    ctx.scope().process_all_names(&mut |name, res| match (ctx.is_pat_binding_and_path, &res) {
        (true, ScopeDef::ModuleDef(ModuleDef::Function(..))) => (),
        (true, ScopeDef::ModuleDef(ModuleDef::Static(..))) => (),
        (true, ScopeDef::Local(..)) => (),
        _ => acc.add_resolution(ctx, name.to_string(), &res),
    });
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::completion::{test_utils::do_completion, CompletionItem, CompletionKind};

    fn do_reference_completion(ra_fixture: &str) -> Vec<CompletionItem> {
        do_completion(ra_fixture, CompletionKind::Reference)
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
            @r###"[]"###
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
                    source_range: [231; 233),
                    delete: [231; 233),
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
                source_range: [91; 91),
                delete: [91; 91),
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: [91; 91),
                delete: [91; 91),
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: [91; 91),
                delete: [91; 91),
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
                source_range: [242; 242),
                delete: [242; 242),
                insert: "a",
                kind: Binding,
            },
            CompletionItem {
                label: "b",
                source_range: [242; 242),
                delete: [242; 242),
                insert: "b",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "quux()",
                source_range: [242; 242),
                delete: [242; 242),
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
                source_range: [95; 95),
                delete: [95; 95),
                insert: "quux()$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux()",
            },
            CompletionItem {
                label: "x",
                source_range: [95; 95),
                delete: [95; 95),
                insert: "x",
                kind: Binding,
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
                source_range: [52; 52),
                delete: [52; 52),
                insert: "T",
                kind: TypeParam,
            },
            CompletionItem {
                label: "quux()",
                source_range: [52; 52),
                delete: [52; 52),
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
                source_range: [54; 54),
                delete: [54; 54),
                insert: "Self",
                kind: TypeParam,
            },
            CompletionItem {
                label: "T",
                source_range: [54; 54),
                delete: [54; 54),
                insert: "T",
                kind: TypeParam,
            },
            CompletionItem {
                label: "X<…>",
                source_range: [54; 54),
                delete: [54; 54),
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
                source_range: [48; 48),
                delete: [48; 48),
                insert: "Self",
                kind: TypeParam,
            },
            CompletionItem {
                label: "X",
                source_range: [48; 48),
                delete: [48; 48),
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
                source_range: [105; 105),
                delete: [105; 105),
                insert: "Baz",
                kind: Enum,
            },
            CompletionItem {
                label: "Foo",
                source_range: [105; 105),
                delete: [105; 105),
                insert: "Foo",
                kind: Struct,
            },
            CompletionItem {
                label: "quux()",
                source_range: [105; 105),
                delete: [105; 105),
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
                source_range: [4; 4),
                delete: [4; 4),
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
                source_range: [117; 117),
                delete: [117; 117),
                insert: "Bar",
                kind: Struct,
            },
            CompletionItem {
                label: "quux()",
                source_range: [117; 117),
                delete: [117; 117),
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
                source_range: [55; 55),
                delete: [55; 55),
                insert: "Foo",
                kind: Struct,
            },
            CompletionItem {
                label: "x()",
                source_range: [55; 55),
                delete: [55; 55),
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
                source_range: [146; 146),
                delete: [146; 146),
                insert: "bar",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "foo()",
                source_range: [146; 146),
                delete: [146; 146),
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
                source_range: [25; 25),
                delete: [25; 25),
                insert: "Self",
                kind: TypeParam,
            },
            CompletionItem {
                label: "self",
                source_range: [25; 25),
                delete: [25; 25),
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
                source_range: [18; 18),
                delete: [18; 18),
                insert: "Option",
                kind: Struct,
            },
            CompletionItem {
                label: "foo()",
                source_range: [18; 18),
                delete: [18; 18),
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
            CompletionItem {
                label: "std",
                source_range: [18; 18),
                delete: [18; 18),
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
                source_range: [18; 18),
                delete: [18; 18),
                insert: "String",
                kind: Struct,
            },
            CompletionItem {
                label: "core",
                source_range: [18; 18),
                delete: [18; 18),
                insert: "core",
                kind: Module,
            },
            CompletionItem {
                label: "foo()",
                source_range: [18; 18),
                delete: [18; 18),
                insert: "foo()$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo()",
            },
            CompletionItem {
                label: "std",
                source_range: [18; 18),
                delete: [18; 18),
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
                label: "bar!",
                source_range: [252; 252),
                delete: [252; 252),
                insert: "bar!($0)",
                kind: Macro,
                detail: "macro_rules! bar",
            },
            CompletionItem {
                label: "baz!",
                source_range: [252; 252),
                delete: [252; 252),
                insert: "baz!($0)",
                kind: Macro,
                detail: "#[macro_export]\nmacro_rules! baz",
            },
            CompletionItem {
                label: "foo!",
                source_range: [252; 252),
                delete: [252; 252),
                insert: "foo!($0)",
                kind: Macro,
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "m1",
                source_range: [252; 252),
                delete: [252; 252),
                insert: "m1",
                kind: Module,
            },
            CompletionItem {
                label: "m2",
                source_range: [252; 252),
                delete: [252; 252),
                insert: "m2",
                kind: Module,
            },
            CompletionItem {
                label: "main()",
                source_range: [252; 252),
                delete: [252; 252),
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
                label: "foo!",
                source_range: [49; 49),
                delete: [49; 49),
                insert: "foo!($0)",
                kind: Macro,
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "foo()",
                source_range: [49; 49),
                delete: [49; 49),
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
                label: "foo!",
                source_range: [57; 57),
                delete: [57; 57),
                insert: "foo!($0)",
                kind: Macro,
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "main()",
                source_range: [57; 57),
                delete: [57; 57),
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
                label: "foo!",
                source_range: [50; 50),
                delete: [50; 50),
                insert: "foo!($0)",
                kind: Macro,
                detail: "macro_rules! foo",
            },
            CompletionItem {
                label: "main()",
                source_range: [50; 50),
                delete: [50; 50),
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
                source_range: [23; 24),
                delete: [23; 24),
                insert: "frobnicate()$0",
                kind: Function,
                lookup: "frobnicate",
                detail: "fn frobnicate()",
            },
            CompletionItem {
                label: "main()",
                source_range: [23; 24),
                delete: [23; 24),
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
                label: "m!",
                source_range: [145; 145),
                delete: [145; 145),
                insert: "m!($0)",
                kind: Macro,
                detail: "macro_rules! m",
            },
            CompletionItem {
                label: "quux(…)",
                source_range: [145; 145),
                delete: [145; 145),
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: [145; 145),
                delete: [145; 145),
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: [145; 145),
                delete: [145; 145),
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
                label: "m!",
                source_range: [145; 146),
                delete: [145; 146),
                insert: "m!($0)",
                kind: Macro,
                detail: "macro_rules! m",
            },
            CompletionItem {
                label: "quux(…)",
                source_range: [145; 146),
                delete: [145; 146),
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: [145; 146),
                delete: [145; 146),
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: [145; 146),
                delete: [145; 146),
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
                label: "m!",
                source_range: [145; 146),
                delete: [145; 146),
                insert: "m!($0)",
                kind: Macro,
                detail: "macro_rules! m",
            },
            CompletionItem {
                label: "quux(…)",
                source_range: [145; 146),
                delete: [145; 146),
                insert: "quux(${1:x})$0",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "x",
                source_range: [145; 146),
                delete: [145; 146),
                insert: "x",
                kind: Binding,
                detail: "i32",
            },
            CompletionItem {
                label: "y",
                source_range: [145; 146),
                delete: [145; 146),
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
                source_range: [82; 82),
                delete: [82; 82),
                insert: "Quux",
            },
            CompletionItem {
                label: "main()",
                source_range: [82; 82),
                delete: [82; 82),
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }
}
