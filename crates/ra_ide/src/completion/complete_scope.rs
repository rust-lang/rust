//! FIXME: write short doc here

use ra_assists::auto_import_text_edit;
use ra_syntax::{ast, AstNode, SmolStr};
use ra_text_edit::TextEditBuilder;
use rustc_hash::FxHashMap;

use crate::completion::{CompletionContext, CompletionItem, CompletionKind, Completions};

pub(super) fn complete_scope(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_trivial_path {
        return;
    }

    ctx.analyzer.process_all_names(ctx.db, &mut |name, res| {
        acc.add_resolution(ctx, name.to_string(), &res)
    });

    // auto-import
    // We fetch ident from the original file, because we need to pre-filter auto-imports
    if ast::NameRef::cast(ctx.token.parent()).is_some() {
        let import_resolver = ImportResolver::new();
        let import_names = import_resolver.all_names(ctx.token.text());
        import_names.into_iter().for_each(|(name, path)| {
            let edit = {
                let mut builder = TextEditBuilder::default();
                builder.replace(ctx.source_range(), name.to_string());
                auto_import_text_edit(
                    &ctx.token.parent(),
                    &ctx.token.parent(),
                    &path,
                    &mut builder,
                );
                builder.finish()
            };

            // Hack: copied this check form conv.rs beacause auto import can produce edits
            // that invalidate assert in conv_with.
            if edit
                .as_atoms()
                .iter()
                .filter(|atom| !ctx.source_range().is_subrange(&atom.delete))
                .all(|atom| ctx.source_range().intersection(&atom.delete).is_none())
            {
                CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    build_import_label(&name, &path),
                )
                .text_edit(edit)
                .add_to(acc);
            }
        });
    }
}

fn build_import_label(name: &str, path: &[SmolStr]) -> String {
    let mut buf = String::with_capacity(64);
    buf.push_str(name);
    buf.push_str(" (");
    fmt_import_path(path, &mut buf);
    buf.push_str(")");
    buf
}

fn fmt_import_path(path: &[SmolStr], buf: &mut String) {
    let mut segments = path.iter();
    if let Some(s) = segments.next() {
        buf.push_str(&s);
    }
    for s in segments {
        buf.push_str("::");
        buf.push_str(&s);
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ImportResolver {
    // todo: use fst crate or something like that
    dummy_names: Vec<(SmolStr, Vec<SmolStr>)>,
}

impl ImportResolver {
    pub(crate) fn new() -> Self {
        let dummy_names = vec![
            (SmolStr::new("fmt"), vec![SmolStr::new("std"), SmolStr::new("fmt")]),
            (SmolStr::new("io"), vec![SmolStr::new("std"), SmolStr::new("io")]),
            (SmolStr::new("iter"), vec![SmolStr::new("std"), SmolStr::new("iter")]),
            (SmolStr::new("hash"), vec![SmolStr::new("std"), SmolStr::new("hash")]),
            (
                SmolStr::new("Debug"),
                vec![SmolStr::new("std"), SmolStr::new("fmt"), SmolStr::new("Debug")],
            ),
            (
                SmolStr::new("Display"),
                vec![SmolStr::new("std"), SmolStr::new("fmt"), SmolStr::new("Display")],
            ),
            (
                SmolStr::new("Hash"),
                vec![SmolStr::new("std"), SmolStr::new("hash"), SmolStr::new("Hash")],
            ),
            (
                SmolStr::new("Hasher"),
                vec![SmolStr::new("std"), SmolStr::new("hash"), SmolStr::new("Hasher")],
            ),
            (
                SmolStr::new("Iterator"),
                vec![SmolStr::new("std"), SmolStr::new("iter"), SmolStr::new("Iterator")],
            ),
        ];

        ImportResolver { dummy_names }
    }

    // Returns a map of importable items filtered by name.
    // The map associates item name with its full path.
    // todo: should return Resolutions
    pub(crate) fn all_names(&self, name: &str) -> FxHashMap<SmolStr, Vec<SmolStr>> {
        if name.len() > 1 {
            self.dummy_names.iter().filter(|(n, _)| n.contains(name)).cloned().collect()
        } else {
            FxHashMap::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_reference_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
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
                insert: "quux($0)",
                kind: Function,
                lookup: "quux",
                detail: "fn quux(x: i32)",
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
}
