use rustc_hash::FxHashMap;
use ra_text_edit::TextEditBuilder;
use ra_syntax::{SmolStr, ast, AstNode};
use ra_assists::auto_import;

use crate::completion::{CompletionItem, Completions, CompletionKind, CompletionContext};

pub(super) fn complete_scope(acc: &mut Completions, ctx: &CompletionContext) {
    if ctx.is_trivial_path {
        let names = ctx.analyzer.all_names(ctx.db);
        names.into_iter().for_each(|(name, res)| acc.add_resolution(ctx, name.to_string(), &res));

        // auto-import
        // We fetch ident from the original file, because we need to pre-filter auto-imports
        if ast::NameRef::cast(ctx.token.parent()).is_some() {
            let import_resolver = ImportResolver::new();
            let import_names = import_resolver.all_names(ctx.token.text());
            import_names.into_iter().for_each(|(name, path)| {
                let edit = {
                    let mut builder = TextEditBuilder::default();
                    builder.replace(ctx.source_range(), name.to_string());
                    auto_import::auto_import_text_edit(
                        ctx.token.parent(),
                        ctx.token.parent(),
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
    use crate::completion::{CompletionKind, check_completion};

    fn check_reference_completion(name: &str, code: &str) {
        check_completion(name, code, CompletionKind::Reference);
    }

    #[test]
    fn completes_bindings_from_let() {
        check_reference_completion(
            "bindings_from_let",
            r"
            fn quux(x: i32) {
                let y = 92;
                1 + <|>;
                let z = ();
            }
            ",
        );
    }

    #[test]
    fn completes_bindings_from_if_let() {
        check_reference_completion(
            "bindings_from_if_let",
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
            ",
        );
    }

    #[test]
    fn completes_bindings_from_for() {
        check_reference_completion(
            "bindings_from_for",
            r"
            fn quux() {
                for x in &[1, 2, 3] {
                    <|>
                }
            }
            ",
        );
    }

    #[test]
    fn completes_generic_params() {
        check_reference_completion(
            "generic_params",
            r"
            fn quux<T>() {
                <|>
            }
            ",
        );
    }

    #[test]
    fn completes_generic_params_in_struct() {
        check_reference_completion(
            "generic_params_in_struct",
            r"
            struct X<T> {
                x: <|>
            }
            ",
        );
    }

    #[test]
    fn completes_module_items() {
        check_reference_completion(
            "module_items",
            r"
            struct Foo;
            enum Baz {}
            fn quux() {
                <|>
            }
            ",
        );
    }

    #[test]
    fn completes_extern_prelude() {
        check_reference_completion(
            "extern_prelude",
            r"
            //- /lib.rs
            use <|>;

            //- /other_crate/lib.rs
            // nothing here
            ",
        );
    }

    #[test]
    fn completes_module_items_in_nested_modules() {
        check_reference_completion(
            "module_items_in_nested_modules",
            r"
            struct Foo;
            mod m {
                struct Bar;
                fn quux() { <|> }
            }
            ",
        );
    }

    #[test]
    fn completes_return_type() {
        check_reference_completion(
            "return_type",
            r"
            struct Foo;
            fn x() -> <|>
            ",
        )
    }

    #[test]
    fn dont_show_both_completions_for_shadowing() {
        check_reference_completion(
            "dont_show_both_completions_for_shadowing",
            r"
            fn foo() {
                let bar = 92;
                {
                    let bar = 62;
                    <|>
                }
            }
            ",
        )
    }

    #[test]
    fn completes_self_in_methods() {
        check_reference_completion("self_in_methods", r"impl S { fn foo(&self) { <|> } }")
    }

    #[test]
    fn completes_prelude() {
        check_reference_completion(
            "completes_prelude",
            "
            //- /main.rs
            fn foo() { let x: <|> }

            //- /std/lib.rs
            #[prelude_import]
            use prelude::*;

            mod prelude {
                struct Option;
            }
            ",
        );
    }
}
