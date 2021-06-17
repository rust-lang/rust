//! Completion for derives
use hir::HasAttrs;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use syntax::ast;

use crate::{
    context::CompletionContext,
    item::{CompletionItem, CompletionItemKind, CompletionKind},
    Completions,
};

pub(super) fn complete_derive(
    acc: &mut Completions,
    ctx: &CompletionContext,
    derive_input: ast::TokenTree,
) {
    if let Some(existing_derives) = super::parse_comma_sep_input(derive_input) {
        for (derive, docs) in get_derive_names_in_scope(ctx) {
            let (label, lookup) = if let Some(derive_completion) = DEFAULT_DERIVE_COMPLETIONS
                .iter()
                .find(|derive_completion| derive_completion.label == derive)
            {
                let mut components = vec![derive_completion.label];
                components.extend(
                    derive_completion
                        .dependencies
                        .iter()
                        .filter(|&&dependency| !existing_derives.contains(dependency)),
                );
                let lookup = components.join(", ");
                let label = components.iter().rev().join(", ");
                (label, Some(lookup))
            } else if existing_derives.contains(&derive) {
                continue;
            } else {
                (derive, None)
            };
            let mut item =
                CompletionItem::new(CompletionKind::Attribute, ctx.source_range(), label);
            item.kind(CompletionItemKind::Attribute);
            if let Some(docs) = docs {
                item.documentation(docs);
            }
            if let Some(lookup) = lookup {
                item.lookup_by(lookup);
            }
            item.add_to(acc);
        }
    }
}

fn get_derive_names_in_scope(
    ctx: &CompletionContext,
) -> FxHashMap<String, Option<hir::Documentation>> {
    let mut result = FxHashMap::default();
    ctx.scope.process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::MacroDef(mac) = scope_def {
            if mac.kind() == hir::MacroKind::Derive {
                result.insert(name.to_string(), mac.docs(ctx.db));
            }
        }
    });
    result
}

struct DeriveDependencies {
    label: &'static str,
    dependencies: &'static [&'static str],
}

/// Standard Rust derives that have dependencies
/// (the dependencies are needed so that the main derive don't break the compilation when added)
const DEFAULT_DERIVE_COMPLETIONS: &[DeriveDependencies] = &[
    DeriveDependencies { label: "Copy", dependencies: &["Clone"] },
    DeriveDependencies { label: "Eq", dependencies: &["PartialEq"] },
    DeriveDependencies { label: "Ord", dependencies: &["PartialOrd", "Eq", "PartialEq"] },
    DeriveDependencies { label: "PartialOrd", dependencies: &["PartialEq"] },
];

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::completion_list;

    fn check(ra_fixture: &str, expect: Expect) {
        let builtin_derives = r#"
#[rustc_builtin_macro]
pub macro Clone {}
#[rustc_builtin_macro]
pub macro Copy {}
#[rustc_builtin_macro]
pub macro Default {}
#[rustc_builtin_macro]
pub macro Debug {}
#[rustc_builtin_macro]
pub macro Hash {}
#[rustc_builtin_macro]
pub macro PartialEq {}
#[rustc_builtin_macro]
pub macro Eq {}
#[rustc_builtin_macro]
pub macro PartialOrd {}
#[rustc_builtin_macro]
pub macro Ord {}

"#;
        let actual = completion_list(&format!("{} {}", builtin_derives, ra_fixture));
        expect.assert_eq(&actual);
    }

    #[test]
    fn no_completion_for_incorrect_derive() {
        check(r#"#[derive{$0)] struct Test;"#, expect![[]])
    }

    #[test]
    fn empty_derive() {
        check(
            r#"#[derive($0)] struct Test;"#,
            expect![[r#"
            at PartialEq
            at Default
            at PartialEq, Eq
            at PartialEq, Eq, PartialOrd, Ord
            at Clone, Copy
            at Debug
            at Clone
            at Hash
            at PartialEq, PartialOrd
        "#]],
        );
    }

    #[test]
    fn derive_with_input() {
        check(
            r#"#[derive(serde::Serialize, PartialEq, $0)] struct Test;"#,
            expect![[r#"
                at Default
                at Eq
                at Eq, PartialOrd, Ord
                at Clone, Copy
                at Debug
                at Clone
                at Hash
                at PartialOrd
            "#]],
        )
    }

    #[test]
    fn derive_with_input2() {
        check(
            r#"#[derive($0 serde::Serialize, PartialEq)] struct Test;"#,
            expect![[r#"
                at Default
                at Eq
                at Eq, PartialOrd, Ord
                at Clone, Copy
                at Debug
                at Clone
                at Hash
                at PartialOrd
            "#]],
        )
    }
}
