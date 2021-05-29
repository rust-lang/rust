//! Completion for derives
use itertools::Itertools;
use rustc_hash::FxHashSet;
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
        for derive_completion in DEFAULT_DERIVE_COMPLETIONS
            .iter()
            .filter(|completion| !existing_derives.contains(completion.label))
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
            let mut item =
                CompletionItem::new(CompletionKind::Attribute, ctx.source_range(), label);
            item.lookup_by(lookup).kind(CompletionItemKind::Attribute);
            item.add_to(acc);
        }

        for custom_derive_name in get_derive_names_in_scope(ctx).difference(&existing_derives) {
            let mut item = CompletionItem::new(
                CompletionKind::Attribute,
                ctx.source_range(),
                custom_derive_name,
            );
            item.kind(CompletionItemKind::Attribute);
            item.add_to(acc);
        }
    }
}
fn get_derive_names_in_scope(ctx: &CompletionContext) -> FxHashSet<String> {
    let mut result = FxHashSet::default();
    ctx.scope.process_all_names(&mut |name, scope_def| {
        if let hir::ScopeDef::MacroDef(mac) = scope_def {
            // FIXME kind() doesn't check whether proc-macro is a derive
            if mac.kind() == hir::MacroKind::Derive || mac.kind() == hir::MacroKind::ProcMacro {
                result.insert(name.to_string());
            }
        }
    });
    result
}

struct DeriveCompletion {
    label: &'static str,
    dependencies: &'static [&'static str],
}

/// Standard Rust derives and the information about their dependencies
/// (the dependencies are needed so that the main derive don't break the compilation when added)
const DEFAULT_DERIVE_COMPLETIONS: &[DeriveCompletion] = &[
    DeriveCompletion { label: "Clone", dependencies: &[] },
    DeriveCompletion { label: "Copy", dependencies: &["Clone"] },
    DeriveCompletion { label: "Debug", dependencies: &[] },
    DeriveCompletion { label: "Default", dependencies: &[] },
    DeriveCompletion { label: "Hash", dependencies: &[] },
    DeriveCompletion { label: "PartialEq", dependencies: &[] },
    DeriveCompletion { label: "Eq", dependencies: &["PartialEq"] },
    DeriveCompletion { label: "PartialOrd", dependencies: &["PartialEq"] },
    DeriveCompletion { label: "Ord", dependencies: &["PartialOrd", "Eq", "PartialEq"] },
];

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Attribute);
        expect.assert_eq(&actual);
    }

    #[test]
    fn empty_derive_completion() {
        check(
            r#"
#[derive($0)]
struct Test {}
        "#,
            expect![[r#"
                at Clone
                at Clone, Copy
                at Debug
                at Default
                at Hash
                at PartialEq
                at PartialEq, Eq
                at PartialEq, PartialOrd
                at PartialEq, Eq, PartialOrd, Ord
            "#]],
        );
    }

    #[test]
    fn no_completion_for_incorrect_derive() {
        check(
            r#"
#[derive{$0)]
struct Test {}
"#,
            expect![[r#""#]],
        )
    }

    #[test]
    fn derive_with_input_completion() {
        check(
            r#"
#[derive(serde::Serialize, PartialEq, $0)]
struct Test {}
"#,
            expect![[r#"
                at Clone
                at Clone, Copy
                at Debug
                at Default
                at Hash
                at Eq
                at PartialOrd
                at Eq, PartialOrd, Ord
            "#]],
        )
    }
}
