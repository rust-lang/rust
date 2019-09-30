//! FIXME: write short doc here

use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_macro_in_item_position(acc: &mut Completions, ctx: &CompletionContext) {
    // Show only macros in top level.
    if ctx.is_new_item {
        ctx.analyzer.process_all_names(ctx.db, &mut |name, res| {
            if let hir::ScopeDef::MacroDef(mac) = res {
                acc.add_macro(ctx, Some(name.to_string()), mac);
            }
        })
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
    fn completes_macros_as_item() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                macro_rules! foo {
                    () => {}
                }

                fn foo() {}

                <|>
                "
            ),
            @r##"[
    CompletionItem {
        label: "foo!",
        source_range: [46; 46),
        delete: [46; 46),
        insert: "foo!($0)",
        kind: Macro,
        detail: "macro_rules! foo",
    },
]"##
        );
    }

    #[test]
    fn completes_vec_macros_with_square_brackets() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /main.rs
                macro_rules! vec {
                    () => {}
                }

                fn foo() {}

                <|>
                "
            ),
            @r##"[
    CompletionItem {
        label: "vec!",
        source_range: [46; 46),
        delete: [46; 46),
        insert: "vec![$0]",
        kind: Macro,
        detail: "macro_rules! vec",
    },
]"##
        );
    }
}
