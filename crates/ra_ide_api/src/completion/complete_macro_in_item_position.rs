use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_macro_in_item_position(acc: &mut Completions, ctx: &CompletionContext) {
    // Show only macros in top level.
    if ctx.is_new_item {
        for (name, res) in ctx.analyzer.all_names(ctx.db) {
            if res.get_macros().is_some() {
                acc.add_resolution(ctx, name.to_string(), &res.only_macros());
            }
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
}
