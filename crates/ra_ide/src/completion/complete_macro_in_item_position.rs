//! FIXME: write short doc here

use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_macro_in_item_position(acc: &mut Completions, ctx: &CompletionContext) {
    // Show only macros in top level.
    if ctx.is_new_item {
        ctx.scope().process_all_names(&mut |name, res| {
            if let hir::ScopeDef::MacroDef(mac) = res {
                acc.add_macro(ctx, Some(name.to_string()), mac);
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};

    use crate::completion::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual)
    }

    #[test]
    fn completes_macros_as_item() {
        check(
            r#"
macro_rules! foo { () => {} }
fn foo() {}

<|>
"#,
            expect![[r#"
                ma foo!(…) macro_rules! foo
            "#]],
        )
    }
}
