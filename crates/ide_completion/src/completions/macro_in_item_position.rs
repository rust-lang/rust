//! Completes macro invocations used in item position.

use crate::{CompletionContext, Completions};

// Ideally this should be removed and moved into `(un)qualified_path` respectively
pub(crate) fn complete_macro_in_item_position(acc: &mut Completions, ctx: &CompletionContext) {
    // Show only macros in top level.
    if !ctx.expects_item() {
        return;
    }

    ctx.scope.process_all_names(&mut |name, res| {
        if let hir::ScopeDef::MacroDef(mac) = res {
            acc.add_macro(ctx, Some(name.clone()), mac);
        }
        // FIXME: This should be done in qualified_path/unqualified_path instead?
        if let hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) = res {
            acc.add_resolution(ctx, name, &res);
        }
    })
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{test_utils::completion_list, CompletionKind};

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

$0
"#,
            expect![[r#"
                ma foo!(…) macro_rules! foo
            "#]],
        )
    }
}
