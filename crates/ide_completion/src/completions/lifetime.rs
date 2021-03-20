//! Completes lifetimes.
use hir::ScopeDef;

use crate::{completions::Completions, context::CompletionContext};

/// Completes lifetimes.
pub(crate) fn complete_lifetime(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.lifetime_allowed {
        return;
    }
    let param_lifetime = match (
        &ctx.lifetime_syntax,
        ctx.lifetime_param_syntax.as_ref().and_then(|lp| lp.lifetime()),
    ) {
        (Some(lt), Some(lp)) if lp == lt.clone() => return,
        (Some(_), Some(lp)) => Some(lp.to_string()),
        _ => None,
    };

    ctx.scope.process_all_names(&mut |name, res| {
        if let ScopeDef::GenericParam(hir::GenericParam::LifetimeParam(_)) = res {
            if param_lifetime != Some(name.to_string()) {
                acc.add_resolution(ctx, name.to_string(), &res);
            }
        }
    });
    if param_lifetime.is_none() {
        acc.add_static_lifetime(ctx);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        test_utils::{check_edit, completion_list_with_config, TEST_CONFIG},
        CompletionConfig, CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        check_with_config(TEST_CONFIG, ra_fixture, expect);
    }

    fn check_with_config(config: CompletionConfig, ra_fixture: &str, expect: Expect) {
        let actual = completion_list_with_config(config, ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual)
    }

    #[test]
    fn check_lifetime_edit() {
        check_edit(
            "'lifetime",
            r#"
fn func<'lifetime>(foo: &'li$0) {}
"#,
            r#"
fn func<'lifetime>(foo: &'lifetime) {}
"#,
        );
    }

    #[test]
    fn complete_lifetime_in_ref() {
        check(
            r#"
fn foo<'lifetime>(foo: &'a$0 usize) {}
"#,
            expect![[r#"
                lt 'lifetime
                lt 'static
            "#]],
        );
    }

    #[test]
    fn complete_lifetime_in_ref_missing_ty() {
        check(
            r#"
fn foo<'lifetime>(foo: &'a$0) {}
"#,
            expect![[r#"
                lt 'lifetime
                lt 'static
            "#]],
        );
    }
    #[test]
    fn complete_lifetime_in_self_ref() {
        check(
            r#"
struct Foo;
impl<'impl> Foo {
    fn foo<'func>(&'a$0 self) {}
}
"#,
            expect![[r#"
                lt 'func
                lt 'impl
                lt 'static
            "#]],
        );
    }

    #[test]
    fn complete_lifetime_in_arg_list() {
        check(
            r#"
struct Foo<'lt>;
fn foo<'lifetime>(_: Foo<'a$0>) {}
"#,
            expect![[r#"
                lt 'lifetime
                lt 'static
            "#]],
        );
    }

    #[test]
    fn complete_lifetime_in_where_pred() {
        check(
            r#"
fn foo2<'lifetime, T>() where 'a$0 {}
"#,
            expect![[r#"
                lt 'lifetime
                lt 'static
            "#]],
        );
    }

    #[test]
    fn complete_lifetime_in_ty_bound() {
        check(
            r#"
fn foo2<'lifetime, T>() where T: 'a$0 {}
"#,
            expect![[r#"
                lt 'lifetime
                lt 'static
            "#]],
        );
        check(
            r#"
fn foo2<'lifetime, T>() where T: Trait<'a$0> {}
"#,
            expect![[r#"
                lt 'lifetime
                lt 'static
            "#]],
        );
    }

    #[test]
    fn dont_complete_lifetime_in_assoc_ty_bound() {
        check(
            r#"
fn foo2<'lifetime, T>() where T: Trait<Item = 'a$0> {}
"#,
            expect![[r#""#]],
        );
    }

    #[test]
    fn complete_lifetime_in_param_list() {
        check(
            r#"
fn foo<'a$0>() {}
"#,
            expect![[r#""#]],
        );
        check(
            r#"
fn foo<'footime, 'lifetime: 'a$0>() {}
"#,
            expect![[r#"
                lt 'footime
            "#]],
        );
    }
}
