//! Completes lifetimes and labels.
//!
//! These completions work a bit differently in that they are only shown when what the user types
//! has a `'` preceding it, as our fake syntax tree is invalid otherwise (due to us not inserting
//! a lifetime but an ident for obvious reasons).
//! Due to this all the tests for lifetimes and labels live in this module for the time being as
//! there is no value in lifting these out into the outline module test since they will either not
//! show up for normal completions, or they won't show completions other than lifetimes depending
//! on the fixture input.
use hir::{Name, ScopeDef, sym};

use crate::{
    completions::Completions,
    context::{CompletionContext, LifetimeContext, LifetimeKind},
};

/// Completes lifetimes.
pub(crate) fn complete_lifetime(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    lifetime_ctx: &LifetimeContext,
) {
    let &LifetimeContext { kind: LifetimeKind::Lifetime { in_lifetime_param_bound, def }, .. } =
        lifetime_ctx
    else {
        return;
    };

    ctx.process_all_names_raw(&mut |name, res| {
        if matches!(res, ScopeDef::GenericParam(hir::GenericParam::LifetimeParam(_))) {
            acc.add_lifetime(ctx, name);
        }
    });
    acc.add_lifetime(ctx, Name::new_symbol_root(sym::tick_static));
    if !in_lifetime_param_bound
        && def.is_some_and(|def| {
            !matches!(def, hir::GenericDef::Function(_) | hir::GenericDef::Impl(_))
        })
    {
        acc.add_lifetime(ctx, Name::new_symbol_root(sym::tick_underscore));
    }
}

/// Completes labels.
pub(crate) fn complete_label(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    lifetime_ctx: &LifetimeContext,
) {
    if !matches!(lifetime_ctx, LifetimeContext { kind: LifetimeKind::LabelRef, .. }) {
        return;
    }
    ctx.process_all_names_raw(&mut |name, res| {
        if let ScopeDef::Label(_) = res {
            acc.add_label(ctx, name);
        }
    });
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::tests::{check, check_edit};

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
        cov_mark::check!(completes_if_lifetime_without_idents);
        check_edit(
            "'lifetime",
            r#"
fn func<'lifetime>(foo: &'$0) {}
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
impl<'r#impl> Foo {
    fn foo<'func>(&'a$0 self) {}
}
"#,
            expect![[r#"
                lt 'func
                lt 'r#impl
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
fn foo<'$0>() {}
"#,
            expect![[r#""#]],
        );
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
                lt 'lifetime
                lt 'static
            "#]],
        );
    }

    #[test]
    fn check_label_edit() {
        check_edit(
            "'label",
            r#"
fn foo() {
    'label: loop {
        break '$0
    }
}
"#,
            r#"
fn foo() {
    'label: loop {
        break 'label
    }
}
"#,
        );
    }

    #[test]
    fn complete_label_in_loop() {
        check(
            r#"
fn foo() {
    'foop: loop {
        break '$0
    }
}
"#,
            expect![[r#"
                lb 'foop
            "#]],
        );
        check(
            r#"
fn foo() {
    'foop: loop {
        continue '$0
    }
}
"#,
            expect![[r#"
                lb 'foop
            "#]],
        );
    }

    #[test]
    fn complete_label_in_block_nested() {
        check(
            r#"
fn foo() {
    'foop: {
        'baap: {
            break '$0
        }
    }
}
"#,
            expect![[r#"
                lb 'baap
                lb 'foop
            "#]],
        );
    }

    #[test]
    fn complete_label_in_loop_with_value() {
        check(
            r#"
fn foo() {
    'foop: loop {
        break '$0 i32;
    }
}
"#,
            expect![[r#"
                lb 'foop
            "#]],
        );
    }

    #[test]
    fn complete_label_in_while_cond() {
        check(
            r#"
fn foo() {
    'outer: while { 'inner: loop { break '$0 } } {}
}
"#,
            expect![[r#"
                lb 'inner
                lb 'outer
            "#]],
        );
    }

    #[test]
    fn complete_label_in_for_iterable() {
        check(
            r#"
//- minicore: iterator
fn foo() {
    'outer: for _ in [{ 'inner: loop { break '$0 } }] {}
}
"#,
            expect![[r#"
                lb 'inner
            "#]],
        );
    }
}
