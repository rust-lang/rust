//! Completes `where` and `for` keywords.

use syntax::ast::{self, Item};

use crate::{CompletionContext, Completions};

pub(crate) fn complete_for_and_where(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    keyword_item: &ast::Item,
) {
    let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);

    match keyword_item {
        Item::Impl(it) => {
            if it.for_token().is_none() && it.trait_().is_none() && it.self_ty().is_some() {
                add_keyword("for", "for $0");
            }
            add_keyword("where", "where $0");
        }
        Item::Enum(_)
        | Item::Fn(_)
        | Item::Struct(_)
        | Item::Trait(_)
        | Item::TypeAlias(_)
        | Item::Union(_) => {
            add_keyword("where", "where $0");
        }
        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::tests::{check, check_edit};

    #[test]
    fn test_else_edit_after_if() {
        check_edit(
            "else",
            r#"fn quux() { if true { () } $0 }"#,
            r#"fn quux() { if true { () } else {
    $0
} }"#,
        );
    }

    #[test]
    fn test_keywords_after_unsafe_in_block_expr() {
        check(
            r"fn my_fn() { unsafe $0 }",
            expect![[r#"
                kw async
                kw extern
                kw fn
                kw impl
                kw impl for
                kw trait
            "#]],
        );
    }

    #[test]
    fn test_completion_await_impls_future() {
        check(
            r#"
//- minicore: future
use core::future::*;
struct A {}
impl Future for A {}
fn foo(a: A) { a.$0 }
"#,
            expect![[r#"
                me into_future() (as IntoFuture) fn(self) -> <Self as IntoFuture>::IntoFuture
                kw await                                                           expr.await
                sn box                                                         Box::new(expr)
                sn call                                                        function(expr)
                sn const                                                             const {}
                sn dbg                                                             dbg!(expr)
                sn dbgr                                                           dbg!(&expr)
                sn deref                                                                *expr
                sn let                                                                    let
                sn letm                                                               let mut
                sn match                                                        match expr {}
                sn ref                                                                  &expr
                sn refm                                                             &mut expr
                sn return                                                         return expr
                sn unsafe                                                           unsafe {}
            "#]],
        );

        check(
            r#"
//- minicore: future
use std::future::*;
fn foo() {
    let a = async {};
    a.$0
}
"#,
            expect![[r#"
                me into_future() (use core::future::IntoFuture) fn(self) -> <Self as IntoFuture>::IntoFuture
                kw await                                                                          expr.await
                sn box                                                                        Box::new(expr)
                sn call                                                                       function(expr)
                sn const                                                                            const {}
                sn dbg                                                                            dbg!(expr)
                sn dbgr                                                                          dbg!(&expr)
                sn deref                                                                               *expr
                sn let                                                                                   let
                sn letm                                                                              let mut
                sn match                                                                       match expr {}
                sn ref                                                                                 &expr
                sn refm                                                                            &mut expr
                sn return                                                                        return expr
                sn unsafe                                                                          unsafe {}
            "#]],
        );
    }

    #[test]
    fn test_completion_await_impls_into_future() {
        check(
            r#"
//- minicore: future
use core::future::*;
struct A {}
impl IntoFuture for A {}
fn foo(a: A) { a.$0 }
"#,
            expect![[r#"
                me into_future() (as IntoFuture) fn(self) -> <Self as IntoFuture>::IntoFuture
                kw await                                                           expr.await
                sn box                                                         Box::new(expr)
                sn call                                                        function(expr)
                sn const                                                             const {}
                sn dbg                                                             dbg!(expr)
                sn dbgr                                                           dbg!(&expr)
                sn deref                                                                *expr
                sn let                                                                    let
                sn letm                                                               let mut
                sn match                                                        match expr {}
                sn ref                                                                  &expr
                sn refm                                                             &mut expr
                sn return                                                         return expr
                sn unsafe                                                           unsafe {}
            "#]],
        );
    }

    #[test]
    fn for_in_impl() {
        check_edit(
            "for",
            r#"
struct X;
impl X $0 {}
"#,
            r#"
struct X;
impl X for $0 {}
"#,
        );
        check_edit(
            "for",
            r#"
fn foo() {
    struct X;
    impl X $0 {}
}
"#,
            r#"
fn foo() {
    struct X;
    impl X for $0 {}
}
"#,
        );
        check_edit(
            "for",
            r#"
fn foo() {
    struct X;
    impl X $0
}
"#,
            r#"
fn foo() {
    struct X;
    impl X for $0
}
"#,
        );
        check_edit(
            "for",
            r#"
fn foo() {
    struct X;
    impl X { fn bar() { $0 } }
}
"#,
            r#"
fn foo() {
    struct X;
    impl X { fn bar() { for $1 in $2 {
    $0
} } }
}
"#,
        );
    }

    #[test]
    fn let_semi() {
        cov_mark::check!(let_semi);
        check_edit(
            "match",
            r#"
fn main() { let x = $0 }
"#,
            r#"
fn main() { let x = match $1 {
    $0
}; }
"#,
        );

        check_edit(
            "if",
            r#"
fn main() {
    let x = $0
    let y = 92;
}
"#,
            r#"
fn main() {
    let x = if $1 {
    $0
};
    let y = 92;
}
"#,
        );

        check_edit(
            "loop",
            r#"
fn main() {
    let x = $0
    bar();
}
"#,
            r#"
fn main() {
    let x = loop {
    $0
};
    bar();
}
"#,
        );
    }

    #[test]
    fn if_completion_in_match_guard() {
        check_edit(
            "if",
            r"
fn main() {
    match () {
        () $0
    }
}
",
            r"
fn main() {
    match () {
        () if $0
    }
}
",
        )
    }

    #[test]
    fn if_completion_in_match_arm_expr() {
        check_edit(
            "if",
            r"
fn main() {
    match () {
        () => $0
    }
}
",
            r"
fn main() {
    match () {
        () => if $1 {
    $0
}
    }
}
",
        )
    }

    #[test]
    fn if_completion_in_match_arm_expr_block() {
        check_edit(
            "if",
            r"
fn main() {
    match () {
        () => {
            $0
        }
    }
}
",
            r"
fn main() {
    match () {
        () => {
            if $1 {
    $0
}
        }
    }
}
",
        )
    }

    #[test]
    fn completes_let_in_block() {
        check_edit(
            "let",
            r#"
fn main() {
    $0
}
"#,
            r#"
fn main() {
    let $1 = $0;
}
"#,
        );
        check_edit(
            "letm",
            r#"
fn main() {
    $0
}
"#,
            r#"
fn main() {
    let mut $1 = $0;
}
"#,
        );
    }

    #[test]
    fn completes_let_in_condition() {
        check_edit(
            "let",
            r#"
fn main() {
    if $0 {}
}
"#,
            r#"
fn main() {
    if let $1 = $0 {}
}
"#,
        );
        check_edit(
            "letm",
            r#"
fn main() {
    if $0 {}
}
"#,
            r#"
fn main() {
    if let mut $1 = $0 {}
}
"#,
        );
    }

    #[test]
    fn completes_let_in_no_empty_condition() {
        check_edit(
            "let",
            r#"
fn main() {
    if $0x {}
}
"#,
            r#"
fn main() {
    if let $1 = $0x {}
}
"#,
        );
        check_edit(
            "letm",
            r#"
fn main() {
    if $0x {}
}
"#,
            r#"
fn main() {
    if let mut $1 = $0x {}
}
"#,
        );
    }

    #[test]
    fn completes_let_in_condition_block() {
        check_edit(
            "let",
            r#"
fn main() {
    if { $0 } {}
}
"#,
            r#"
fn main() {
    if { let $1 = $0; } {}
}
"#,
        );
        check_edit(
            "letm",
            r#"
fn main() {
    if { $0 } {}
}
"#,
            r#"
fn main() {
    if { let mut $1 = $0; } {}
}
"#,
        );
    }
}
