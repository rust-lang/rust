//! Complete fields in record literals and patterns.
use ide_db::{helpers::FamousDefs, SymbolKind};
use syntax::ast::Expr;

use crate::{item::CompletionKind, CompletionContext, CompletionItem, Completions};

pub(crate) fn complete_record(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let missing_fields = match (ctx.record_pat_syntax.as_ref(), ctx.record_lit_syntax.as_ref()) {
        (None, None) => return None,
        (Some(_), Some(_)) => unreachable!("A record cannot be both a literal and a pattern"),
        (Some(record_pat), _) => ctx.sema.record_pattern_missing_fields(record_pat),
        (_, Some(record_lit)) => {
            let ty = ctx.sema.type_of_expr(&Expr::RecordExpr(record_lit.clone()));
            let default_trait = FamousDefs(&ctx.sema, ctx.krate).core_default_Default();
            let impl_default_trait = default_trait
                .and_then(|default_trait| ty.map(|ty| ty.impls_trait(ctx.db, default_trait, &[])))
                .unwrap_or(false);

            let missing_fields = ctx.sema.record_literal_missing_fields(record_lit);
            if impl_default_trait && !missing_fields.is_empty() {
                let completion_text = "..Default::default()";
                let completion_text = completion_text
                    .strip_prefix(ctx.token.to_string().as_str())
                    .unwrap_or(completion_text);
                acc.add(
                    CompletionItem::new(
                        CompletionKind::Snippet,
                        ctx.source_range(),
                        "..Default::default()",
                    )
                    .insert_text(completion_text)
                    .kind(SymbolKind::Field)
                    .build(),
                );
            }

            missing_fields
        }
    };

    for (field, ty) in missing_fields {
        acc.add_field(ctx, field, &ty);
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide_db::helpers::FamousDefs;

    use crate::{
        test_utils::{self, completion_list},
        CompletionKind,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual);
    }

    fn check_snippet(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(
            &format!("//- /main.rs crate:main deps:core\n{}\n{}", ra_fixture, FamousDefs::FIXTURE),
            CompletionKind::Snippet,
        );
        expect.assert_eq(&actual);
    }

    fn check_edit(what: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
        test_utils::check_edit(
            what,
            &format!(
                "//- /main.rs crate:main deps:core{}\n{}",
                ra_fixture_before,
                FamousDefs::FIXTURE,
            ),
            &(ra_fixture_after.to_owned() + "\n"),
        );
    }

    #[test]
    fn test_record_literal_field_default() {
        let test_code = r#"
struct S { foo: u32, bar: usize }

impl core::default::Default for S {
    fn default() -> Self {
        S {
            foo: 0,
            bar: 0,
        }
    }
}

fn process(f: S) {
    let other = S {
        foo: 5,
        .$0
    };
}
"#;
        check(
            test_code,
            expect![[r#"
                fd bar usize
            "#]],
        );

        check_snippet(
            test_code,
            expect![[r#"
                sn pd
                sn ppd
                fd ..Default::default()
            "#]],
        );
    }

    #[test]
    fn test_record_literal_field_default_completion() {
        check_edit(
            "..Default::default()",
            r#"
struct S { foo: u32, bar: usize }

impl core::default::Default for S {
    fn default() -> Self {
        S {
            foo: 0,
            bar: 0,
        }
    }
}

fn process(f: S) {
    let other = S {
        foo: 5,
        .$0
    };
}
"#,
            r#"
struct S { foo: u32, bar: usize }

impl core::default::Default for S {
    fn default() -> Self {
        S {
            foo: 0,
            bar: 0,
        }
    }
}

fn process(f: S) {
    let other = S {
        foo: 5,
        ..Default::default()
    };
}
"#,
        );
    }

    #[test]
    fn test_record_literal_field_without_default() {
        let test_code = r#"
struct S { foo: u32, bar: usize }

fn process(f: S) {
    let other = S {
        foo: 5,
        .$0
    };
}
"#;
        check(
            test_code,
            expect![[r#"
                fd bar usize
            "#]],
        );

        check_snippet(
            test_code,
            expect![[r#"
                sn pd
                sn ppd
            "#]],
        );
    }

    #[test]
    fn test_record_pattern_field() {
        check(
            r#"
struct S { foo: u32 }

fn process(f: S) {
    match f {
        S { f$0: 92 } => (),
    }
}
"#,
            expect![[r#"
                fd foo u32
            "#]],
        );
    }

    #[test]
    fn test_record_pattern_enum_variant() {
        check(
            r#"
enum E { S { foo: u32, bar: () } }

fn process(e: E) {
    match e {
        E::S { $0 } => (),
    }
}
"#,
            expect![[r#"
                fd foo u32
                fd bar ()
            "#]],
        );
    }

    #[test]
    fn test_record_pattern_field_in_simple_macro() {
        check(
            r"
macro_rules! m { ($e:expr) => { $e } }
struct S { foo: u32 }

fn process(f: S) {
    m!(match f {
        S { f$0: 92 } => (),
    })
}
",
            expect![[r#"
                fd foo u32
            "#]],
        );
    }

    #[test]
    fn only_missing_fields_are_completed_in_destruct_pats() {
        check(
            r#"
struct S {
    foo1: u32, foo2: u32,
    bar: u32, baz: u32,
}

fn main() {
    let s = S {
        foo1: 1, foo2: 2,
        bar: 3, baz: 4,
    };
    if let S { foo1, foo2: a, $0 } = s {}
}
"#,
            expect![[r#"
                fd bar u32
                fd baz u32
            "#]],
        );
    }

    #[test]
    fn test_record_literal_field() {
        check(
            r#"
struct A { the_field: u32 }
fn foo() {
   A { the$0 }
}
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn test_record_literal_enum_variant() {
        check(
            r#"
enum E { A { a: u32 } }
fn foo() {
    let _ = E::A { $0 }
}
"#,
            expect![[r#"
                fd a u32
            "#]],
        );
    }

    #[test]
    fn test_record_literal_two_structs() {
        check(
            r#"
struct A { a: u32 }
struct B { b: u32 }

fn foo() {
   let _: A = B { $0 }
}
"#,
            expect![[r#"
                fd b u32
            "#]],
        );
    }

    #[test]
    fn test_record_literal_generic_struct() {
        check(
            r#"
struct A<T> { a: T }

fn foo() {
   let _: A<u32> = A { $0 }
}
"#,
            expect![[r#"
                fd a u32
            "#]],
        );
    }

    #[test]
    fn test_record_literal_field_in_simple_macro() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
struct A { the_field: u32 }
fn foo() {
   m!(A { the$0 })
}
"#,
            expect![[r#"
                fd the_field u32
            "#]],
        );
    }

    #[test]
    fn only_missing_fields_are_completed() {
        check(
            r#"
struct S {
    foo1: u32, foo2: u32,
    bar: u32, baz: u32,
}

fn main() {
    let foo1 = 1;
    let s = S { foo1, foo2: 5, $0 }
}
"#,
            expect![[r#"
                fd bar u32
                fd baz u32
            "#]],
        );
    }

    #[test]
    fn completes_functional_update() {
        check(
            r#"
struct S { foo1: u32, foo2: u32 }

fn main() {
    let foo1 = 1;
    let s = S { foo1, $0 .. loop {} }
}
"#,
            expect![[r#"
                fd foo2 u32
            "#]],
        );
    }
}
