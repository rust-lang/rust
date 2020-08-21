//! Complete fields in record literals and patterns.
use crate::completion::{CompletionContext, Completions};

pub(super) fn complete_record(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    let missing_fields = match (ctx.record_pat_syntax.as_ref(), ctx.record_lit_syntax.as_ref()) {
        (None, None) => return None,
        (Some(_), Some(_)) => unreachable!("A record cannot be both a literal and a pattern"),
        (Some(record_pat), _) => ctx.sema.record_pattern_missing_fields(record_pat),
        (_, Some(record_lit)) => ctx.sema.record_literal_missing_fields(record_lit),
    };

    for (field, ty) in missing_fields {
        acc.add_field(ctx, field, &ty)
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::completion::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_record_pattern_field() {
        check(
            r#"
struct S { foo: u32 }

fn process(f: S) {
    match f {
        S { f<|>: 92 } => (),
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
        E::S { <|> } => (),
    }
}
"#,
            expect![[r#"
                fd bar ()
                fd foo u32
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
        S { f<|>: 92 } => (),
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
    if let S { foo1, foo2: a, <|> } = s {}
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
   A { the<|> }
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
    let _ = E::A { <|> }
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
   let _: A = B { <|> }
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
   let _: A<u32> = A { <|> }
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
   m!(A { the<|> })
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
    let s = S { foo1, foo2: 5, <|> }
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
    let s = S { foo1, <|> .. loop {} }
}
"#,
            expect![[r#"
                fd foo2 u32
            "#]],
        );
    }
}
