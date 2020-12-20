//! Completes constats and paths in patterns.

use hir::StructKind;

use crate::{CompletionContext, Completions};

/// Completes constants and paths in patterns.
pub(crate) fn complete_pattern(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_pat_binding_or_const || ctx.is_irrefutable_pat_binding) {
        return;
    }
    if ctx.record_pat_syntax.is_some() {
        return;
    }

    // FIXME: ideally, we should look at the type we are matching against and
    // suggest variants + auto-imports
    ctx.scope.process_all_names(&mut |name, res| {
        let add_resolution = match &res {
            hir::ScopeDef::ModuleDef(def) => match def {
                hir::ModuleDef::Adt(hir::Adt::Struct(strukt)) => {
                    acc.add_struct_pat(ctx, strukt.clone(), Some(name.clone()));
                    true
                }
                hir::ModuleDef::Variant(variant)
                    if !ctx.is_irrefutable_pat_binding
                        // render_resolution already does some pattern completion tricks for tuple variants
                        && variant.kind(ctx.db) == StructKind::Record =>
                {
                    acc.add_variant_pat(ctx, variant.clone(), Some(name.clone()));
                    true
                }
                hir::ModuleDef::Adt(hir::Adt::Enum(..))
                | hir::ModuleDef::Variant(..)
                | hir::ModuleDef::Const(..)
                | hir::ModuleDef::Module(..) => !ctx.is_irrefutable_pat_binding,
                _ => false,
            },
            hir::ScopeDef::MacroDef(_) => true,
            _ => false,
        };
        if add_resolution {
            acc.add_resolution(ctx, name.to_string(), &res);
        }
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Reference);
        expect.assert_eq(&actual)
    }

    fn check_snippet(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Snippet);
        expect.assert_eq(&actual)
    }

    #[test]
    fn completes_enum_variants_and_modules() {
        check(
            r#"
enum E { X }
use self::E::X;
const Z: E = E::X;
mod m {}

static FOO: E = E::X;
struct Bar { f: u32 }

fn foo() {
   match E::X { <|> }
}
"#,
            expect![[r#"
                en E
                ct Z
                st Bar
                ev X   ()
                md m
            "#]],
        );
    }

    #[test]
    fn completes_in_simple_macro_call() {
        check(
            r#"
macro_rules! m { ($e:expr) => { $e } }
enum E { X }

fn foo() {
   m!(match E::X { <|> })
}
"#,
            expect![[r#"
                en E
                ma m!(…) macro_rules! m
            "#]],
        );
    }

    #[test]
    fn completes_in_irrefutable_let() {
        check(
            r#"
enum E { X }
use self::E::X;
const Z: E = E::X;
mod m {}

static FOO: E = E::X;
struct Bar { f: u32 }

fn foo() {
   let <|>
}
"#,
            expect![[r#"
                st Bar
            "#]],
        );
    }

    #[test]
    fn completes_in_param() {
        check(
            r#"
enum E { X }

static FOO: E = E::X;
struct Bar { f: u32 }

fn foo(<|>) {
}
"#,
            expect![[r#"
                st Bar
            "#]],
        );
    }

    #[test]
    fn completes_pat_in_let() {
        check_snippet(
            r#"
struct Bar { f: u32 }

fn foo() {
   let <|>
}
"#,
            expect![[r#"
                bn Bar Bar { ${1:f} }$0
            "#]],
        );
    }

    #[test]
    fn completes_param_pattern() {
        check_snippet(
            r#"
struct Foo { bar: String, baz: String }
struct Bar(String, String);
struct Baz;
fn outer(<|>) {}
"#,
            expect![[r#"
                bn Foo Foo { ${1:bar}, ${2:baz} }: Foo$0
                bn Bar Bar($1, $2): Bar$0
            "#]],
        )
    }

    #[test]
    fn completes_let_pattern() {
        check_snippet(
            r#"
struct Foo { bar: String, baz: String }
struct Bar(String, String);
struct Baz;
fn outer() {
    let <|>
}
"#,
            expect![[r#"
                bn Foo Foo { ${1:bar}, ${2:baz} }$0
                bn Bar Bar($1, $2)$0
            "#]],
        )
    }

    #[test]
    fn completes_refutable_pattern() {
        check_snippet(
            r#"
struct Foo { bar: i32, baz: i32 }
struct Bar(String, String);
struct Baz;
fn outer() {
    match () {
        <|>
    }
}
"#,
            expect![[r#"
                bn Foo Foo { ${1:bar}, ${2:baz} }$0
                bn Bar Bar($1, $2)$0
            "#]],
        )
    }

    #[test]
    fn omits_private_fields_pat() {
        check_snippet(
            r#"
mod foo {
    pub struct Foo { pub bar: i32, baz: i32 }
    pub struct Bar(pub String, String);
    pub struct Invisible(String, String);
}
use foo::*;

fn outer() {
    match () {
        <|>
    }
}
"#,
            expect![[r#"
                bn Foo Foo { ${1:bar}, .. }$0
                bn Bar Bar($1, ..)$0
            "#]],
        )
    }
}
