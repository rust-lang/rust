//! Completes constats and paths in patterns.

use crate::{CompletionContext, Completions};

/// Completes constats and paths in patterns.
pub(crate) fn complete_pattern(acc: &mut Completions, ctx: &CompletionContext) {
    if !(ctx.is_pat_binding_or_const || ctx.is_irrefutable_let_pat_binding) {
        return;
    }
    if ctx.record_pat_syntax.is_some() {
        return;
    }

    // FIXME: ideally, we should look at the type we are matching against and
    // suggest variants + auto-imports
    ctx.scope.process_all_names(&mut |name, res| {
        let add_resolution = match &res {
            hir::ScopeDef::ModuleDef(def) => {
                if ctx.is_irrefutable_let_pat_binding {
                    matches!(def, hir::ModuleDef::Adt(hir::Adt::Struct(_)))
                } else {
                    matches!(
                        def,
                        hir::ModuleDef::Adt(hir::Adt::Enum(..))
                            | hir::ModuleDef::Adt(hir::Adt::Struct(..))
                            | hir::ModuleDef::Variant(..)
                            | hir::ModuleDef::Const(..)
                            | hir::ModuleDef::Module(..)
                    )
                }
            }
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
}
