//! FIXME: write short doc here

use crate::completion::{CompletionContext, Completions};

/// Completes constats and paths in patterns.
pub(super) fn complete_pattern(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_pat_binding_or_const {
        return;
    }
    if ctx.record_pat_syntax.is_some() {
        return;
    }

    // FIXME: ideally, we should look at the type we are matching against and
    // suggest variants + auto-imports
    ctx.scope().process_all_names(&mut |name, res| {
        match &res {
            hir::ScopeDef::ModuleDef(def) => match def {
                hir::ModuleDef::Adt(hir::Adt::Enum(..))
                | hir::ModuleDef::Adt(hir::Adt::Struct(..))
                | hir::ModuleDef::EnumVariant(..)
                | hir::ModuleDef::Const(..)
                | hir::ModuleDef::Module(..) => (),
                _ => return,
            },
            hir::ScopeDef::MacroDef(_) => (),
            _ => return,
        };

        acc.add_resolution(ctx, name.to_string(), &res)
    });
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
                st Bar
                en E
                ev X   ()
                ct Z
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
}
