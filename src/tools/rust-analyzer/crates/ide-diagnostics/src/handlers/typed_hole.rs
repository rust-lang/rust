use hir::{db::ExpandDatabase, ClosureStyle, HirDisplay, StructKind};
use ide_db::{
    assists::{Assist, AssistId, AssistKind, GroupLabel},
    label::Label,
    source_change::SourceChange,
};
use syntax::AstNode;
use text_edit::TextEdit;

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: typed-hole
//
// This diagnostic is triggered when an underscore expression is used in an invalid position.
pub(crate) fn typed_hole(ctx: &DiagnosticsContext<'_>, d: &hir::TypedHole) -> Diagnostic {
    let display_range = ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into()));
    let (message, fixes) = if d.expected.is_unknown() {
        ("`_` expressions may only appear on the left-hand side of an assignment".to_owned(), None)
    } else {
        (
            format!(
                "invalid `_` expression, expected type `{}`",
                d.expected.display(ctx.sema.db).with_closure_style(ClosureStyle::ClosureWithId),
            ),
            fixes(ctx, d),
        )
    };

    Diagnostic::new("typed-hole", message, display_range.range).with_fixes(fixes)
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::TypedHole) -> Option<Vec<Assist>> {
    let db = ctx.sema.db;
    let root = db.parse_or_expand(d.expr.file_id);
    let original_range =
        d.expr.as_ref().map(|it| it.to_node(&root)).syntax().original_file_range_opt(db)?;
    let scope = ctx.sema.scope(d.expr.value.to_node(&root).syntax())?;
    let mut assists = vec![];
    scope.process_all_names(&mut |name, def| {
        let ty = match def {
            hir::ScopeDef::ModuleDef(it) => match it {
                hir::ModuleDef::Function(it) => it.ty(db),
                hir::ModuleDef::Adt(hir::Adt::Struct(it)) if it.kind(db) != StructKind::Record => {
                    it.constructor_ty(db)
                }
                hir::ModuleDef::Variant(it) if it.kind(db) != StructKind::Record => {
                    it.constructor_ty(db)
                }
                hir::ModuleDef::Const(it) => it.ty(db),
                hir::ModuleDef::Static(it) => it.ty(db),
                _ => return,
            },
            hir::ScopeDef::GenericParam(hir::GenericParam::ConstParam(it)) => it.ty(db),
            hir::ScopeDef::Local(it) => it.ty(db),
            _ => return,
        };
        // FIXME: should also check coercions if it is at a coercion site
        if !ty.contains_unknown() && ty.could_unify_with(db, &d.expected) {
            assists.push(Assist {
                id: AssistId("typed-hole", AssistKind::QuickFix),
                label: Label::new(format!("Replace `_` with `{}`", name.display(db))),
                group: Some(GroupLabel("Replace `_` with a matching entity in scope".to_owned())),
                target: original_range.range,
                source_change: Some(SourceChange::from_text_edit(
                    original_range.file_id,
                    TextEdit::replace(original_range.range, name.display(db).to_string()),
                )),
                trigger_signature_help: false,
            });
        }
    });
    if assists.is_empty() {
        None
    } else {
        Some(assists)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fixes};

    #[test]
    fn unknown() {
        check_diagnostics(
            r#"
fn main() {
    _;
  //^ error: `_` expressions may only appear on the left-hand side of an assignment
}
"#,
        );
    }

    #[test]
    fn concrete_expectation() {
        check_diagnostics(
            r#"
fn main() {
    if _ {}
     //^ error: invalid `_` expression, expected type `bool`
    let _: fn() -> i32 = _;
                       //^ error: invalid `_` expression, expected type `fn() -> i32`
    let _: fn() -> () = _; // FIXME: This should trigger an assist because `main` matches via *coercion*
                      //^ error: invalid `_` expression, expected type `fn()`
}
"#,
        );
    }

    #[test]
    fn integer_ty_var() {
        check_diagnostics(
            r#"
fn main() {
    let mut x = 3;
    x = _;
      //^ 💡 error: invalid `_` expression, expected type `i32`
}
"#,
        );
    }

    #[test]
    fn ty_var_resolved() {
        check_diagnostics(
            r#"
fn main() {
    let mut x = t();
    x = _;
      //^ 💡 error: invalid `_` expression, expected type `&str`
    x = "";
}
fn t<T>() -> T { loop {} }
"#,
        );
    }

    #[test]
    fn valid_positions() {
        check_diagnostics(
            r#"
fn main() {
    let x = [(); _];
    let y: [(); 10] = [(); _];
    _ = 0;
    (_,) = (1,);
}
"#,
        );
    }

    #[test]
    fn check_quick_fix() {
        check_fixes(
            r#"
enum Foo {
    Bar
}
use Foo::Bar;
const C: Foo = Foo::Bar;
fn main<const CP: Foo>(param: Foo) {
    let local = Foo::Bar;
    let _: Foo = _$0;
               //^ error: invalid `_` expression, expected type `fn()`
}
"#,
            vec![
                r#"
enum Foo {
    Bar
}
use Foo::Bar;
const C: Foo = Foo::Bar;
fn main<const CP: Foo>(param: Foo) {
    let local = Foo::Bar;
    let _: Foo = local;
               //^ error: invalid `_` expression, expected type `fn()`
}
"#,
                r#"
enum Foo {
    Bar
}
use Foo::Bar;
const C: Foo = Foo::Bar;
fn main<const CP: Foo>(param: Foo) {
    let local = Foo::Bar;
    let _: Foo = param;
               //^ error: invalid `_` expression, expected type `fn()`
}
"#,
                r#"
enum Foo {
    Bar
}
use Foo::Bar;
const C: Foo = Foo::Bar;
fn main<const CP: Foo>(param: Foo) {
    let local = Foo::Bar;
    let _: Foo = CP;
               //^ error: invalid `_` expression, expected type `fn()`
}
"#,
                r#"
enum Foo {
    Bar
}
use Foo::Bar;
const C: Foo = Foo::Bar;
fn main<const CP: Foo>(param: Foo) {
    let local = Foo::Bar;
    let _: Foo = Bar;
               //^ error: invalid `_` expression, expected type `fn()`
}
"#,
                r#"
enum Foo {
    Bar
}
use Foo::Bar;
const C: Foo = Foo::Bar;
fn main<const CP: Foo>(param: Foo) {
    let local = Foo::Bar;
    let _: Foo = C;
               //^ error: invalid `_` expression, expected type `fn()`
}
"#,
            ],
        );
    }
}
