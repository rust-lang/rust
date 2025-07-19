use hir::HirDisplay;
use ide_db::{assists::AssistId, defs::Definition};
use stdx::to_upper_snake_case;
use syntax::{
    AstNode,
    ast::{self, HasName, syntax_factory::SyntaxFactory},
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::{self},
};

// Assist: promote_local_to_const
//
// Promotes a local variable to a const item changing its name to a `SCREAMING_SNAKE_CASE` variant
// if the local uses no non-const expressions.
//
// ```
// fn main() {
//     let foo$0 = true;
//
//     if foo {
//         println!("It's true");
//     } else {
//         println!("It's false");
//     }
// }
// ```
// ->
// ```
// fn main() {
//     const $0FOO: bool = true;
//
//     if FOO {
//         println!("It's true");
//     } else {
//         println!("It's false");
//     }
// }
// ```
pub(crate) fn promote_local_to_const(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let pat = ctx.find_node_at_offset::<ast::IdentPat>()?;
    let name = pat.name()?;
    if !pat.is_simple_ident() {
        cov_mark::hit!(promote_local_non_simple_ident);
        return None;
    }
    let let_stmt = pat.syntax().parent().and_then(ast::LetStmt::cast)?;

    let module = ctx.sema.scope(pat.syntax())?.module();
    let local = ctx.sema.to_def(&pat)?;
    let ty = ctx.sema.type_of_pat(&pat.into())?.original;

    let ty = match ty.display_source_code(ctx.db(), module.into(), false) {
        Ok(ty) => ty,
        Err(_) => return None,
    };

    let initializer = let_stmt.initializer()?;
    if !utils::is_body_const(&ctx.sema, &initializer) {
        cov_mark::hit!(promote_local_non_const);
        return None;
    }

    acc.add(
        AssistId::refactor("promote_local_to_const"),
        "Promote local to constant",
        let_stmt.syntax().text_range(),
        |edit| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = edit.make_editor(let_stmt.syntax());
            let name = to_upper_snake_case(&name.to_string());
            let usages = Definition::Local(local).usages(&ctx.sema).all();
            if let Some(usages) = usages.references.get(&ctx.file_id()) {
                let name_ref = make.name_ref(&name);

                for usage in usages {
                    let Some(usage_name) = usage.name.as_name_ref().cloned() else { continue };
                    if let Some(record_field) = ast::RecordExprField::for_name_ref(&usage_name) {
                        let path = make.ident_path(&name);
                        let name_expr = make.expr_path(path);
                        utils::replace_record_field_expr(ctx, edit, record_field, name_expr);
                    } else {
                        let usage_range = usage.range;
                        edit.replace(usage_range, name_ref.syntax().text());
                    }
                }
            }

            let item = make.item_const(None, make.name(&name), make.ty(&ty), initializer);

            if let Some((cap, name)) = ctx.config.snippet_cap.zip(item.name()) {
                let tabstop = edit.make_tabstop_before(cap);
                editor.add_annotation(name.syntax().clone(), tabstop);
            }

            editor.replace(let_stmt.syntax(), item.syntax());

            editor.add_mappings(make.finish_with_mappings());
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn simple() {
        check_assist(
            promote_local_to_const,
            r"
fn foo() {
    let x$0 = 0;
    let y = x;
}
",
            r"
fn foo() {
    const $0X: i32 = 0;
    let y = X;
}
",
        );
    }

    #[test]
    fn multiple_uses() {
        check_assist(
            promote_local_to_const,
            r"
fn foo() {
    let x$0 = 0;
    let y = x;
    let z = (x, x, x, x);
}
",
            r"
fn foo() {
    const $0X: i32 = 0;
    let y = X;
    let z = (X, X, X, X);
}
",
        );
    }

    #[test]
    fn usage_in_field_shorthand() {
        check_assist(
            promote_local_to_const,
            r"
struct Foo {
    bar: usize,
}

fn main() {
    let $0bar = 0;
    let foo = Foo { bar };
}
",
            r"
struct Foo {
    bar: usize,
}

fn main() {
    const $0BAR: usize = 0;
    let foo = Foo { bar: BAR };
}
",
        )
    }

    #[test]
    fn usage_in_macro() {
        check_assist(
            promote_local_to_const,
            r"
macro_rules! identity {
    ($body:expr) => {
        $body
    }
}

fn baz() -> usize {
    let $0foo = 2;
    identity![foo]
}
",
            r"
macro_rules! identity {
    ($body:expr) => {
        $body
    }
}

fn baz() -> usize {
    const $0FOO: usize = 2;
    identity![FOO]
}
",
        )
    }

    #[test]
    fn usage_shorthand_in_macro() {
        check_assist(
            promote_local_to_const,
            r"
struct Foo {
    foo: usize,
}

macro_rules! identity {
    ($body:expr) => {
        $body
    };
}

fn baz() -> Foo {
    let $0foo = 2;
    identity![Foo { foo }]
}
",
            r"
struct Foo {
    foo: usize,
}

macro_rules! identity {
    ($body:expr) => {
        $body
    };
}

fn baz() -> Foo {
    const $0FOO: usize = 2;
    identity![Foo { foo: FOO }]
}
",
        )
    }

    #[test]
    fn not_applicable_non_const_meth_call() {
        cov_mark::check!(promote_local_non_const);
        check_assist_not_applicable(
            promote_local_to_const,
            r"
struct Foo;
impl Foo {
    fn foo(self) {}
}
fn foo() {
    let x$0 = Foo.foo();
}
",
        );
    }

    #[test]
    fn not_applicable_non_const_call() {
        check_assist_not_applicable(
            promote_local_to_const,
            r"
fn bar(self) {}
fn foo() {
    let x$0 = bar();
}
",
        );
    }

    #[test]
    fn not_applicable_unknown_ty() {
        check_assist(
            promote_local_to_const,
            r"
fn foo() {
    let x$0 = bar();
}
",
            r"
fn foo() {
    const $0X: _ = bar();
}
",
        );
    }

    #[test]
    fn not_applicable_non_simple_ident() {
        cov_mark::check!(promote_local_non_simple_ident);
        check_assist_not_applicable(
            promote_local_to_const,
            r"
fn foo() {
    let ref x$0 = ();
}
",
        );
        check_assist_not_applicable(
            promote_local_to_const,
            r"
fn foo() {
    let mut x$0 = ();
}
",
        );
    }
}
