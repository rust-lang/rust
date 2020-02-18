//! FIXME: write short doc here

use std::iter;

use hir::{db::HirDatabase, Adt, HasSource, Semantics};
use ra_syntax::ast::{self, edit::IndentLevel, make, AstNode, NameOwner};

use crate::{Assist, AssistCtx, AssistId};
use ra_ide_db::RootDatabase;

// Assist: fill_match_arms
//
// Adds missing clauses to a `match` expression.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         <|>
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => (),
//         Action::Stop => (),
//     }
// }
// ```
pub(crate) fn fill_match_arms(ctx: AssistCtx) -> Option<Assist> {
    let match_expr = ctx.find_node_at_offset::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;

    // We already have some match arms, so we don't provide any assists.
    // Unless if there is only one trivial match arm possibly created
    // by match postfix complete. Trivial match arm is the catch all arm.
    let mut existing_arms = match_arm_list.arms();
    if let Some(arm) = existing_arms.next() {
        if !is_trivial(&arm) || existing_arms.next().is_some() {
            return None;
        }
    };

    let expr = match_expr.expr()?;
    let enum_def = resolve_enum_def(&ctx.sema, &expr)?;
    let module = ctx.sema.scope(expr.syntax()).module()?;

    let variants = enum_def.variants(ctx.db);
    if variants.is_empty() {
        return None;
    }

    let db = ctx.db;

    ctx.add_assist(AssistId("fill_match_arms"), "Fill match arms", |edit| {
        let indent_level = IndentLevel::from_node(match_arm_list.syntax());

        let new_arm_list = {
            let arms = variants
                .into_iter()
                .filter_map(|variant| build_pat(db, module, variant))
                .map(|pat| make::match_arm(iter::once(pat), make::expr_unit()));
            indent_level.increase_indent(make::match_arm_list(arms))
        };

        edit.target(match_expr.syntax().text_range());
        edit.set_cursor(expr.syntax().text_range().start());
        edit.replace_ast(match_arm_list, new_arm_list);
    })
}

fn is_trivial(arm: &ast::MatchArm) -> bool {
    match arm.pat() {
        Some(ast::Pat::PlaceholderPat(..)) => true,
        _ => false,
    }
}

fn resolve_enum_def(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<hir::Enum> {
    sema.type_of_expr(&expr)?.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
        Some(Adt::Enum(e)) => Some(e),
        _ => None,
    })
}

fn build_pat(
    db: &impl HirDatabase,
    module: hir::Module,
    var: hir::EnumVariant,
) -> Option<ast::Pat> {
    let path = crate::ast_transform::path_to_ast(module.find_use_path(db, var.into())?);

    // FIXME: use HIR for this; it doesn't currently expose struct vs. tuple vs. unit variants though
    let pat: ast::Pat = match var.source(db).value.kind() {
        ast::StructKind::Tuple(field_list) => {
            let pats =
                iter::repeat(make::placeholder_pat().into()).take(field_list.fields().count());
            make::tuple_struct_pat(path, pats).into()
        }
        ast::StructKind::Record(field_list) => {
            let pats = field_list.fields().map(|f| make::bind_pat(f.name().unwrap()).into());
            make::record_pat(path, pats).into()
        }
        ast::StructKind::Unit => make::path_pat(path),
    };

    Some(pat)
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_target};

    use super::fill_match_arms;

    #[test]
    fn fill_match_arms_empty_body() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs,
                Cs(String),
                Ds(String, String),
                Es{ x: usize, y: usize }
            }

            fn main() {
                let a = A::As;
                match a<|> {}
            }
            "#,
            r#"
            enum A {
                As,
                Bs,
                Cs(String),
                Ds(String, String),
                Es{ x: usize, y: usize }
            }

            fn main() {
                let a = A::As;
                match <|>a {
                    A::As => (),
                    A::Bs => (),
                    A::Cs(_) => (),
                    A::Ds(_, _) => (),
                    A::Es { x, y } => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn test_fill_match_arm_refs() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
            }

            fn foo(a: &A) {
                match a<|> {
                }
            }
            "#,
            r#"
            enum A {
                As,
            }

            fn foo(a: &A) {
                match <|>a {
                    A::As => (),
                }
            }
            "#,
        );

        check_assist(
            fill_match_arms,
            r#"
            enum A {
                Es{ x: usize, y: usize }
            }

            fn foo(a: &mut A) {
                match a<|> {
                }
            }
            "#,
            r#"
            enum A {
                Es{ x: usize, y: usize }
            }

            fn foo(a: &mut A) {
                match <|>a {
                    A::Es { x, y } => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_target() {
        check_assist_target(
            fill_match_arms,
            r#"
            enum E { X, Y }

            fn main() {
                match E::X<|> {}
            }
            "#,
            "match E::X {}",
        );
    }

    #[test]
    fn fill_match_arms_trivial_arm() {
        check_assist(
            fill_match_arms,
            r#"
            enum E { X, Y }

            fn main() {
                match E::X {
                    <|>_ => {},
                }
            }
            "#,
            r#"
            enum E { X, Y }

            fn main() {
                match <|>E::X {
                    E::X => (),
                    E::Y => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_qualifies_path() {
        check_assist(
            fill_match_arms,
            r#"
            mod foo { pub enum E { X, Y } }
            use foo::E::X;

            fn main() {
                match X {
                    <|>
                }
            }
            "#,
            r#"
            mod foo { pub enum E { X, Y } }
            use foo::E::X;

            fn main() {
                match <|>X {
                    X => (),
                    foo::E::Y => (),
                }
            }
            "#,
        );
    }
}
