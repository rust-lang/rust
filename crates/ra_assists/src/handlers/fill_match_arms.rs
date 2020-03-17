//! FIXME: write short doc here

use std::{collections::LinkedList, iter};

use hir::{Adt, HasSource, Semantics};
use ra_ide_db::RootDatabase;

use crate::{Assist, AssistCtx, AssistId};
use ra_syntax::{
    ast::{self, edit::IndentLevel, make, AstNode, NameOwner},
    SyntaxKind, SyntaxNode,
};

use ast::{MatchArm, MatchGuard, Pat};

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

    let expr = match_expr.expr()?;
    let enum_def = resolve_enum_def(&ctx.sema, &expr)?;
    let module = ctx.sema.scope(expr.syntax()).module()?;

    let variants = enum_def.variants(ctx.db);
    if variants.is_empty() {
        return None;
    }

    let mut arms: Vec<MatchArm> = match_arm_list.arms().collect();
    if arms.len() == 1 {
        if let Some(Pat::PlaceholderPat(..)) = arms[0].pat() {
            arms.clear();
        }
    }

    let mut has_partial_match = false;
    let db = ctx.db;
    let missing_arms: Vec<MatchArm> = variants
        .into_iter()
        .filter_map(|variant| build_pat(db, module, variant))
        .filter(|variant_pat| {
            !arms.iter().filter_map(|arm| arm.pat().map(|_| arm)).any(|arm| {
                let pat = arm.pat().unwrap();

                // Special casee OrPat as separate top-level pats
                let pats: Vec<Pat> = match Pat::from(pat.clone()) {
                    Pat::OrPat(pats) => pats.pats().collect::<Vec<_>>(),
                    _ => vec![pat],
                };

                pats.iter().any(|pat| {
                    match does_arm_pat_match_variant(pat, arm.guard(), variant_pat) {
                        ArmMatch::Yes => true,
                        ArmMatch::No => false,
                        ArmMatch::Partial => {
                            has_partial_match = true;
                            true
                        }
                    }
                })
            })
        })
        .map(|pat| make::match_arm(iter::once(pat), make::expr_unit()))
        .collect();

    if missing_arms.is_empty() && !has_partial_match {
        return None;
    }

    ctx.add_assist(AssistId("fill_match_arms"), "Fill match arms", |edit| {
        arms.extend(missing_arms);
        if has_partial_match {
            arms.push(make::match_arm(
                iter::once(make::placeholder_pat().into()),
                make::expr_unit(),
            ));
        }

        let indent_level = IndentLevel::from_node(match_arm_list.syntax());
        let new_arm_list = indent_level.increase_indent(make::match_arm_list(arms));

        edit.target(match_expr.syntax().text_range());
        edit.set_cursor(expr.syntax().text_range().start());
        edit.replace_ast(match_arm_list, new_arm_list);
    })
}

enum ArmMatch {
    Yes,
    No,
    Partial,
}

fn does_arm_pat_match_variant(arm: &Pat, arm_guard: Option<MatchGuard>, var: &Pat) -> ArmMatch {
    let arm = flatten_pats(arm.clone());
    let var = flatten_pats(var.clone());
    let mut arm = arm.iter();
    let mut var = var.iter();

    // If the first part of the Pat don't match, there's no match
    match (arm.next(), var.next()) {
        (Some(arm), Some(var)) if arm.text() == var.text() => {}
        _ => return ArmMatch::No,
    }

    // If we have a guard we automatically know we have a partial match
    if arm_guard.is_some() {
        return ArmMatch::Partial;
    }

    if arm.clone().count() != var.clone().count() {
        return ArmMatch::Partial;
    }

    let direct_match = arm.zip(var).all(|(arm, var)| {
        if arm.text() == var.text() {
            return true;
        }
        match (arm.kind(), var.kind()) {
            (SyntaxKind::PLACEHOLDER_PAT, SyntaxKind::PLACEHOLDER_PAT) => true,
            (SyntaxKind::DOT_DOT_PAT, SyntaxKind::PLACEHOLDER_PAT) => true,
            (SyntaxKind::BIND_PAT, SyntaxKind::PLACEHOLDER_PAT) => true,
            _ => false,
        }
    });

    match direct_match {
        true => ArmMatch::Yes,
        false => ArmMatch::Partial,
    }
}

fn flatten_pats(pat: Pat) -> Vec<SyntaxNode> {
    let mut pats: LinkedList<SyntaxNode> = pat.syntax().children().collect();
    let mut out: Vec<SyntaxNode> = vec![];
    while let Some(p) = pats.pop_front() {
        pats.extend(p.children());
        out.push(p);
    }
    out
}

fn resolve_enum_def(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<hir::Enum> {
    sema.type_of_expr(&expr)?.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
        Some(Adt::Enum(e)) => Some(e),
        _ => None,
    })
}

fn build_pat(db: &RootDatabase, module: hir::Module, var: hir::EnumVariant) -> Option<ast::Pat> {
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
    fn partial_fill_multi() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs(i32, Option<i32>)
            }
            fn main() {
                match A::As<|> {
                    A::Bs(_, Some(_)) => (),
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs(i32, Option<i32>)
            }
            fn main() {
                match <|>A::As {
                    A::Bs(_, Some(_)) => (),
                    A::As => (),
                    _ => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn partial_fill_record() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs{x:i32, y:Option<i32>},
            }
            fn main() {
                match A::As<|> {
                    A::Bs{x,y:Some(_)} => (),
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs{x:i32, y:Option<i32>},
            }
            fn main() {
                match <|>A::As {
                    A::Bs{x,y:Some(_)} => (),
                    A::As => (),
                    _ => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn partial_fill_or_pat() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs,
                Cs(Option<i32>),
            }
            fn main() {
                match A::As<|> {
                    A::Cs(_) | A::Bs => (),
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs,
                Cs(Option<i32>),
            }
            fn main() {
                match <|>A::As {
                    A::Cs(_) | A::Bs => (),
                    A::As => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn partial_fill_or_pat2() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs,
                Cs(Option<i32>),
            }
            fn main() {
                match A::As<|> {
                    A::Cs(Some(_)) | A::Bs => (),
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs,
                Cs(Option<i32>),
            }
            fn main() {
                match <|>A::As {
                    A::Cs(Some(_)) | A::Bs => (),
                    A::As => (),
                    _ => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn partial_fill() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs,
                Cs,
                Ds(String),
                Es(B),
            }
            enum B {
                Xs,
                Ys,
            }
            fn main() {
                match A::As<|> {
                    A::Bs if 0 < 1 => (),
                    A::Ds(_value) => (),
                    A::Es(B::Xs) => (),
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs,
                Cs,
                Ds(String),
                Es(B),
            }
            enum B {
                Xs,
                Ys,
            }
            fn main() {
                match <|>A::As {
                    A::Bs if 0 < 1 => (),
                    A::Ds(_value) => (),
                    A::Es(B::Xs) => (),
                    A::As => (),
                    A::Cs => (),
                    _ => (),
                }
            }
            "#,
        );
    }

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
