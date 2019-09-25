use std::iter;

use hir::{db::HirDatabase, Adt, HasSource};
use ra_syntax::ast::{self, AstNode, NameOwner};

use crate::{ast_builder::Make, Assist, AssistCtx, AssistId};

pub(crate) fn fill_match_arms(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let match_expr = ctx.node_at_offset::<ast::MatchExpr>()?;
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
    let enum_def = {
        let file_id = ctx.frange.file_id;
        let analyzer = hir::SourceAnalyzer::new(ctx.db, file_id, expr.syntax(), None);
        resolve_enum_def(ctx.db, &analyzer, &expr)?
    };
    let variant_list = enum_def.variant_list()?;

    ctx.add_action(AssistId("fill_match_arms"), "fill match arms", |edit| {
        let variants = variant_list.variants();
        let arms = variants
            .filter_map(build_pat)
            .map(|pat| Make::<ast::MatchArm>::from(iter::once(pat), Make::<ast::Expr>::unit()));
        let new_arm_list = Make::<ast::MatchArmList>::from_arms(arms);

        edit.target(match_expr.syntax().text_range());
        edit.set_cursor(expr.syntax().text_range().start());
        edit.replace_node_and_indent(match_arm_list.syntax(), new_arm_list.syntax().text());
    });

    ctx.build()
}

fn is_trivial(arm: &ast::MatchArm) -> bool {
    arm.pats().any(|pat| match pat {
        ast::Pat::PlaceholderPat(..) => true,
        _ => false,
    })
}

fn resolve_enum_def(
    db: &impl HirDatabase,
    analyzer: &hir::SourceAnalyzer,
    expr: &ast::Expr,
) -> Option<ast::EnumDef> {
    let expr_ty = analyzer.type_of(db, &expr)?;

    analyzer.autoderef(db, expr_ty).find_map(|ty| match ty.as_adt() {
        Some((Adt::Enum(e), _)) => Some(e.source(db).ast),
        _ => None,
    })
}

fn build_pat(var: ast::EnumVariant) -> Option<ast::Pat> {
    let path = Make::<ast::Path>::from(var.parent_enum().name()?, var.name()?);

    let pat: ast::Pat = match var.kind() {
        ast::StructKind::Tuple(field_list) => {
            let pats = iter::repeat(Make::<ast::PlaceholderPat>::placeholder().into())
                .take(field_list.fields().count());
            Make::<ast::TupleStructPat>::from(path, pats).into()
        }
        ast::StructKind::Named(field_list) => {
            let pats = field_list
                .fields()
                .map(|f| Make::<ast::BindPat>::from_name(f.name().unwrap()).into());
            Make::<ast::RecordPat>::from(path, pats).into()
        }
        ast::StructKind::Unit => Make::<ast::PathPat>::from_path(path).into(),
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
                    A::Es{ x, y } => (),
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
                    A::Es{ x, y } => (),
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
}
