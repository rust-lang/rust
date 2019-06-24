use std::fmt::Write;

use hir::{
    AdtDef, FieldSource, HasSource,
    db::HirDatabase,
};
use ra_syntax::ast::{self, AstNode};

use crate::{AssistCtx, Assist, AssistId};

fn is_trivial_arm(arm: &ast::MatchArm) -> bool {
    for (i, p) in arm.pats().enumerate() {
        if i > 0 {
            return false;
        }

        match p.kind() {
            ast::PatKind::PlaceholderPat(_) => {}
            _ => {
                return false;
            }
        };
    }
    return true;
}

pub(crate) fn fill_match_arms(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let match_expr = ctx.node_at_offset::<ast::MatchExpr>()?;

    // We already have some match arms, so we don't provide any assists.
    // Unless if there is only one trivial match arm possibly created
    // by match postfix complete. Trivial match arm is the catch all arm.
    match match_expr.match_arm_list() {
        Some(arm_list) => {
            for (i, a) in arm_list.arms().enumerate() {
                if i > 0 {
                    return None;
                }

                if !is_trivial_arm(a) {
                    return None;
                }
            }
        }
        _ => {}
    }

    let expr = match_expr.expr()?;
    let analyzer = hir::SourceAnalyzer::new(ctx.db, ctx.frange.file_id, expr.syntax(), None);
    let match_expr_ty = analyzer.type_of(ctx.db, expr)?;
    let enum_def = analyzer.autoderef(ctx.db, match_expr_ty).find_map(|ty| match ty.as_adt() {
        Some((AdtDef::Enum(e), _)) => Some(e),
        _ => None,
    })?;
    let enum_name = enum_def.name(ctx.db)?;
    let db = ctx.db;

    ctx.add_action(AssistId("fill_match_arms"), "fill match arms", |edit| {
        let mut buf = format!("match {} {{\n", expr.syntax().text().to_string());
        let variants = enum_def.variants(db);
        for variant in variants {
            let name = match variant.name(db) {
                Some(it) => it,
                None => continue,
            };
            write!(&mut buf, "    {}::{}", enum_name, name.to_string()).unwrap();

            let pat = variant
                .fields(db)
                .into_iter()
                .map(|field| {
                    let name = field.name(db).to_string();
                    let src = field.source(db);
                    match src.ast {
                        FieldSource::Named(_) => name,
                        FieldSource::Pos(_) => "_".to_string(),
                    }
                })
                .collect::<Vec<_>>();

            match pat.first().map(|s| s.as_str()) {
                Some("_") => write!(&mut buf, "({})", pat.join(", ")).unwrap(),
                Some(_) => write!(&mut buf, "{{{}}}", pat.join(", ")).unwrap(),
                None => (),
            };

            buf.push_str(" => (),\n");
        }
        buf.push_str("}");
        edit.target(match_expr.syntax().range());
        edit.set_cursor(expr.syntax().range().start());
        edit.replace_node_and_indent(match_expr.syntax(), buf);
    });

    ctx.build()
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
                Es{x: usize, y: usize}
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
                Es{x: usize, y: usize}
            }

            fn main() {
                let a = A::As;
                match <|>a {
                    A::As => (),
                    A::Bs => (),
                    A::Cs(_) => (),
                    A::Ds(_, _) => (),
                    A::Es{x, y} => (),
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
                Es{x: usize, y: usize}
            }

            fn foo(a: &mut A) {
                match a<|> {
                }
            }
            "#,
            r#"
            enum A {
                Es{x: usize, y: usize}
            }

            fn foo(a: &mut A) {
                match <|>a {
                    A::Es{x, y} => (),
                }
            }
            "#,
        );

        check_assist(
            fill_match_arms,
            r#"
            enum E { X, Y}

            fn main() {
                match &E::X<|>
            }
            "#,
            r#"
            enum E { X, Y}

            fn main() {
                match <|>&E::X {
                    E::X => (),
                    E::Y => (),
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_no_body() {
        check_assist(
            fill_match_arms,
            r#"
            enum E { X, Y}

            fn main() {
                match E::X<|>
            }
            "#,
            r#"
            enum E { X, Y}

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
    fn fill_match_arms_target() {
        check_assist_target(
            fill_match_arms,
            r#"
            enum E { X, Y}

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
