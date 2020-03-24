//! FIXME: write short doc here

use std::iter;

use hir::{Adt, HasSource, ModuleDef, Semantics};
use itertools::Itertools;
use ra_ide_db::RootDatabase;

use crate::{Assist, AssistCtx, AssistId};
use ra_syntax::ast::{self, edit::IndentLevel, make, AstNode, NameOwner};

use ast::{MatchArm, Pat};

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
//         Action::Move { distance } => {}
//         Action::Stop => {}
//     }
// }
// ```
pub(crate) fn fill_match_arms(ctx: AssistCtx) -> Option<Assist> {
    let match_expr = ctx.find_node_at_offset::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;

    let expr = match_expr.expr()?;

    let mut arms: Vec<MatchArm> = match_arm_list.arms().collect();
    if arms.len() == 1 {
        if let Some(Pat::PlaceholderPat(..)) = arms[0].pat() {
            arms.clear();
        }
    }

    let module = ctx.sema.scope(expr.syntax()).module()?;

    let missing_arms: Vec<MatchArm> = if let Some(enum_def) = resolve_enum_def(&ctx.sema, &expr) {
        let variants = enum_def.variants(ctx.db);

        variants
            .into_iter()
            .filter_map(|variant| build_pat(ctx.db, module, variant))
            .filter(|variant_pat| is_variant_missing(&mut arms, variant_pat))
            .map(|pat| make::match_arm(iter::once(pat), make::expr_empty_block()))
            .collect()
    } else if let Some(enum_defs) = resolve_tuple_of_enum_def(&ctx.sema, &expr) {
        // Partial fill not currently supported for tuple of enums.
        if !arms.is_empty() {
            return None;
        }

        // We do not currently support filling match arms for a tuple
        // containing a single enum.
        if enum_defs.len() < 2 {
            return None;
        }

        // When calculating the match arms for a tuple of enums, we want
        // to create a match arm for each possible combination of enum
        // values. The `multi_cartesian_product` method transforms
        // Vec<Vec<EnumVariant>> into Vec<(EnumVariant, .., EnumVariant)>
        // where each tuple represents a proposed match arm.
        enum_defs
            .into_iter()
            .map(|enum_def| enum_def.variants(ctx.db))
            .multi_cartesian_product()
            .map(|variants| {
                let patterns =
                    variants.into_iter().filter_map(|variant| build_pat(ctx.db, module, variant));
                ast::Pat::from(make::tuple_pat(patterns))
            })
            .filter(|variant_pat| is_variant_missing(&mut arms, variant_pat))
            .map(|pat| make::match_arm(iter::once(pat), make::expr_empty_block()))
            .collect()
    } else {
        return None;
    };

    if missing_arms.is_empty() {
        return None;
    }

    ctx.add_assist(AssistId("fill_match_arms"), "Fill match arms", |edit| {
        arms.extend(missing_arms);

        let indent_level = IndentLevel::from_node(match_arm_list.syntax());
        let new_arm_list = indent_level.increase_indent(make::match_arm_list(arms));

        edit.target(match_expr.syntax().text_range());
        edit.set_cursor(expr.syntax().text_range().start());
        edit.replace_ast(match_arm_list, new_arm_list);
    })
}

fn is_variant_missing(existing_arms: &mut Vec<MatchArm>, var: &Pat) -> bool {
    existing_arms.iter().filter_map(|arm| arm.pat()).all(|pat| {
        // Special casee OrPat as separate top-level pats
        let top_level_pats: Vec<Pat> = match pat {
            Pat::OrPat(pats) => pats.pats().collect::<Vec<_>>(),
            _ => vec![pat],
        };

        !top_level_pats.iter().any(|pat| does_pat_match_variant(pat, var))
    })
}

fn does_pat_match_variant(pat: &Pat, var: &Pat) -> bool {
    let pat_head = pat.syntax().first_child().map(|node| node.text());
    let var_head = var.syntax().first_child().map(|node| node.text());

    pat_head == var_head
}

fn resolve_enum_def(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<hir::Enum> {
    sema.type_of_expr(&expr)?.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
        Some(Adt::Enum(e)) => Some(e),
        _ => None,
    })
}

fn resolve_tuple_of_enum_def(
    sema: &Semantics<RootDatabase>,
    expr: &ast::Expr,
) -> Option<Vec<hir::Enum>> {
    sema.type_of_expr(&expr)?
        .tuple_fields(sema.db)
        .iter()
        .map(|ty| {
            ty.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
                Some(Adt::Enum(e)) => Some(e),
                // For now we only handle expansion for a tuple of enums. Here
                // we map non-enum items to None and rely on `collect` to
                // convert Vec<Option<hir::Enum>> into Option<Vec<hir::Enum>>.
                _ => None,
            })
        })
        .collect()
}

fn build_pat(db: &RootDatabase, module: hir::Module, var: hir::EnumVariant) -> Option<ast::Pat> {
    let path = crate::ast_transform::path_to_ast(module.find_use_path(db, ModuleDef::from(var))?);

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
    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::fill_match_arms;

    #[test]
    fn all_match_arms_provided() {
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs{x:i32, y:Option<i32>},
                Cs(i32, Option<i32>),
            }
            fn main() {
                match A::As<|> {
                    A::As,
                    A::Bs{x,y:Some(_)} => {}
                    A::Cs(_, Some(_)) => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn tuple_of_non_enum() {
        // for now this case is not handled, although it potentially could be
        // in the future
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            fn main() {
                match (0, false)<|> {
                }
            }
            "#,
        );
    }

    #[test]
    fn partial_fill_record_tuple() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                As,
                Bs{x:i32, y:Option<i32>},
                Cs(i32, Option<i32>),
            }
            fn main() {
                match A::As<|> {
                    A::Bs{x,y:Some(_)} => {}
                    A::Cs(_, Some(_)) => {}
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs{x:i32, y:Option<i32>},
                Cs(i32, Option<i32>),
            }
            fn main() {
                match <|>A::As {
                    A::Bs{x,y:Some(_)} => {}
                    A::Cs(_, Some(_)) => {}
                    A::As => {}
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
                    A::Cs(_) | A::Bs => {}
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
                    A::Cs(_) | A::Bs => {}
                    A::As => {}
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
                    A::Bs if 0 < 1 => {}
                    A::Ds(_value) => { let x = 1; }
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
                    A::Bs if 0 < 1 => {}
                    A::Ds(_value) => { let x = 1; }
                    A::Es(B::Xs) => (),
                    A::As => {}
                    A::Cs => {}
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
                    A::As => {}
                    A::Bs => {}
                    A::Cs(_) => {}
                    A::Ds(_, _) => {}
                    A::Es { x, y } => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_tuple_of_enum() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                One,
                Two,
            }
            enum B {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (a<|>, b) {}
            }
            "#,
            r#"
            enum A {
                One,
                Two,
            }
            enum B {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                let b = B::One;
                match <|>(a, b) {
                    (A::One, B::One) => {}
                    (A::One, B::Two) => {}
                    (A::Two, B::One) => {}
                    (A::Two, B::Two) => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_tuple_of_enum_ref() {
        check_assist(
            fill_match_arms,
            r#"
            enum A {
                One,
                Two,
            }
            enum B {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (&a<|>, &b) {}
            }
            "#,
            r#"
            enum A {
                One,
                Two,
            }
            enum B {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                let b = B::One;
                match <|>(&a, &b) {
                    (A::One, B::One) => {}
                    (A::One, B::Two) => {}
                    (A::Two, B::One) => {}
                    (A::Two, B::Two) => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_tuple_of_enum_partial() {
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            enum A {
                One,
                Two,
            }
            enum B {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (a<|>, b) {
                    (A::Two, B::One) => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_tuple_of_enum_not_applicable() {
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            enum A {
                One,
                Two,
            }
            enum B {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (a<|>, b) {
                    (A::Two, B::One) => {}
                    (A::One, B::One) => {}
                    (A::One, B::Two) => {}
                    (A::Two, B::Two) => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_single_element_tuple_of_enum() {
        // For now we don't hande the case of a single element tuple, but
        // we could handle this in the future if `make::tuple_pat` allowed
        // creating a tuple with a single pattern.
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            enum A {
                One,
                Two,
            }

            fn main() {
                let a = A::One;
                match (a<|>, ) {
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
                    A::As => {}
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
                    A::Es { x, y } => {}
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
                    <|>_ => {}
                }
            }
            "#,
            r#"
            enum E { X, Y }

            fn main() {
                match <|>E::X {
                    E::X => {}
                    E::Y => {}
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
                    X => {}
                    foo::E::Y => {}
                }
            }
            "#,
        );
    }
}
