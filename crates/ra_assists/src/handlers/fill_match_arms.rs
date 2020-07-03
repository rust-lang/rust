use std::iter;

use hir::{Adt, HasSource, ModuleDef, Semantics};
use itertools::Itertools;
use ra_ide_db::RootDatabase;
use ra_syntax::ast::{self, make, AstNode, MatchArm, NameOwner, Pat};
use test_utils::mark;

use crate::{
    utils::{render_snippet, Cursor, FamousDefs},
    AssistContext, AssistId, AssistKind, Assists,
};

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
//         $0Action::Move { distance } => {}
//         Action::Stop => {}
//     }
// }
// ```
pub(crate) fn fill_match_arms(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
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
        let variants = enum_def.variants(ctx.db());

        let mut variants = variants
            .into_iter()
            .filter_map(|variant| build_pat(ctx.db(), module, variant))
            .filter(|variant_pat| is_variant_missing(&mut arms, variant_pat))
            .map(|pat| make::match_arm(iter::once(pat), make::expr_empty_block()))
            .collect::<Vec<_>>();
        if Some(enum_def) == FamousDefs(&ctx.sema, module.krate()).core_option_Option() {
            // Match `Some` variant first.
            mark::hit!(option_order);
            variants.reverse()
        }
        variants
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
            .map(|enum_def| enum_def.variants(ctx.db()))
            .multi_cartesian_product()
            .map(|variants| {
                let patterns =
                    variants.into_iter().filter_map(|variant| build_pat(ctx.db(), module, variant));
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

    let target = match_expr.syntax().text_range();
    acc.add(
        AssistId("fill_match_arms", AssistKind::QuickFix),
        "Fill match arms",
        target,
        |builder| {
            let new_arm_list = match_arm_list.remove_placeholder();
            let n_old_arms = new_arm_list.arms().count();
            let new_arm_list = new_arm_list.append_arms(missing_arms);
            let first_new_arm = new_arm_list.arms().nth(n_old_arms);
            let old_range = match_arm_list.syntax().text_range();
            match (first_new_arm, ctx.config.snippet_cap) {
                (Some(first_new_arm), Some(cap)) => {
                    let snippet = render_snippet(
                        cap,
                        new_arm_list.syntax(),
                        Cursor::Before(first_new_arm.syntax()),
                    );
                    builder.replace_snippet(cap, old_range, snippet);
                }
                _ => builder.replace(old_range, new_arm_list.to_string()),
            }
        },
    )
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
    let first_node_text = |pat: &Pat| pat.syntax().first_child().map(|node| node.text());

    let pat_head = match pat {
        Pat::BindPat(bind_pat) => {
            if let Some(p) = bind_pat.pat() {
                first_node_text(&p)
            } else {
                return false;
            }
        }
        pat => first_node_text(pat),
    };

    let var_head = first_node_text(var);

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
    use test_utils::mark;

    use crate::{
        tests::{check_assist, check_assist_not_applicable, check_assist_target},
        utils::FamousDefs,
    };

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
                Bs { x: i32, y: Option<i32> },
                Cs(i32, Option<i32>),
            }
            fn main() {
                match A::As<|> {
                    A::Bs { x, y: Some(_) } => {}
                    A::Cs(_, Some(_)) => {}
                }
            }
            "#,
            r#"
            enum A {
                As,
                Bs { x: i32, y: Option<i32> },
                Cs(i32, Option<i32>),
            }
            fn main() {
                match A::As {
                    A::Bs { x, y: Some(_) } => {}
                    A::Cs(_, Some(_)) => {}
                    $0A::As => {}
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
                match A::As {
                    A::Cs(_) | A::Bs => {}
                    $0A::As => {}
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
                match A::As {
                    A::Bs if 0 < 1 => {}
                    A::Ds(_value) => { let x = 1; }
                    A::Es(B::Xs) => (),
                    $0A::As => {}
                    A::Cs => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn partial_fill_bind_pat() {
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
                    A::As(_) => {}
                    a @ A::Bs(_) => {}
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
                match A::As {
                    A::As(_) => {}
                    a @ A::Bs(_) => {}
                    $0A::Cs(_) => {}
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
                Es { x: usize, y: usize }
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
                Es { x: usize, y: usize }
            }

            fn main() {
                let a = A::As;
                match a {
                    $0A::As => {}
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
            enum A { One, Two }
            enum B { One, Two }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (a<|>, b) {}
            }
            "#,
            r#"
            enum A { One, Two }
            enum B { One, Two }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (a, b) {
                    $0(A::One, B::One) => {}
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
            enum A { One, Two }
            enum B { One, Two }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (&a<|>, &b) {}
            }
            "#,
            r#"
            enum A { One, Two }
            enum B { One, Two }

            fn main() {
                let a = A::One;
                let b = B::One;
                match (&a, &b) {
                    $0(A::One, B::One) => {}
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
            enum A { One, Two }
            enum B { One, Two }

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
            enum A { One, Two }
            enum B { One, Two }

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
            enum A { One, Two }

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
            enum A { As }

            fn foo(a: &A) {
                match a<|> {
                }
            }
            "#,
            r#"
            enum A { As }

            fn foo(a: &A) {
                match a {
                    $0A::As => {}
                }
            }
            "#,
        );

        check_assist(
            fill_match_arms,
            r#"
            enum A {
                Es { x: usize, y: usize }
            }

            fn foo(a: &mut A) {
                match a<|> {
                }
            }
            "#,
            r#"
            enum A {
                Es { x: usize, y: usize }
            }

            fn foo(a: &mut A) {
                match a {
                    $0A::Es { x, y } => {}
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
                match E::X {
                    $0E::X => {}
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
                match X {
                    $0X => {}
                    foo::E::Y => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_preserves_comments() {
        check_assist(
            fill_match_arms,
            r#"
            enum A { One, Two }
            fn foo(a: A) {
                match a {
                    // foo bar baz<|>
                    A::One => {}
                    // This is where the rest should be
                }
            }
            "#,
            r#"
            enum A { One, Two }
            fn foo(a: A) {
                match a {
                    // foo bar baz
                    A::One => {}
                    // This is where the rest should be
                    $0A::Two => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_preserves_comments_empty() {
        check_assist(
            fill_match_arms,
            r#"
            enum A { One, Two }
            fn foo(a: A) {
                match a {
                    // foo bar baz<|>
                }
            }
            "#,
            r#"
            enum A { One, Two }
            fn foo(a: A) {
                match a {
                    // foo bar baz
                    $0A::One => {}
                    A::Two => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_placeholder() {
        check_assist(
            fill_match_arms,
            r#"
            enum A { One, Two, }
            fn foo(a: A) {
                match a<|> {
                    _ => (),
                }
            }
            "#,
            r#"
            enum A { One, Two, }
            fn foo(a: A) {
                match a {
                    $0A::One => {}
                    A::Two => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn option_order() {
        mark::check!(option_order);
        let before = r#"
fn foo(opt: Option<i32>) {
    match opt<|> {
    }
}
"#;
        let before = &format!("//- /main.rs crate:main deps:core{}{}", before, FamousDefs::FIXTURE);

        check_assist(
            fill_match_arms,
            before,
            r#"
fn foo(opt: Option<i32>) {
    match opt {
        $0Some(_) => {}
        None => {}
    }
}
"#,
        );
    }
}
