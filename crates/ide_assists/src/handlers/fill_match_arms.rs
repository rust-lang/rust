use std::iter;

use either::Either;
use hir::{Adt, HasSource, ModuleDef, Semantics};
use ide_db::helpers::{mod_path_to_ast, FamousDefs};
use ide_db::RootDatabase;
use itertools::Itertools;
use syntax::ast::{self, make, AstNode, MatchArm, NameOwner, Pat};

use crate::{
    utils::{self, render_snippet, Cursor},
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
//         $0
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
    let match_expr = ctx.find_node_at_offset_with_descend::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;

    let expr = match_expr.expr()?;

    let mut arms: Vec<MatchArm> = match_arm_list.arms().collect();
    if arms.len() == 1 {
        if let Some(Pat::WildcardPat(..)) = arms[0].pat() {
            arms.clear();
        }
    }

    let top_lvl_pats: Vec<_> = arms
        .iter()
        .filter_map(ast::MatchArm::pat)
        .flat_map(|pat| match pat {
            // Special case OrPat as separate top-level pats
            Pat::OrPat(or_pat) => Either::Left(or_pat.pats()),
            _ => Either::Right(iter::once(pat)),
        })
        // Exclude top level wildcards so that they are expanded by this assist, retains status quo in #8129.
        .filter(|pat| !matches!(pat, Pat::WildcardPat(_)))
        .collect();

    let module = ctx.sema.scope(expr.syntax()).module()?;

    let missing_arms: Vec<MatchArm> = if let Some(enum_def) = resolve_enum_def(&ctx.sema, &expr) {
        let variants = enum_def.variants(ctx.db());

        let mut variants = variants
            .into_iter()
            .filter_map(|variant| build_pat(ctx.db(), module, variant))
            .filter(|variant_pat| is_variant_missing(&top_lvl_pats, variant_pat))
            .map(|pat| make::match_arm(iter::once(pat), make::expr_empty_block()))
            .collect::<Vec<_>>();
        if Some(enum_def)
            == FamousDefs(&ctx.sema, Some(module.krate()))
                .core_option_Option()
                .map(|x| lift_enum(x))
        {
            // Match `Some` variant first.
            cov_mark::hit!(option_order);
            variants.reverse()
        }
        variants
    } else if let Some(enum_defs) = resolve_tuple_of_enum_def(&ctx.sema, &expr) {
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
            .filter(|variant_pat| is_variant_missing(&top_lvl_pats, variant_pat))
            .map(|pat| make::match_arm(iter::once(pat), make::expr_empty_block()))
            .collect()
    } else {
        return None;
    };

    if missing_arms.is_empty() {
        return None;
    }

    let target = ctx.sema.original_range(match_expr.syntax()).range;
    acc.add(
        AssistId("fill_match_arms", AssistKind::QuickFix),
        "Fill match arms",
        target,
        |builder| {
            let new_arm_list = match_arm_list.remove_placeholder();
            let n_old_arms = new_arm_list.arms().count();
            let new_arm_list = new_arm_list.append_arms(missing_arms);
            let first_new_arm = new_arm_list.arms().nth(n_old_arms);
            let old_range = ctx.sema.original_range(match_arm_list.syntax()).range;
            match (first_new_arm, ctx.config.snippet_cap) {
                (Some(first_new_arm), Some(cap)) => {
                    let extend_lifetime;
                    let cursor =
                        match first_new_arm.syntax().descendants().find_map(ast::WildcardPat::cast)
                        {
                            Some(it) => {
                                extend_lifetime = it.syntax().clone();
                                Cursor::Replace(&extend_lifetime)
                            }
                            None => Cursor::Before(first_new_arm.syntax()),
                        };
                    let snippet = render_snippet(cap, new_arm_list.syntax(), cursor);
                    builder.replace_snippet(cap, old_range, snippet);
                }
                _ => builder.replace(old_range, new_arm_list.to_string()),
            }
        },
    )
}

fn is_variant_missing(existing_pats: &[Pat], var: &Pat) -> bool {
    !existing_pats.iter().any(|pat| does_pat_match_variant(pat, var))
}

// Fixme: this is still somewhat limited, use hir_ty::diagnostics::match_check?
fn does_pat_match_variant(pat: &Pat, var: &Pat) -> bool {
    match (pat, var) {
        (Pat::WildcardPat(_), _) => true,
        (Pat::TuplePat(tpat), Pat::TuplePat(tvar)) => {
            tpat.fields().zip(tvar.fields()).all(|(p, v)| does_pat_match_variant(&p, &v))
        }
        _ => utils::does_pat_match_variant(pat, var),
    }
}

#[derive(Eq, PartialEq, Clone)]
enum ExtendedEnum {
    Bool,
    Enum(hir::Enum),
}

#[derive(Eq, PartialEq, Clone)]
enum ExtendedVariant {
    True,
    False,
    Variant(hir::Variant),
}

fn lift_enum(e: hir::Enum) -> ExtendedEnum {
    ExtendedEnum::Enum(e)
}

impl ExtendedEnum {
    fn variants(&self, db: &RootDatabase) -> Vec<ExtendedVariant> {
        match self {
            ExtendedEnum::Enum(e) => {
                e.variants(db).into_iter().map(|x| ExtendedVariant::Variant(x)).collect::<Vec<_>>()
            }
            ExtendedEnum::Bool => {
                Vec::<ExtendedVariant>::from([ExtendedVariant::True, ExtendedVariant::False])
            }
        }
    }
}

fn resolve_enum_def(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<ExtendedEnum> {
    sema.type_of_expr(&expr)?.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
        Some(Adt::Enum(e)) => Some(ExtendedEnum::Enum(e)),
        _ => {
            if ty.is_bool() {
                Some(ExtendedEnum::Bool)
            } else {
                None
            }
        }
    })
}

fn resolve_tuple_of_enum_def(
    sema: &Semantics<RootDatabase>,
    expr: &ast::Expr,
) -> Option<Vec<ExtendedEnum>> {
    sema.type_of_expr(&expr)?
        .tuple_fields(sema.db)
        .iter()
        .map(|ty| {
            ty.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
                Some(Adt::Enum(e)) => Some(lift_enum(e)),
                // For now we only handle expansion for a tuple of enums. Here
                // we map non-enum items to None and rely on `collect` to
                // convert Vec<Option<hir::Enum>> into Option<Vec<hir::Enum>>.
                _ => {
                    if ty.is_bool() {
                        Some(ExtendedEnum::Bool)
                    } else {
                        None
                    }
                }
            })
        })
        .collect()
}

fn build_pat(db: &RootDatabase, module: hir::Module, var: ExtendedVariant) -> Option<ast::Pat> {
    match var {
        ExtendedVariant::Variant(var) => {
            let path = mod_path_to_ast(&module.find_use_path(db, ModuleDef::from(var))?);

            // FIXME: use HIR for this; it doesn't currently expose struct vs. tuple vs. unit variants though
            let pat: ast::Pat = match var.source(db)?.value.kind() {
                ast::StructKind::Tuple(field_list) => {
                    let pats =
                        iter::repeat(make::wildcard_pat().into()).take(field_list.fields().count());
                    make::tuple_struct_pat(path, pats).into()
                }
                ast::StructKind::Record(field_list) => {
                    let pats =
                        field_list.fields().map(|f| make::ident_pat(f.name().unwrap()).into());
                    make::record_pat(path, pats).into()
                }
                ast::StructKind::Unit => make::path_pat(path),
            };

            Some(pat)
        }
        ExtendedVariant::True => Some(ast::Pat::from(make::literal_pat("true"))),
        ExtendedVariant::False => Some(ast::Pat::from(make::literal_pat("false"))),
    }
}

#[cfg(test)]
mod tests {
    use ide_db::helpers::FamousDefs;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

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
                match A::As$0 {
                    A::As,
                    A::Bs{x,y:Some(_)} => {}
                    A::Cs(_, Some(_)) => {}
                }
            }
            "#,
        );
    }

    #[test]
    fn all_boolean_match_arms_provided() {
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            fn foo(a: bool) {
                match a$0 {
                    true => {}
                    false => {}
                }
            }
            "#,
        )
    }

    #[test]
    fn tuple_of_non_enum() {
        // for now this case is not handled, although it potentially could be
        // in the future
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            fn main() {
                match (0, false)$0 {
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_match_arms_boolean() {
        check_assist(
            fill_match_arms,
            r#"
            fn foo(a: bool) {
                match a$0 {
                }
            }
            "#,
            r#"
            fn foo(a: bool) {
                match a {
                    $0true => {}
                    false => {}
                }
            }
            "#,
        )
    }

    #[test]
    fn partial_fill_boolean() {
        check_assist(
            fill_match_arms,
            r#"
            fn foo(a: bool) {
                match a$0 {
                    true => {}
                }
            }
            "#,
            r#"
            fn foo(a: bool) {
                match a {
                    true => {}
                    $0false => {}
                }
            }
            "#,
        )
    }

    #[test]
    fn all_boolean_tuple_arms_provided() {
        check_assist_not_applicable(
            fill_match_arms,
            r#"
            fn foo(a: bool) {
                match (a, a)$0 {
                    (true, true) => {}
                    (true, false) => {}
                    (false, true) => {}
                    (false, false) => {}
                }
            }
            "#,
        )
    }

    #[test]
    fn fill_boolean_tuple() {
        check_assist(
            fill_match_arms,
            r#"
            fn foo(a: bool) {
                match (a, a)$0 {
                }
            }
            "#,
            r#"
            fn foo(a: bool) {
                match (a, a) {
                    $0(true, true) => {}
                    (true, false) => {}
                    (false, true) => {}
                    (false, false) => {}
                }
            }
            "#,
        )
    }

    #[test]
    fn partial_fill_boolean_tuple() {
        check_assist(
            fill_match_arms,
            r#"
            fn foo(a: bool) {
                match (a, a)$0 {
                    (false, true) => {}
                }
            }
            "#,
            r#"
            fn foo(a: bool) {
                match (a, a) {
                    (false, true) => {}
                    $0(true, true) => {}
                    (true, false) => {}
                    (false, false) => {}
                }
            }
            "#,
        )
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
                match A::As$0 {
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
    fn partial_fill_option() {
        check_assist(
            fill_match_arms,
            r#"
enum Option<T> { Some(T), None }
use Option::*;

fn main() {
    match None$0 {
        None => {}
    }
}
            "#,
            r#"
enum Option<T> { Some(T), None }
use Option::*;

fn main() {
    match None {
        None => {}
        Some(${0:_}) => {}
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
enum A { As, Bs, Cs(Option<i32>) }
fn main() {
    match A::As$0 {
        A::Cs(_) | A::Bs => {}
    }
}
"#,
            r#"
enum A { As, Bs, Cs(Option<i32>) }
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
enum A { As, Bs, Cs, Ds(String), Es(B) }
enum B { Xs, Ys }
fn main() {
    match A::As$0 {
        A::Bs if 0 < 1 => {}
        A::Ds(_value) => { let x = 1; }
        A::Es(B::Xs) => (),
    }
}
"#,
            r#"
enum A { As, Bs, Cs, Ds(String), Es(B) }
enum B { Xs, Ys }
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
enum A { As, Bs, Cs(Option<i32>) }
fn main() {
    match A::As$0 {
        A::As(_) => {}
        a @ A::Bs(_) => {}
    }
}
"#,
            r#"
enum A { As, Bs, Cs(Option<i32>) }
fn main() {
    match A::As {
        A::As(_) => {}
        a @ A::Bs(_) => {}
        A::Cs(${0:_}) => {}
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
enum A { As, Bs, Cs(String), Ds(String, String), Es { x: usize, y: usize } }

fn main() {
    let a = A::As;
    match a$0 {}
}
"#,
            r#"
enum A { As, Bs, Cs(String), Ds(String, String), Es { x: usize, y: usize } }

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
                match (a$0, b) {}
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
                match (&a$0, &b) {}
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
        check_assist(
            fill_match_arms,
            r#"
enum A { One, Two }
enum B { One, Two }

fn main() {
    let a = A::One;
    let b = B::One;
    match (a$0, b) {
        (A::Two, B::One) => {}
    }
}
"#,
            r#"
enum A { One, Two }
enum B { One, Two }

fn main() {
    let a = A::One;
    let b = B::One;
    match (a, b) {
        (A::Two, B::One) => {}
        $0(A::One, B::One) => {}
        (A::One, B::Two) => {}
        (A::Two, B::Two) => {}
    }
}
"#,
        );
    }

    #[test]
    fn fill_match_arms_tuple_of_enum_partial_with_wildcards() {
        let ra_fixture = r#"
fn main() {
    let a = Some(1);
    let b = Some(());
    match (a$0, b) {
        (Some(_), _) => {}
        (None, Some(_)) => {}
    }
}
"#;
        check_assist(
            fill_match_arms,
            &format!("//- /main.rs crate:main deps:core{}{}", ra_fixture, FamousDefs::FIXTURE),
            r#"
fn main() {
    let a = Some(1);
    let b = Some(());
    match (a, b) {
        (Some(_), _) => {}
        (None, Some(_)) => {}
        $0(None, None) => {}
    }
}
"#,
        );
    }

    #[test]
    fn fill_match_arms_partial_with_deep_pattern() {
        // Fixme: cannot handle deep patterns
        let ra_fixture = r#"
fn main() {
    match $0Some(true) {
        Some(true) => {}
        None => {}
    }
}
"#;
        check_assist_not_applicable(
            fill_match_arms,
            &format!("//- /main.rs crate:main deps:core{}{}", ra_fixture, FamousDefs::FIXTURE),
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
                match (a$0, b) {
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
        check_assist(
            fill_match_arms,
            r#"
            enum A { One, Two }

            fn main() {
                let a = A::One;
                match (a$0, ) {
                }
            }
            "#,
            r#"
            enum A { One, Two }

            fn main() {
                let a = A::One;
                match (a, ) {
                    $0(A::One,) => {}
                    (A::Two,) => {}
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
                match a$0 {
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
                match a$0 {
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
                match E::X$0 {}
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
                    $0_ => {}
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
                    $0
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
                    // foo bar baz$0
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
                    // foo bar baz$0
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
                match a$0 {
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
        cov_mark::check!(option_order);
        let before = r#"
fn foo(opt: Option<i32>) {
    match opt$0 {
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
        Some(${0:_}) => {}
        None => {}
    }
}
"#,
        );
    }

    #[test]
    fn works_inside_macro_call() {
        check_assist(
            fill_match_arms,
            r#"
macro_rules! m { ($expr:expr) => {$expr}}
enum Test {
    A,
    B,
    C,
}

fn foo(t: Test) {
    m!(match t$0 {});
}"#,
            r#"macro_rules! m { ($expr:expr) => {$expr}}
enum Test {
    A,
    B,
    C,
}

fn foo(t: Test) {
    m!(match t {
    $0Test::A => {}
    Test::B => {}
    Test::C => {}
});
}"#,
        );
    }
}
