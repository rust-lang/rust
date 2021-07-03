use std::iter::{self, Peekable};

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
//         $0Action::Move { distance } => todo!(),
//         Action::Stop => todo!(),
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

    let mut missing_pats: Peekable<Box<dyn Iterator<Item = ast::Pat>>> = if let Some(enum_def) =
        resolve_enum_def(&ctx.sema, &expr)
    {
        let variants = enum_def.variants(ctx.db());

        let missing_pats = variants
            .into_iter()
            .filter_map(|variant| build_pat(ctx.db(), module, variant))
            .filter(|variant_pat| is_variant_missing(&top_lvl_pats, variant_pat));

        let missing_pats: Box<dyn Iterator<Item = _>> = if Some(enum_def)
            == FamousDefs(&ctx.sema, Some(module.krate())).core_option_Option().map(lift_enum)
        {
            // Match `Some` variant first.
            cov_mark::hit!(option_order);
            Box::new(missing_pats.rev())
        } else {
            Box::new(missing_pats)
        };
        missing_pats.peekable()
    } else if let Some(enum_defs) = resolve_tuple_of_enum_def(&ctx.sema, &expr) {
        let mut n_arms = 1;
        let variants_of_enums: Vec<Vec<ExtendedVariant>> = enum_defs
            .into_iter()
            .map(|enum_def| enum_def.variants(ctx.db()))
            .inspect(|variants| n_arms *= variants.len())
            .collect();

        // When calculating the match arms for a tuple of enums, we want
        // to create a match arm for each possible combination of enum
        // values. The `multi_cartesian_product` method transforms
        // Vec<Vec<EnumVariant>> into Vec<(EnumVariant, .., EnumVariant)>
        // where each tuple represents a proposed match arm.

        // A number of arms grows very fast on even a small tuple of large enums.
        // We skip the assist beyond an arbitrary threshold.
        if n_arms > 256 {
            return None;
        }
        let missing_pats = variants_of_enums
            .into_iter()
            .multi_cartesian_product()
            .inspect(|_| cov_mark::hit!(fill_match_arms_lazy_computation))
            .map(|variants| {
                let patterns =
                    variants.into_iter().filter_map(|variant| build_pat(ctx.db(), module, variant));
                ast::Pat::from(make::tuple_pat(patterns))
            })
            .filter(|variant_pat| is_variant_missing(&top_lvl_pats, variant_pat));
        (Box::new(missing_pats) as Box<dyn Iterator<Item = _>>).peekable()
    } else {
        return None;
    };

    if missing_pats.peek().is_none() {
        return None;
    }

    let target = ctx.sema.original_range(match_expr.syntax()).range;
    acc.add(
        AssistId("fill_match_arms", AssistKind::QuickFix),
        "Fill match arms",
        target,
        |builder| {
            let new_match_arm_list = match_arm_list.clone_for_update();
            let missing_arms = missing_pats
                .map(|pat| make::match_arm(iter::once(pat), None, make::ext::expr_todo()))
                .map(|it| it.clone_for_update());

            let catch_all_arm = new_match_arm_list
                .arms()
                .find(|arm| matches!(arm.pat(), Some(ast::Pat::WildcardPat(_))));
            if let Some(arm) = catch_all_arm {
                arm.remove()
            }
            let mut first_new_arm = None;
            for arm in missing_arms {
                first_new_arm.get_or_insert_with(|| arm.clone());
                new_match_arm_list.add_arm(arm);
            }

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
                    let snippet = render_snippet(cap, new_match_arm_list.syntax(), cursor);
                    builder.replace_snippet(cap, old_range, snippet);
                }
                _ => builder.replace(old_range, new_match_arm_list.to_string()),
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

#[derive(Eq, PartialEq, Clone, Copy)]
enum ExtendedEnum {
    Bool,
    Enum(hir::Enum),
}

#[derive(Eq, PartialEq, Clone, Copy)]
enum ExtendedVariant {
    True,
    False,
    Variant(hir::Variant),
}

fn lift_enum(e: hir::Enum) -> ExtendedEnum {
    ExtendedEnum::Enum(e)
}

impl ExtendedEnum {
    fn variants(self, db: &RootDatabase) -> Vec<ExtendedVariant> {
        match self {
            ExtendedEnum::Enum(e) => {
                e.variants(db).into_iter().map(ExtendedVariant::Variant).collect::<Vec<_>>()
            }
            ExtendedEnum::Bool => {
                Vec::<ExtendedVariant>::from([ExtendedVariant::True, ExtendedVariant::False])
            }
        }
    }
}

fn resolve_enum_def(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<ExtendedEnum> {
    sema.type_of_expr(expr)?.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
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
    sema.type_of_expr(expr)?
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
                    let pats = field_list
                        .fields()
                        .map(|f| make::ext::simple_ident_pat(f.name().unwrap()).into());
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
    use crate::tests::{
        check_assist, check_assist_not_applicable, check_assist_target, check_assist_unresolved,
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
        $0true => todo!(),
        false => todo!(),
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
        $0false => todo!(),
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
        $0(true, true) => todo!(),
        (true, false) => todo!(),
        (false, true) => todo!(),
        (false, false) => todo!(),
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
        $0(true, true) => todo!(),
        (true, false) => todo!(),
        (false, false) => todo!(),
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
        $0A::As => todo!(),
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
//- minicore: option
fn main() {
    match None$0 {
        None => {}
    }
}
"#,
            r#"
fn main() {
    match None {
        None => {}
        Some(${0:_}) => todo!(),
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
        $0A::As => todo!(),
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
        $0A::As => todo!(),
        A::Cs => todo!(),
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
        A::Cs(${0:_}) => todo!(),
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
        $0A::As => todo!(),
        A::Bs => todo!(),
        A::Cs(_) => todo!(),
        A::Ds(_, _) => todo!(),
        A::Es { x, y } => todo!(),
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
        $0(A::One, B::One) => todo!(),
        (A::One, B::Two) => todo!(),
        (A::Two, B::One) => todo!(),
        (A::Two, B::Two) => todo!(),
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
        $0(A::One, B::One) => todo!(),
        (A::One, B::Two) => todo!(),
        (A::Two, B::One) => todo!(),
        (A::Two, B::Two) => todo!(),
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
        $0(A::One, B::One) => todo!(),
        (A::One, B::Two) => todo!(),
        (A::Two, B::Two) => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fill_match_arms_tuple_of_enum_partial_with_wildcards() {
        check_assist(
            fill_match_arms,
            r#"
//- minicore: option
fn main() {
    let a = Some(1);
    let b = Some(());
    match (a$0, b) {
        (Some(_), _) => {}
        (None, Some(_)) => {}
    }
}
"#,
            r#"
fn main() {
    let a = Some(1);
    let b = Some(());
    match (a, b) {
        (Some(_), _) => {}
        (None, Some(_)) => {}
        $0(None, None) => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fill_match_arms_partial_with_deep_pattern() {
        // Fixme: cannot handle deep patterns
        check_assist_not_applicable(
            fill_match_arms,
            r#"
//- minicore: option
fn main() {
    match $0Some(true) {
        Some(true) => {}
        None => {}
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
        $0(A::One,) => todo!(),
        (A::Two,) => todo!(),
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
        $0A::As => todo!(),
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
        $0A::Es { x, y } => todo!(),
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
        $0E::X => todo!(),
        E::Y => todo!(),
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
        $0X => todo!(),
        foo::E::Y => todo!(),
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
        $0A::Two => todo!(),
        // This is where the rest should be
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
        $0A::One => todo!(),
        A::Two => todo!(),
        // foo bar baz
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
        $0A::One => todo!(),
        A::Two => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn option_order() {
        cov_mark::check!(option_order);
        check_assist(
            fill_match_arms,
            r#"
//- minicore: option
fn foo(opt: Option<i32>) {
    match opt$0 {
    }
}
"#,
            r#"
fn foo(opt: Option<i32>) {
    match opt {
        Some(${0:_}) => todo!(),
        None => todo!(),
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
            r#"
macro_rules! m { ($expr:expr) => {$expr}}
enum Test {
    A,
    B,
    C,
}

fn foo(t: Test) {
    m!(match t {
    $0Test::A => todo!(),
    Test::B => todo!(),
    Test::C => todo!(),
});
}"#,
        );
    }

    #[test]
    fn lazy_computation() {
        // Computing a single missing arm is enough to determine applicability of the assist.
        cov_mark::check_count!(fill_match_arms_lazy_computation, 1);
        check_assist_unresolved(
            fill_match_arms,
            r#"
enum A { One, Two, }
fn foo(tuple: (A, A)) {
    match $0tuple {};
}
"#,
        );
    }

    #[test]
    fn adds_comma_before_new_arms() {
        check_assist(
            fill_match_arms,
            r#"
fn foo(t: bool) {
    match $0t {
        true => 1 + 2
    }
}"#,
            r#"
fn foo(t: bool) {
    match t {
        true => 1 + 2,
        $0false => todo!(),
    }
}"#,
        );
    }

    #[test]
    fn does_not_add_extra_comma() {
        check_assist(
            fill_match_arms,
            r#"
fn foo(t: bool) {
    match $0t {
        true => 1 + 2,
    }
}"#,
            r#"
fn foo(t: bool) {
    match t {
        true => 1 + 2,
        $0false => todo!(),
    }
}"#,
        );
    }
}
