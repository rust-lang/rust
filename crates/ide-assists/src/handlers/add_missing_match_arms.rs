use std::iter::{self, Peekable};

use either::Either;
use hir::{Adt, Crate, HasAttrs, HasSource, ModuleDef, Semantics};
use ide_db::RootDatabase;
use ide_db::{famous_defs::FamousDefs, helpers::mod_path_to_ast};
use itertools::Itertools;
use syntax::ast::edit_in_place::Removable;
use syntax::ast::{self, make, AstNode, HasName, MatchArmList, MatchExpr, Pat};

use crate::{utils, AssistContext, AssistId, AssistKind, Assists};

// Assist: add_missing_match_arms
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
pub(crate) fn add_missing_match_arms(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let match_expr = ctx.find_node_at_offset_with_descend::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;
    let target_range = ctx.sema.original_range(match_expr.syntax()).range;

    if let None = cursor_at_trivial_match_arm_list(ctx, &match_expr, &match_arm_list) {
        let arm_list_range = ctx.sema.original_range(match_arm_list.syntax()).range;
        let cursor_in_range = arm_list_range.contains_range(ctx.selection_trimmed());
        if cursor_in_range {
            cov_mark::hit!(not_applicable_outside_of_range_right);
            return None;
        }
    }

    let expr = match_expr.expr()?;

    let mut has_catch_all_arm = false;

    let top_lvl_pats: Vec<_> = match_arm_list
        .arms()
        .filter_map(|arm| Some((arm.pat()?, arm.guard().is_some())))
        .flat_map(|(pat, has_guard)| {
            match pat {
                // Special case OrPat as separate top-level pats
                Pat::OrPat(or_pat) => Either::Left(or_pat.pats()),
                _ => Either::Right(iter::once(pat)),
            }
            .map(move |pat| (pat, has_guard))
        })
        .map(|(pat, has_guard)| {
            has_catch_all_arm |= !has_guard && matches!(pat, Pat::WildcardPat(_));
            pat
        })
        // Exclude top level wildcards so that they are expanded by this assist, retains status quo in #8129.
        .filter(|pat| !matches!(pat, Pat::WildcardPat(_)))
        .collect();

    let module = ctx.sema.scope(expr.syntax())?.module();
    let (mut missing_pats, is_non_exhaustive, has_hidden_variants): (
        Peekable<Box<dyn Iterator<Item = (ast::Pat, bool)>>>,
        bool,
        bool,
    ) = if let Some(enum_def) = resolve_enum_def(&ctx.sema, &expr) {
        let is_non_exhaustive = enum_def.is_non_exhaustive(ctx.db(), module.krate());

        let variants = enum_def.variants(ctx.db());

        let has_hidden_variants =
            variants.iter().any(|variant| variant.should_be_hidden(ctx.db(), module.krate()));

        let missing_pats = variants
            .into_iter()
            .filter_map(|variant| {
                Some((
                    build_pat(ctx.db(), module, variant, ctx.config.prefer_no_std)?,
                    variant.should_be_hidden(ctx.db(), module.krate()),
                ))
            })
            .filter(|(variant_pat, _)| is_variant_missing(&top_lvl_pats, variant_pat));

        let option_enum = FamousDefs(&ctx.sema, module.krate()).core_option_Option().map(lift_enum);
        let missing_pats: Box<dyn Iterator<Item = _>> = if Some(enum_def) == option_enum {
            // Match `Some` variant first.
            cov_mark::hit!(option_order);
            Box::new(missing_pats.rev())
        } else {
            Box::new(missing_pats)
        };
        (missing_pats.peekable(), is_non_exhaustive, has_hidden_variants)
    } else if let Some(enum_defs) = resolve_tuple_of_enum_def(&ctx.sema, &expr) {
        let is_non_exhaustive =
            enum_defs.iter().any(|enum_def| enum_def.is_non_exhaustive(ctx.db(), module.krate()));

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

        let has_hidden_variants = variants_of_enums
            .iter()
            .flatten()
            .any(|variant| variant.should_be_hidden(ctx.db(), module.krate()));

        let missing_pats = variants_of_enums
            .into_iter()
            .multi_cartesian_product()
            .inspect(|_| cov_mark::hit!(add_missing_match_arms_lazy_computation))
            .map(|variants| {
                let is_hidden = variants
                    .iter()
                    .any(|variant| variant.should_be_hidden(ctx.db(), module.krate()));
                let patterns = variants.into_iter().filter_map(|variant| {
                    build_pat(ctx.db(), module, variant, ctx.config.prefer_no_std)
                });

                (ast::Pat::from(make::tuple_pat(patterns)), is_hidden)
            })
            .filter(|(variant_pat, _)| is_variant_missing(&top_lvl_pats, variant_pat));
        (
            (Box::new(missing_pats) as Box<dyn Iterator<Item = _>>).peekable(),
            is_non_exhaustive,
            has_hidden_variants,
        )
    } else if let Some((enum_def, len)) = resolve_array_of_enum_def(&ctx.sema, &expr) {
        let is_non_exhaustive = enum_def.is_non_exhaustive(ctx.db(), module.krate());
        let variants = enum_def.variants(ctx.db());

        if len.pow(variants.len() as u32) > 256 {
            return None;
        }

        let has_hidden_variants =
            variants.iter().any(|variant| variant.should_be_hidden(ctx.db(), module.krate()));

        let variants_of_enums = vec![variants; len];

        let missing_pats = variants_of_enums
            .into_iter()
            .multi_cartesian_product()
            .inspect(|_| cov_mark::hit!(add_missing_match_arms_lazy_computation))
            .map(|variants| {
                let is_hidden = variants
                    .iter()
                    .any(|variant| variant.should_be_hidden(ctx.db(), module.krate()));
                let patterns = variants.into_iter().filter_map(|variant| {
                    build_pat(ctx.db(), module, variant.clone(), ctx.config.prefer_no_std)
                });
                (ast::Pat::from(make::slice_pat(patterns)), is_hidden)
            })
            .filter(|(variant_pat, _)| is_variant_missing(&top_lvl_pats, variant_pat));
        (
            (Box::new(missing_pats) as Box<dyn Iterator<Item = _>>).peekable(),
            is_non_exhaustive,
            has_hidden_variants,
        )
    } else {
        return None;
    };

    let mut needs_catch_all_arm = is_non_exhaustive && !has_catch_all_arm;

    if !needs_catch_all_arm
        && ((has_hidden_variants && has_catch_all_arm) || missing_pats.peek().is_none())
    {
        return None;
    }

    acc.add(
        AssistId("add_missing_match_arms", AssistKind::QuickFix),
        "Fill match arms",
        target_range,
        |edit| {
            let new_match_arm_list = match_arm_list.clone_for_update();

            // having any hidden variants means that we need a catch-all arm
            needs_catch_all_arm |= has_hidden_variants;

            let missing_arms = missing_pats
                .filter(|(_, hidden)| {
                    // filter out hidden patterns because they're handled by the catch-all arm
                    !hidden
                })
                .map(|(pat, _)| {
                    make::match_arm(iter::once(pat), None, make::ext::expr_todo())
                        .clone_for_update()
                });

            let catch_all_arm = new_match_arm_list
                .arms()
                .find(|arm| matches!(arm.pat(), Some(ast::Pat::WildcardPat(_))));
            if let Some(arm) = catch_all_arm {
                let is_empty_expr = arm.expr().map_or(true, |e| match e {
                    ast::Expr::BlockExpr(b) => {
                        b.statements().next().is_none() && b.tail_expr().is_none()
                    }
                    ast::Expr::TupleExpr(t) => t.fields().next().is_none(),
                    _ => false,
                });
                if is_empty_expr {
                    arm.remove();
                } else {
                    cov_mark::hit!(add_missing_match_arms_empty_expr);
                }
            }

            let mut first_new_arm = None;
            for arm in missing_arms {
                first_new_arm.get_or_insert_with(|| arm.clone());
                new_match_arm_list.add_arm(arm);
            }

            if needs_catch_all_arm && !has_catch_all_arm {
                cov_mark::hit!(added_wildcard_pattern);
                let arm = make::match_arm(
                    iter::once(make::wildcard_pat().into()),
                    None,
                    make::ext::expr_todo(),
                )
                .clone_for_update();
                first_new_arm.get_or_insert_with(|| arm.clone());
                new_match_arm_list.add_arm(arm);
            }

            if let (Some(first_new_arm), Some(cap)) = (first_new_arm, ctx.config.snippet_cap) {
                match first_new_arm.syntax().descendants().find_map(ast::WildcardPat::cast) {
                    Some(it) => edit.add_placeholder_snippet(cap, it),
                    None => edit.add_tabstop_before(cap, first_new_arm),
                }
            }

            // FIXME: Hack for mutable syntax trees not having great support for macros
            // Just replace the element that the original range came from
            let old_place = {
                // Find the original element
                let old_file_range = ctx.sema.original_range(match_arm_list.syntax());
                let file = ctx.sema.parse(old_file_range.file_id);
                let old_place = file.syntax().covering_element(old_file_range.range);

                // Make `old_place` mut
                match old_place {
                    syntax::SyntaxElement::Node(it) => {
                        syntax::SyntaxElement::from(edit.make_syntax_mut(it))
                    }
                    syntax::SyntaxElement::Token(it) => {
                        // Don't have a way to make tokens mut, so instead make the parent mut
                        // and find the token again
                        let parent = edit.make_syntax_mut(it.parent().unwrap());
                        let mut_token =
                            parent.covering_element(it.text_range()).into_token().unwrap();

                        syntax::SyntaxElement::from(mut_token)
                    }
                }
            };

            syntax::ted::replace(old_place, new_match_arm_list.syntax());
        },
    )
}

fn cursor_at_trivial_match_arm_list(
    ctx: &AssistContext<'_>,
    match_expr: &MatchExpr,
    match_arm_list: &MatchArmList,
) -> Option<()> {
    // match x { $0 }
    if match_arm_list.arms().next() == None {
        cov_mark::hit!(add_missing_match_arms_empty_body);
        return Some(());
    }

    // match x {
    //     bar => baz,
    //     $0
    // }
    if let Some(last_arm) = match_arm_list.arms().last() {
        let last_arm_range = last_arm.syntax().text_range();
        let match_expr_range = match_expr.syntax().text_range();
        if last_arm_range.end() <= ctx.offset() && ctx.offset() < match_expr_range.end() {
            cov_mark::hit!(add_missing_match_arms_end_of_last_arm);
            return Some(());
        }
    }

    // match { _$0 => {...} }
    let wild_pat = ctx.find_node_at_offset_with_descend::<ast::WildcardPat>()?;
    let arm = wild_pat.syntax().parent().and_then(ast::MatchArm::cast)?;
    let arm_match_expr = arm.syntax().ancestors().nth(2).and_then(ast::MatchExpr::cast)?;
    if arm_match_expr == *match_expr {
        cov_mark::hit!(add_missing_match_arms_trivial_arm);
        return Some(());
    }

    None
}

fn is_variant_missing(existing_pats: &[Pat], var: &Pat) -> bool {
    !existing_pats.iter().any(|pat| does_pat_match_variant(pat, var))
}

// Fixme: this is still somewhat limited, use hir_ty::diagnostics::match_check?
fn does_pat_match_variant(pat: &Pat, var: &Pat) -> bool {
    match (pat, var) {
        (Pat::WildcardPat(_), _) => true,
        (Pat::SlicePat(spat), Pat::SlicePat(svar)) => {
            spat.pats().zip(svar.pats()).all(|(p, v)| does_pat_match_variant(&p, &v))
        }
        (Pat::TuplePat(tpat), Pat::TuplePat(tvar)) => {
            tpat.fields().zip(tvar.fields()).all(|(p, v)| does_pat_match_variant(&p, &v))
        }
        (Pat::OrPat(opat), _) => opat.pats().any(|p| does_pat_match_variant(&p, var)),
        _ => utils::does_pat_match_variant(pat, var),
    }
}

#[derive(Eq, PartialEq, Clone, Copy)]
enum ExtendedEnum {
    Bool,
    Enum(hir::Enum),
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
enum ExtendedVariant {
    True,
    False,
    Variant(hir::Variant),
}

impl ExtendedVariant {
    fn should_be_hidden(self, db: &RootDatabase, krate: Crate) -> bool {
        match self {
            ExtendedVariant::Variant(var) => {
                var.attrs(db).has_doc_hidden() && var.module(db).krate() != krate
            }
            _ => false,
        }
    }
}

fn lift_enum(e: hir::Enum) -> ExtendedEnum {
    ExtendedEnum::Enum(e)
}

impl ExtendedEnum {
    fn is_non_exhaustive(self, db: &RootDatabase, krate: Crate) -> bool {
        match self {
            ExtendedEnum::Enum(e) => {
                e.attrs(db).by_key("non_exhaustive").exists() && e.module(db).krate() != krate
            }
            _ => false,
        }
    }

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

fn resolve_enum_def(sema: &Semantics<'_, RootDatabase>, expr: &ast::Expr) -> Option<ExtendedEnum> {
    sema.type_of_expr(expr)?.adjusted().autoderef(sema.db).find_map(|ty| match ty.as_adt() {
        Some(Adt::Enum(e)) => Some(ExtendedEnum::Enum(e)),
        _ => ty.is_bool().then_some(ExtendedEnum::Bool),
    })
}

fn resolve_tuple_of_enum_def(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
) -> Option<Vec<ExtendedEnum>> {
    sema.type_of_expr(expr)?
        .adjusted()
        .tuple_fields(sema.db)
        .iter()
        .map(|ty| {
            ty.autoderef(sema.db).find_map(|ty| {
                match ty.as_adt() {
                    Some(Adt::Enum(e)) => Some(lift_enum(e)),
                    // For now we only handle expansion for a tuple of enums. Here
                    // we map non-enum items to None and rely on `collect` to
                    // convert Vec<Option<hir::Enum>> into Option<Vec<hir::Enum>>.
                    _ => ty.is_bool().then_some(ExtendedEnum::Bool),
                }
            })
        })
        .collect::<Option<Vec<ExtendedEnum>>>()
        .and_then(|list| if list.is_empty() { None } else { Some(list) })
}

fn resolve_array_of_enum_def(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
) -> Option<(ExtendedEnum, usize)> {
    sema.type_of_expr(expr)?.adjusted().as_array(sema.db).and_then(|(ty, len)| {
        ty.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
            Some(Adt::Enum(e)) => Some((lift_enum(e), len)),
            _ => ty.is_bool().then_some((ExtendedEnum::Bool, len)),
        })
    })
}

fn build_pat(
    db: &RootDatabase,
    module: hir::Module,
    var: ExtendedVariant,
    prefer_no_std: bool,
) -> Option<ast::Pat> {
    match var {
        ExtendedVariant::Variant(var) => {
            let path =
                mod_path_to_ast(&module.find_use_path(db, ModuleDef::from(var), prefer_no_std)?);

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

    use super::add_missing_match_arms;

    #[test]
    fn all_match_arms_provided() {
        check_assist_not_applicable(
            add_missing_match_arms,
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
    fn not_applicable_outside_of_range_left() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
enum A { X, Y }

fn foo(a: A) {
    $0 match a {
        A::X => { }
    }
}
        "#,
        );
    }

    #[test]
    fn not_applicable_outside_of_range_right() {
        cov_mark::check!(not_applicable_outside_of_range_right);
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
enum A { X, Y }

fn foo(a: A) {
    match a {$0
        A::X => { }
    }
}
        "#,
        );
    }

    #[test]
    fn all_boolean_match_arms_provided() {
        check_assist_not_applicable(
            add_missing_match_arms,
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
            add_missing_match_arms,
            r#"
fn main() {
    match (0, false)$0 {
    }
}
"#,
        );
    }

    #[test]
    fn add_missing_match_arms_boolean() {
        check_assist(
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
            r#"
fn foo(a: bool) {
    match (a, a)$0 {
        (true | false, true) => {}
        (true, false) => {}
        (false, false) => {}
    }
}
"#,
        );

        check_assist_not_applicable(
            add_missing_match_arms,
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
            add_missing_match_arms,
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
    fn fill_boolean_array() {
        check_assist(
            add_missing_match_arms,
            r#"
fn foo(a: bool) {
    match [a]$0 {
    }
}
"#,
            r#"
fn foo(a: bool) {
    match [a] {
        $0[true] => todo!(),
        [false] => todo!(),
    }
}
"#,
        );

        check_assist(
            add_missing_match_arms,
            r#"
fn foo(a: bool) {
    match [a,]$0 {
    }
}
"#,
            r#"
fn foo(a: bool) {
    match [a,] {
        $0[true] => todo!(),
        [false] => todo!(),
    }
}
"#,
        );

        check_assist(
            add_missing_match_arms,
            r#"
fn foo(a: bool) {
    match [a, a]$0 {
        [true, true] => todo!(),
    }
}
"#,
            r#"
fn foo(a: bool) {
    match [a, a] {
        [true, true] => todo!(),
        $0[true, false] => todo!(),
        [false, true] => todo!(),
        [false, false] => todo!(),
    }
}
"#,
        );

        check_assist(
            add_missing_match_arms,
            r#"
fn foo(a: bool) {
    match [a, a]$0 {
    }
}
"#,
            r#"
fn foo(a: bool) {
    match [a, a] {
        $0[true, true] => todo!(),
        [true, false] => todo!(),
        [false, true] => todo!(),
        [false, false] => todo!(),
    }
}
"#,
        )
    }

    #[test]
    fn partial_fill_boolean_tuple() {
        check_assist(
            add_missing_match_arms,
            r#"
fn foo(a: bool) {
    match (a, a)$0 {
        (true | false, true) => {}
    }
}
"#,
            r#"
fn foo(a: bool) {
    match (a, a) {
        (true | false, true) => {}
        $0(true, false) => todo!(),
        (false, false) => todo!(),
    }
}
"#,
        );

        check_assist(
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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
    fn add_missing_match_arms_empty_body() {
        cov_mark::check!(add_missing_match_arms_empty_body);
        check_assist(
            add_missing_match_arms,
            r#"
enum A { As, Bs, Cs(String), Ds(String, String), Es { x: usize, y: usize } }

fn main() {
    let a = A::As;
    match a {$0}
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
    fn add_missing_match_arms_end_of_last_arm() {
        cov_mark::check!(add_missing_match_arms_end_of_last_arm);
        check_assist(
            add_missing_match_arms,
            r#"
enum A { One, Two }
enum B { One, Two }

fn main() {
    let a = A::One;
    let b = B::One;
    match (a, b) {
        (A::Two, B::One) => {},$0
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
        (A::Two, B::One) => {},
        $0(A::One, B::One) => todo!(),
        (A::One, B::Two) => todo!(),
        (A::Two, B::Two) => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn add_missing_match_arms_tuple_of_enum() {
        check_assist(
            add_missing_match_arms,
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
    fn add_missing_match_arms_tuple_of_enum_ref() {
        check_assist(
            add_missing_match_arms,
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
    fn add_missing_match_arms_tuple_of_enum_partial() {
        check_assist(
            add_missing_match_arms,
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

        check_assist(
            add_missing_match_arms,
            r#"
enum E { A, B, C }
fn main() {
    use E::*;
    match (A, B, C)$0 {
        (A | B , A, A | B | C) => (),
        (A | B | C , B | C, A | B | C) => (),
    }
}
"#,
            r#"
enum E { A, B, C }
fn main() {
    use E::*;
    match (A, B, C) {
        (A | B , A, A | B | C) => (),
        (A | B | C , B | C, A | B | C) => (),
        $0(C, A, A) => todo!(),
        (C, A, B) => todo!(),
        (C, A, C) => todo!(),
    }
}
"#,
        )
    }

    #[test]
    fn add_missing_match_arms_tuple_of_enum_partial_with_wildcards() {
        check_assist(
            add_missing_match_arms,
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
    fn add_missing_match_arms_partial_with_deep_pattern() {
        // Fixme: cannot handle deep patterns
        check_assist_not_applicable(
            add_missing_match_arms,
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
    fn add_missing_match_arms_tuple_of_enum_not_applicable() {
        check_assist_not_applicable(
            add_missing_match_arms,
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
    fn add_missing_match_arms_single_element_tuple_of_enum() {
        check_assist(
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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
    fn add_missing_match_arms_target_simple() {
        check_assist_target(
            add_missing_match_arms,
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
    fn add_missing_match_arms_target_complex() {
        check_assist_target(
            add_missing_match_arms,
            r#"
enum E { X, Y }

fn main() {
    match E::X$0 {
        E::X => {}
    }
}
"#,
            "match E::X {
        E::X => {}
    }",
        );
    }

    #[test]
    fn add_missing_match_arms_trivial_arm() {
        cov_mark::check!(add_missing_match_arms_trivial_arm);
        check_assist(
            add_missing_match_arms,
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
    fn wildcard_inside_expression_not_applicable() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
enum E { X, Y }

fn foo(e : E) {
    match e {
        _ => {
            println!("1");$0
            println!("2");
        }
    }
}
"#,
        );
    }

    #[test]
    fn add_missing_match_arms_qualifies_path() {
        check_assist(
            add_missing_match_arms,
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
    fn add_missing_match_arms_preserves_comments() {
        check_assist(
            add_missing_match_arms,
            r#"
enum A { One, Two }
fn foo(a: A) {
    match a $0 {
        // foo bar baz
        A::One => {}
        // This is where the rest should be
    }
}
"#,
            r#"
enum A { One, Two }
fn foo(a: A) {
    match a  {
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
    fn add_missing_match_arms_preserves_comments_empty() {
        check_assist(
            add_missing_match_arms,
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
    fn add_missing_match_arms_placeholder() {
        check_assist(
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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
        cov_mark::check_count!(add_missing_match_arms_lazy_computation, 1);
        check_assist_unresolved(
            add_missing_match_arms,
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
            add_missing_match_arms,
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
            add_missing_match_arms,
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

    #[test]
    fn does_not_remove_catch_all_with_non_empty_expr() {
        cov_mark::check!(add_missing_match_arms_empty_expr);
        check_assist(
            add_missing_match_arms,
            r#"
fn foo(t: bool) {
    match $0t {
        _ => 1 + 2,
    }
}"#,
            r#"
fn foo(t: bool) {
    match t {
        _ => 1 + 2,
        $0true => todo!(),
        false => todo!(),
    }
}"#,
        );
    }

    #[test]
    fn does_not_fill_hidden_variants() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
    }
}
//- /e.rs crate:e
pub enum E { A, #[doc(hidden)] B, }
"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        $0e::E::A => todo!(),
        _ => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn does_not_fill_hidden_variants_tuple() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: (bool, ::e::E)) {
    match $0t {
    }
}
//- /e.rs crate:e
pub enum E { A, #[doc(hidden)] B, }
"#,
            r#"
fn foo(t: (bool, ::e::E)) {
    match t {
        $0(true, e::E::A) => todo!(),
        (false, e::E::A) => todo!(),
        _ => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fills_wildcard_with_only_hidden_variants() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }
"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        ${0:_} => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn does_not_fill_wildcard_when_hidden_variants_are_explicit() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
        e::E::A => todo!(),
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }
"#,
        );
    }

    #[test]
    fn does_not_fill_wildcard_with_wildcard() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
        _ => todo!(),
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }
"#,
        );
    }

    #[test]
    fn fills_wildcard_on_non_exhaustive_with_explicit_matches() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
        e::E::A => todo!(),
    }
}
//- /e.rs crate:e
#[non_exhaustive]
pub enum E { A, }
"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        e::E::A => todo!(),
        ${0:_} => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fills_wildcard_on_non_exhaustive_without_matches() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
    }
}
//- /e.rs crate:e
#[non_exhaustive]
pub enum E { A, }
"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        $0e::E::A => todo!(),
        _ => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fills_wildcard_on_non_exhaustive_with_doc_hidden() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
    }
}
//- /e.rs crate:e
#[non_exhaustive]
pub enum E { A, #[doc(hidden)] B }"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        $0e::E::A => todo!(),
        _ => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fills_wildcard_on_non_exhaustive_with_doc_hidden_with_explicit_arms() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
        e::E::A => todo!(),
    }
}
//- /e.rs crate:e
#[non_exhaustive]
pub enum E { A, #[doc(hidden)] B }"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        e::E::A => todo!(),
        ${0:_} => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn fill_wildcard_with_partial_wildcard() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E, b: bool) {
    match $0t {
        _ if b => todo!(),
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }"#,
            r#"
fn foo(t: ::e::E, b: bool) {
    match t {
        _ if b => todo!(),
        ${0:_} => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn does_not_fill_wildcard_with_partial_wildcard_and_wildcard() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E, b: bool) {
    match $0t {
        _ if b => todo!(),
        _ => todo!(),
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }"#,
        );
    }

    #[test]
    fn non_exhaustive_doc_hidden_tuple_fills_wildcard() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E) {
    match $0t {
    }
}
//- /e.rs crate:e
#[non_exhaustive]
pub enum E { A, #[doc(hidden)] B, }"#,
            r#"
fn foo(t: ::e::E) {
    match t {
        $0e::E::A => todo!(),
        _ => todo!(),
    }
}
"#,
        );
    }

    #[test]
    fn ignores_doc_hidden_for_crate_local_enums() {
        check_assist(
            add_missing_match_arms,
            r#"
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match $0t {
    }
}"#,
            r#"
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match t {
        $0E::A => todo!(),
        E::B => todo!(),
    }
}"#,
        );
    }

    #[test]
    fn ignores_non_exhaustive_for_crate_local_enums() {
        check_assist(
            add_missing_match_arms,
            r#"
#[non_exhaustive]
enum E { A, B, }

fn foo(t: E) {
    match $0t {
    }
}"#,
            r#"
#[non_exhaustive]
enum E { A, B, }

fn foo(t: E) {
    match t {
        $0E::A => todo!(),
        E::B => todo!(),
    }
}"#,
        );
    }

    #[test]
    fn ignores_doc_hidden_and_non_exhaustive_for_crate_local_enums() {
        check_assist(
            add_missing_match_arms,
            r#"
#[non_exhaustive]
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match $0t {
    }
}"#,
            r#"
#[non_exhaustive]
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match t {
        $0E::A => todo!(),
        E::B => todo!(),
    }
}"#,
        );
    }
}
