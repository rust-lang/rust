use std::iter::{self, Peekable};

use either::Either;
use hir::{Adt, AsAssocItem, Crate, HasAttrs, ImportPathConfig, ModuleDef, Semantics, sym};
use ide_db::RootDatabase;
use ide_db::assists::ExprFillDefaultMode;
use ide_db::syntax_helpers::suggest_name;
use ide_db::{famous_defs::FamousDefs, helpers::mod_path_to_ast};
use itertools::Itertools;
use syntax::ToSmolStr;
use syntax::ast::edit::IndentLevel;
use syntax::ast::edit_in_place::Indent;
use syntax::ast::syntax_factory::SyntaxFactory;
use syntax::ast::{self, AstNode, MatchArmList, MatchExpr, Pat, make};

use crate::{AssistContext, AssistId, Assists, utils};

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
//         Action::Move { distance } => ${1:todo!()},
//         Action::Stop => ${2:todo!()},$0
//     }
// }
// ```
pub(crate) fn add_missing_match_arms(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let match_expr = ctx.find_node_at_offset_with_descend::<ast::MatchExpr>()?;
    let match_arm_list = match_expr.match_arm_list()?;
    let arm_list_range = ctx.sema.original_range_opt(match_arm_list.syntax())?;

    if cursor_at_trivial_match_arm_list(ctx, &match_expr, &match_arm_list).is_none() {
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

    let cfg = ctx.config.import_path_config();

    let make = SyntaxFactory::with_mappings();

    let scope = ctx.sema.scope(expr.syntax())?;
    let module = scope.module();
    let self_ty = if ctx.config.prefer_self_ty {
        scope
            .containing_function()
            .and_then(|function| function.as_assoc_item(ctx.db())?.implementing_ty(ctx.db()))
    } else {
        None
    };
    let (mut missing_pats, is_non_exhaustive, has_hidden_variants): (
        Peekable<Box<dyn Iterator<Item = (ast::Pat, bool)>>>,
        bool,
        bool,
    ) = if let Some(enum_def) = resolve_enum_def(&ctx.sema, &expr, self_ty.as_ref()) {
        let is_non_exhaustive = enum_def.is_non_exhaustive(ctx.db(), module.krate());

        let variants = enum_def.variants(ctx.db());

        let has_hidden_variants =
            variants.iter().any(|variant| variant.should_be_hidden(ctx.db(), module.krate()));

        let missing_pats = variants
            .into_iter()
            .filter_map(|variant| {
                Some((
                    build_pat(ctx, &make, module, variant, cfg)?,
                    variant.should_be_hidden(ctx.db(), module.krate()),
                ))
            })
            .filter(|(variant_pat, _)| is_variant_missing(&top_lvl_pats, variant_pat));

        let option_enum = FamousDefs(&ctx.sema, module.krate()).core_option_Option();
        let missing_pats: Box<dyn Iterator<Item = _>> = if matches!(enum_def, ExtendedEnum::Enum { enum_: e, .. } if Some(e) == option_enum)
        {
            // Match `Some` variant first.
            cov_mark::hit!(option_order);
            Box::new(missing_pats.rev())
        } else {
            Box::new(missing_pats)
        };
        (missing_pats.peekable(), is_non_exhaustive, has_hidden_variants)
    } else if let Some(enum_defs) = resolve_tuple_of_enum_def(&ctx.sema, &expr, self_ty.as_ref()) {
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
                let patterns = variants
                    .into_iter()
                    .filter_map(|variant| build_pat(ctx, &make, module, variant, cfg));

                (ast::Pat::from(make.tuple_pat(patterns)), is_hidden)
            })
            .filter(|(variant_pat, _)| is_variant_missing(&top_lvl_pats, variant_pat));
        (
            (Box::new(missing_pats) as Box<dyn Iterator<Item = _>>).peekable(),
            is_non_exhaustive,
            has_hidden_variants,
        )
    } else if let Some((enum_def, len)) =
        resolve_array_of_enum_def(&ctx.sema, &expr, self_ty.as_ref())
    {
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
                let patterns = variants
                    .into_iter()
                    .filter_map(|variant| build_pat(ctx, &make, module, variant, cfg));

                (ast::Pat::from(make.slice_pat(patterns)), is_hidden)
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
        AssistId::quick_fix("add_missing_match_arms"),
        "Fill match arms",
        ctx.sema.original_range(match_expr.syntax()).range,
        |builder| {
            // having any hidden variants means that we need a catch-all arm
            needs_catch_all_arm |= has_hidden_variants;

            let missing_arms = missing_pats
                .filter(|(_, hidden)| {
                    // filter out hidden patterns because they're handled by the catch-all arm
                    !hidden
                })
                .map(|(pat, _)| {
                    make.match_arm(
                        pat,
                        None,
                        match ctx.config.expr_fill_default {
                            ExprFillDefaultMode::Todo => make::ext::expr_todo(),
                            ExprFillDefaultMode::Underscore => make::ext::expr_underscore(),
                            ExprFillDefaultMode::Default => make::ext::expr_todo(),
                        },
                    )
                });

            let mut arms: Vec<_> = match_arm_list
                .arms()
                .filter(|arm| {
                    if matches!(arm.pat(), Some(ast::Pat::WildcardPat(_))) {
                        let is_empty_expr = arm.expr().is_none_or(|e| match e {
                            ast::Expr::BlockExpr(b) => {
                                b.statements().next().is_none() && b.tail_expr().is_none()
                            }
                            ast::Expr::TupleExpr(t) => t.fields().next().is_none(),
                            _ => false,
                        });
                        if is_empty_expr {
                            false
                        } else {
                            cov_mark::hit!(add_missing_match_arms_empty_expr);
                            true
                        }
                    } else {
                        true
                    }
                })
                .collect();

            let first_new_arm_idx = arms.len();
            arms.extend(missing_arms);

            if needs_catch_all_arm && !has_catch_all_arm {
                cov_mark::hit!(added_wildcard_pattern);
                let arm = make.match_arm(
                    make.wildcard_pat().into(),
                    None,
                    match ctx.config.expr_fill_default {
                        ExprFillDefaultMode::Todo => make::ext::expr_todo(),
                        ExprFillDefaultMode::Underscore => make::ext::expr_underscore(),
                        ExprFillDefaultMode::Default => make::ext::expr_todo(),
                    },
                );
                arms.push(arm);
            }

            let new_match_arm_list = make.match_arm_list(arms);

            // FIXME: Hack for syntax trees not having great support for macros
            // Just replace the element that the original range came from
            let old_place = {
                // Find the original element
                let file = ctx.sema.parse(arm_list_range.file_id);
                let old_place = file.syntax().covering_element(arm_list_range.range);

                match old_place {
                    syntax::SyntaxElement::Node(it) => it,
                    syntax::SyntaxElement::Token(it) => {
                        // If a token is found, it is '{' or '}'
                        // The parent is `{ ... }`
                        it.parent().expect("Token must have a parent.")
                    }
                }
            };

            let mut editor = builder.make_editor(&old_place);
            new_match_arm_list.indent(IndentLevel::from_node(&old_place));
            editor.replace(old_place, new_match_arm_list.syntax());

            if let Some(cap) = ctx.config.snippet_cap {
                if let Some(it) = new_match_arm_list
                    .arms()
                    .nth(first_new_arm_idx)
                    .and_then(|arm| arm.syntax().descendants().find_map(ast::WildcardPat::cast))
                {
                    editor.add_annotation(it.syntax(), builder.make_placeholder_snippet(cap));
                }

                for arm in new_match_arm_list.arms().skip(first_new_arm_idx) {
                    if let Some(expr) = arm.expr() {
                        editor.add_annotation(expr.syntax(), builder.make_placeholder_snippet(cap));
                    }
                }

                if let Some(arm) = new_match_arm_list.arms().skip(first_new_arm_idx).last() {
                    editor.add_annotation(arm.syntax(), builder.make_tabstop_after(cap));
                }
            }

            editor.add_mappings(make.take());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn cursor_at_trivial_match_arm_list(
    ctx: &AssistContext<'_>,
    match_expr: &MatchExpr,
    match_arm_list: &MatchArmList,
) -> Option<()> {
    // match x { $0 }
    if match_arm_list.arms().next().is_none() {
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

#[derive(Eq, PartialEq, Clone)]
enum ExtendedEnum {
    Bool,
    Enum { enum_: hir::Enum, use_self: bool },
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
enum ExtendedVariant {
    True,
    False,
    Variant { variant: hir::Variant, use_self: bool },
}

impl ExtendedVariant {
    fn should_be_hidden(self, db: &RootDatabase, krate: Crate) -> bool {
        match self {
            ExtendedVariant::Variant { variant: var, .. } => {
                var.attrs(db).has_doc_hidden() && var.module(db).krate() != krate
            }
            _ => false,
        }
    }
}

impl ExtendedEnum {
    fn enum_(
        db: &RootDatabase,
        enum_: hir::Enum,
        enum_ty: &hir::Type<'_>,
        self_ty: Option<&hir::Type<'_>>,
    ) -> Self {
        ExtendedEnum::Enum {
            enum_,
            use_self: self_ty.is_some_and(|self_ty| self_ty.could_unify_with_deeply(db, enum_ty)),
        }
    }

    fn is_non_exhaustive(&self, db: &RootDatabase, krate: Crate) -> bool {
        match self {
            ExtendedEnum::Enum { enum_: e, .. } => {
                e.attrs(db).by_key(sym::non_exhaustive).exists() && e.module(db).krate() != krate
            }
            _ => false,
        }
    }

    fn variants(&self, db: &RootDatabase) -> Vec<ExtendedVariant> {
        match *self {
            ExtendedEnum::Enum { enum_: e, use_self } => e
                .variants(db)
                .into_iter()
                .map(|variant| ExtendedVariant::Variant { variant, use_self })
                .collect::<Vec<_>>(),
            ExtendedEnum::Bool => {
                Vec::<ExtendedVariant>::from([ExtendedVariant::True, ExtendedVariant::False])
            }
        }
    }
}

fn resolve_enum_def(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
    self_ty: Option<&hir::Type<'_>>,
) -> Option<ExtendedEnum> {
    sema.type_of_expr(expr)?.adjusted().autoderef(sema.db).find_map(|ty| match ty.as_adt() {
        Some(Adt::Enum(e)) => Some(ExtendedEnum::enum_(sema.db, e, &ty, self_ty)),
        _ => ty.is_bool().then_some(ExtendedEnum::Bool),
    })
}

fn resolve_tuple_of_enum_def(
    sema: &Semantics<'_, RootDatabase>,
    expr: &ast::Expr,
    self_ty: Option<&hir::Type<'_>>,
) -> Option<Vec<ExtendedEnum>> {
    sema.type_of_expr(expr)?
        .adjusted()
        .tuple_fields(sema.db)
        .iter()
        .map(|ty| {
            ty.autoderef(sema.db).find_map(|ty| {
                match ty.as_adt() {
                    Some(Adt::Enum(e)) => Some(ExtendedEnum::enum_(sema.db, e, &ty, self_ty)),
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
    self_ty: Option<&hir::Type<'_>>,
) -> Option<(ExtendedEnum, usize)> {
    sema.type_of_expr(expr)?.adjusted().as_array(sema.db).and_then(|(ty, len)| {
        ty.autoderef(sema.db).find_map(|ty| match ty.as_adt() {
            Some(Adt::Enum(e)) => Some((ExtendedEnum::enum_(sema.db, e, &ty, self_ty), len)),
            _ => ty.is_bool().then_some((ExtendedEnum::Bool, len)),
        })
    })
}

fn build_pat(
    ctx: &AssistContext<'_>,
    make: &SyntaxFactory,
    module: hir::Module,
    var: ExtendedVariant,
    cfg: ImportPathConfig,
) -> Option<ast::Pat> {
    let db = ctx.db();
    match var {
        ExtendedVariant::Variant { variant: var, use_self } => {
            let edition = module.krate().edition(db);
            let path = if use_self {
                make::path_from_segments(
                    [
                        make::path_segment(make::name_ref_self_ty()),
                        make::path_segment(make::name_ref(
                            &var.name(db).display(db, edition).to_smolstr(),
                        )),
                    ],
                    false,
                )
            } else {
                mod_path_to_ast(&module.find_path(db, ModuleDef::from(var), cfg)?, edition)
            };
            let fields = var.fields(db);
            let pat: ast::Pat = match var.kind(db) {
                hir::StructKind::Tuple => {
                    let mut name_generator = suggest_name::NameGenerator::default();
                    let pats = fields.into_iter().map(|f| {
                        let name = name_generator.for_type(&f.ty(db), db, edition);
                        match name {
                            Some(name) => make::ext::simple_ident_pat(make.name(&name)).into(),
                            None => make.wildcard_pat().into(),
                        }
                    });
                    make.tuple_struct_pat(path, pats).into()
                }
                hir::StructKind::Record => {
                    let fields = fields
                        .into_iter()
                        .map(|f| make.ident_pat(false, false, make.name(f.name(db).as_str())))
                        .map(|ident| make.record_pat_field_shorthand(ident.into()));
                    let fields = make.record_pat_field_list(fields, None);
                    make.record_pat_with_fields(path, fields).into()
                }
                hir::StructKind::Unit => make.path_pat(path),
            };
            Some(pat)
        }
        ExtendedVariant::True => Some(ast::Pat::from(make.literal_pat("true"))),
        ExtendedVariant::False => Some(ast::Pat::from(make.literal_pat("false"))),
    }
}

#[cfg(test)]
mod tests {
    use crate::AssistConfig;
    use crate::tests::{
        TEST_CONFIG, check_assist, check_assist_not_applicable, check_assist_target,
        check_assist_unresolved, check_assist_with_config,
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
        true => ${1:todo!()},
        false => ${2:todo!()},$0
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
        false => ${1:todo!()},$0
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
        (true, true) => ${1:todo!()},
        (true, false) => ${2:todo!()},
        (false, true) => ${3:todo!()},
        (false, false) => ${4:todo!()},$0
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
        [true] => ${1:todo!()},
        [false] => ${2:todo!()},$0
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
        [true] => ${1:todo!()},
        [false] => ${2:todo!()},$0
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
        [true, false] => ${1:todo!()},
        [false, true] => ${2:todo!()},
        [false, false] => ${3:todo!()},$0
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
        [true, true] => ${1:todo!()},
        [true, false] => ${2:todo!()},
        [false, true] => ${3:todo!()},
        [false, false] => ${4:todo!()},$0
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
        (true, false) => ${1:todo!()},
        (false, false) => ${2:todo!()},$0
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
        (true, true) => ${1:todo!()},
        (true, false) => ${2:todo!()},
        (false, false) => ${3:todo!()},$0
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
        A::As => ${1:todo!()},$0
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
        Some(${1:_}) => ${2:todo!()},$0
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
        A::As => ${1:todo!()},$0
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
        A::As => ${1:todo!()},
        A::Cs => ${2:todo!()},$0
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
        A::Cs(${1:_}) => ${2:todo!()},$0
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
        A::As => ${1:todo!()},
        A::Bs => ${2:todo!()},
        A::Cs(_) => ${3:todo!()},
        A::Ds(_, _) => ${4:todo!()},
        A::Es { x, y } => ${5:todo!()},$0
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
        (A::One, B::One) => ${1:todo!()},
        (A::One, B::Two) => ${2:todo!()},
        (A::Two, B::Two) => ${3:todo!()},$0
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
        (A::One, B::One) => ${1:todo!()},
        (A::One, B::Two) => ${2:todo!()},
        (A::Two, B::One) => ${3:todo!()},
        (A::Two, B::Two) => ${4:todo!()},$0
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
        (A::One, B::One) => ${1:todo!()},
        (A::One, B::Two) => ${2:todo!()},
        (A::Two, B::One) => ${3:todo!()},
        (A::Two, B::Two) => ${4:todo!()},$0
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
        (A::One, B::One) => ${1:todo!()},
        (A::One, B::Two) => ${2:todo!()},
        (A::Two, B::Two) => ${3:todo!()},$0
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
        (C, A, A) => ${1:todo!()},
        (C, A, B) => ${2:todo!()},
        (C, A, C) => ${3:todo!()},$0
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
        (None, None) => ${1:todo!()},$0
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
        (A::One,) => ${1:todo!()},
        (A::Two,) => ${2:todo!()},$0
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
        A::As => ${1:todo!()},$0
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
        A::Es { x, y } => ${1:todo!()},$0
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
        E::X => ${1:todo!()},
        E::Y => ${2:todo!()},$0
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
        X => ${1:todo!()},
        foo::E::Y => ${2:todo!()},$0
    }
}
"#,
        );
    }

    // FIXME: Preserving comments is quite hard in the current transitional syntax editing model.
    // Once we migrate to new trivia model addressed in #6854, remove the ignore attribute.
    #[ignore]
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
        A::Two => ${1:todo!()},$0
        // This is where the rest should be
    }
}
"#,
        );
    }

    // FIXME: Preserving comments is quite hard in the current transitional syntax editing model.
    // Once we migrate to new trivia model addressed in #6854, remove the ignore attribute.
    #[ignore]
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
        A::One => ${1:todo!()},
        A::Two => ${2:todo!()},$0
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
        A::One => ${1:todo!()},
        A::Two => ${2:todo!()},$0
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
        Some(${1:_}) => ${2:todo!()},
        None => ${3:todo!()},$0
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
        Test::A => ${1:todo!()},
        Test::B => ${2:todo!()},
        Test::C => ${3:todo!()},$0
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
        false => ${1:todo!()},$0
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
        false => ${1:todo!()},$0
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
        true => ${1:todo!()},
        false => ${2:todo!()},$0
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
        e::E::A => ${1:todo!()},
        _ => ${2:todo!()},$0
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
        (true, e::E::A) => ${1:todo!()},
        (false, e::E::A) => ${2:todo!()},
        _ => ${3:todo!()},$0
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
        ${1:_} => ${2:todo!()},$0
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
        ${1:_} => ${2:todo!()},$0
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
        e::E::A => ${1:todo!()},
        _ => ${2:todo!()},$0
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
        e::E::A => ${1:todo!()},
        _ => ${2:todo!()},$0
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
        ${1:_} => ${2:todo!()},$0
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
        ${1:_} => ${2:todo!()},$0
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
        e::E::A => ${1:todo!()},
        _ => ${2:todo!()},$0
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
        E::A => ${1:todo!()},
        E::B => ${2:todo!()},$0
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
        E::A => ${1:todo!()},
        E::B => ${2:todo!()},$0
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
        E::A => ${1:todo!()},
        E::B => ${2:todo!()},$0
    }
}"#,
        );
    }

    #[test]
    fn not_applicable_when_match_arm_list_cannot_be_upmapped() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
macro_rules! foo {
    ($($t:tt)*) => {
        $($t)* {}
    }
}

enum E { A }

fn main() {
    foo!(match E::A$0);
}
"#,
        );
    }

    /// See [`discussion`](https://github.com/rust-lang/rust-analyzer/pull/15594#discussion_r1322960614)
    #[test]
    fn missing_field_name() {
        check_assist(
            add_missing_match_arms,
            r#"
enum A {
    A,
    Missing { a: u32, : u32, c: u32 }
}

fn a() {
    let b = A::A;
    match b$0 {}
}"#,
            r#"
enum A {
    A,
    Missing { a: u32, : u32, c: u32 }
}

fn a() {
    let b = A::A;
    match b {
        A::A => ${1:todo!()},
        A::Missing { a, u32, c } => ${2:todo!()},$0
    }
}"#,
        )
    }

    #[test]
    fn suggest_name_for_tuple_struct_patterns() {
        // single tuple struct
        check_assist(
            add_missing_match_arms,
            r#"
struct S;

pub enum E {
    A
    B(S),
}

fn f() {
    let value = E::A;
    match value {
        $0
    }
}
"#,
            r#"
struct S;

pub enum E {
    A
    B(S),
}

fn f() {
    let value = E::A;
    match value {
        E::A => ${1:todo!()},
        E::B(s) => ${2:todo!()},$0
    }
}
"#,
        );

        // multiple tuple struct patterns
        check_assist(
            add_missing_match_arms,
            r#"
struct S1;
struct S2;

pub enum E {
    A
    B(S1, S2),
}

fn f() {
    let value = E::A;
    match value {
        $0
    }
}
"#,
            r#"
struct S1;
struct S2;

pub enum E {
    A
    B(S1, S2),
}

fn f() {
    let value = E::A;
    match value {
        E::A => ${1:todo!()},
        E::B(s1, s2) => ${2:todo!()},$0
    }
}
"#,
        );
    }

    #[test]
    fn prefer_self() {
        check_assist_with_config(
            add_missing_match_arms,
            AssistConfig { prefer_self_ty: true, ..TEST_CONFIG },
            r#"
enum Foo {
    Bar,
    Baz,
}

impl Foo {
    fn qux(&self) {
        match self {
            $0_ => {}
        }
    }
}
            "#,
            r#"
enum Foo {
    Bar,
    Baz,
}

impl Foo {
    fn qux(&self) {
        match self {
            Self::Bar => ${1:todo!()},
            Self::Baz => ${2:todo!()},$0
        }
    }
}
            "#,
        );
    }

    #[test]
    fn prefer_self_with_generics() {
        check_assist_with_config(
            add_missing_match_arms,
            AssistConfig { prefer_self_ty: true, ..TEST_CONFIG },
            r#"
enum Foo<T> {
    Bar(T),
    Baz,
}

impl<T> Foo<T> {
    fn qux(&self) {
        match self {
            $0_ => {}
        }
    }
}
            "#,
            r#"
enum Foo<T> {
    Bar(T),
    Baz,
}

impl<T> Foo<T> {
    fn qux(&self) {
        match self {
            Self::Bar(${1:_}) => ${2:todo!()},
            Self::Baz => ${3:todo!()},$0
        }
    }
}
            "#,
        );
        check_assist_with_config(
            add_missing_match_arms,
            AssistConfig { prefer_self_ty: true, ..TEST_CONFIG },
            r#"
enum Foo<T> {
    Bar(T),
    Baz,
}

impl<T> Foo<T> {
    fn qux(v: Foo<i32>) {
        match v {
            $0_ => {}
        }
    }
}
            "#,
            r#"
enum Foo<T> {
    Bar(T),
    Baz,
}

impl<T> Foo<T> {
    fn qux(v: Foo<i32>) {
        match v {
            Foo::Bar(${1:_}) => ${2:todo!()},
            Foo::Baz => ${3:todo!()},$0
        }
    }
}
            "#,
        );
    }
}
