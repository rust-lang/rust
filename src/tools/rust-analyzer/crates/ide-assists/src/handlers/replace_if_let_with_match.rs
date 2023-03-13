use std::iter::{self, successors};

use either::Either;
use ide_db::{
    defs::NameClass,
    syntax_helpers::node_ext::{is_pattern_cond, single_let},
    ty_filter::TryEnum,
    RootDatabase,
};
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, HasName,
    },
    AstNode, TextRange, T,
};

use crate::{
    utils::{does_nested_pattern, does_pat_match_variant, unwrap_trivial_block},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: replace_if_let_with_match
//
// Replaces a `if let` expression with a `match` expression.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     $0if let Action::Move { distance } = action {
//         foo(distance)
//     } else {
//         bar()
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => foo(distance),
//         _ => bar(),
//     }
// }
// ```
pub(crate) fn replace_if_let_with_match(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let if_expr: ast::IfExpr = ctx.find_node_at_offset()?;
    let available_range = TextRange::new(
        if_expr.syntax().text_range().start(),
        if_expr.then_branch()?.syntax().text_range().start(),
    );
    let cursor_in_range = available_range.contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }
    let mut else_block = None;
    let if_exprs = successors(Some(if_expr.clone()), |expr| match expr.else_branch()? {
        ast::ElseBranch::IfExpr(expr) => Some(expr),
        ast::ElseBranch::Block(block) => {
            else_block = Some(block);
            None
        }
    });
    let scrutinee_to_be_expr = if_expr.condition()?;
    let scrutinee_to_be_expr = match single_let(scrutinee_to_be_expr.clone()) {
        Some(cond) => cond.expr()?,
        None => scrutinee_to_be_expr,
    };

    let mut pat_seen = false;
    let mut cond_bodies = Vec::new();
    for if_expr in if_exprs {
        let cond = if_expr.condition()?;
        let cond = match single_let(cond.clone()) {
            Some(let_) => {
                let pat = let_.pat()?;
                let expr = let_.expr()?;
                // FIXME: If one `let` is wrapped in parentheses and the second is not,
                // we'll exit here.
                if scrutinee_to_be_expr.syntax().text() != expr.syntax().text() {
                    // Only if all condition expressions are equal we can merge them into a match
                    return None;
                }
                pat_seen = true;
                Either::Left(pat)
            }
            // Multiple `let`, unsupported.
            None if is_pattern_cond(cond.clone()) => return None,
            None => Either::Right(cond),
        };
        let body = if_expr.then_branch()?;
        cond_bodies.push((cond, body));
    }

    if !pat_seen && cond_bodies.len() != 1 {
        // Don't offer turning an if (chain) without patterns into a match,
        // unless its a simple `if cond { .. } (else { .. })`
        return None;
    }

    let let_ = if pat_seen { " let" } else { "" };

    acc.add(
        AssistId("replace_if_let_with_match", AssistKind::RefactorRewrite),
        format!("Replace if{let_} with match"),
        available_range,
        move |edit| {
            let match_expr = {
                let else_arm = make_else_arm(ctx, else_block, &cond_bodies);
                let make_match_arm = |(pat, body): (_, ast::BlockExpr)| {
                    let body = body.reset_indent().indent(IndentLevel(1));
                    match pat {
                        Either::Left(pat) => {
                            make::match_arm(iter::once(pat), None, unwrap_trivial_block(body))
                        }
                        Either::Right(_) if !pat_seen => make::match_arm(
                            iter::once(make::literal_pat("true").into()),
                            None,
                            unwrap_trivial_block(body),
                        ),
                        Either::Right(expr) => make::match_arm(
                            iter::once(make::wildcard_pat().into()),
                            Some(expr),
                            unwrap_trivial_block(body),
                        ),
                    }
                };
                let arms = cond_bodies.into_iter().map(make_match_arm).chain(iter::once(else_arm));
                let match_expr = make::expr_match(scrutinee_to_be_expr, make::match_arm_list(arms));
                match_expr.indent(IndentLevel::from_node(if_expr.syntax()))
            };

            let has_preceding_if_expr =
                if_expr.syntax().parent().map_or(false, |it| ast::IfExpr::can_cast(it.kind()));
            let expr = if has_preceding_if_expr {
                // make sure we replace the `else if let ...` with a block so we don't end up with `else expr`
                make::block_expr(None, Some(match_expr)).into()
            } else {
                match_expr
            };
            edit.replace_ast::<ast::Expr>(if_expr.into(), expr);
        },
    )
}

fn make_else_arm(
    ctx: &AssistContext<'_>,
    else_block: Option<ast::BlockExpr>,
    conditionals: &[(Either<ast::Pat, ast::Expr>, ast::BlockExpr)],
) -> ast::MatchArm {
    let (pattern, expr) = if let Some(else_block) = else_block {
        let pattern = match conditionals {
            [(Either::Right(_), _)] => make::literal_pat("false").into(),
            [(Either::Left(pat), _)] => match ctx
                .sema
                .type_of_pat(pat)
                .and_then(|ty| TryEnum::from_ty(&ctx.sema, &ty.adjusted()))
            {
                Some(it) => {
                    if does_pat_match_variant(pat, &it.sad_pattern()) {
                        it.happy_pattern_wildcard()
                    } else if does_nested_pattern(pat) {
                        make::wildcard_pat().into()
                    } else {
                        it.sad_pattern()
                    }
                }
                None => make::wildcard_pat().into(),
            },
            _ => make::wildcard_pat().into(),
        };
        (pattern, unwrap_trivial_block(else_block))
    } else {
        let pattern = match conditionals {
            [(Either::Right(_), _)] => make::literal_pat("false").into(),
            _ => make::wildcard_pat().into(),
        };
        (pattern, make::expr_unit())
    };
    make::match_arm(iter::once(pattern), None, expr)
}

// Assist: replace_match_with_if_let
//
// Replaces a binary `match` with a wildcard pattern and no guards with an `if let` expression.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     $0match action {
//         Action::Move { distance } => foo(distance),
//         _ => bar(),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     if let Action::Move { distance } = action {
//         foo(distance)
//     } else {
//         bar()
//     }
// }
// ```
pub(crate) fn replace_match_with_if_let(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let match_expr: ast::MatchExpr = ctx.find_node_at_offset()?;
    let match_arm_list = match_expr.match_arm_list()?;
    let available_range = TextRange::new(
        match_expr.syntax().text_range().start(),
        match_arm_list.syntax().text_range().start(),
    );
    let cursor_in_range = available_range.contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let mut arms = match_arm_list.arms();
    let (first_arm, second_arm) = (arms.next()?, arms.next()?);
    if arms.next().is_some() || first_arm.guard().is_some() || second_arm.guard().is_some() {
        return None;
    }

    let (if_let_pat, then_expr, else_expr) = pick_pattern_and_expr_order(
        &ctx.sema,
        first_arm.pat()?,
        second_arm.pat()?,
        first_arm.expr()?,
        second_arm.expr()?,
    )?;
    let scrutinee = match_expr.expr()?;

    let let_ = match &if_let_pat {
        ast::Pat::LiteralPat(p)
            if p.literal()
                .map(|it| it.token().kind())
                .map_or(false, |it| it == T![true] || it == T![false]) =>
        {
            ""
        }
        _ => " let",
    };
    let target = match_expr.syntax().text_range();
    acc.add(
        AssistId("replace_match_with_if_let", AssistKind::RefactorRewrite),
        format!("Replace match with if{let_}"),
        target,
        move |edit| {
            fn make_block_expr(expr: ast::Expr) -> ast::BlockExpr {
                // Blocks with modifiers (unsafe, async, etc.) are parsed as BlockExpr, but are
                // formatted without enclosing braces. If we encounter such block exprs,
                // wrap them in another BlockExpr.
                match expr {
                    ast::Expr::BlockExpr(block) if block.modifier().is_none() => block,
                    expr => make::block_expr(iter::empty(), Some(expr)),
                }
            }

            let condition = match if_let_pat {
                ast::Pat::LiteralPat(p)
                    if p.literal().map_or(false, |it| it.token().kind() == T![true]) =>
                {
                    scrutinee
                }
                ast::Pat::LiteralPat(p)
                    if p.literal().map_or(false, |it| it.token().kind() == T![false]) =>
                {
                    make::expr_prefix(T![!], scrutinee)
                }
                _ => make::expr_let(if_let_pat, scrutinee).into(),
            };
            let then_block = make_block_expr(then_expr.reset_indent());
            let else_expr = if is_empty_expr(&else_expr) { None } else { Some(else_expr) };
            let if_let_expr = make::expr_if(
                condition.into(),
                then_block,
                else_expr.map(make_block_expr).map(ast::ElseBranch::Block),
            )
            .indent(IndentLevel::from_node(match_expr.syntax()));

            edit.replace_ast::<ast::Expr>(match_expr.into(), if_let_expr);
        },
    )
}

/// Pick the pattern for the if let condition and return the expressions for the `then` body and `else` body in that order.
fn pick_pattern_and_expr_order(
    sema: &hir::Semantics<'_, RootDatabase>,
    pat: ast::Pat,
    pat2: ast::Pat,
    expr: ast::Expr,
    expr2: ast::Expr,
) -> Option<(ast::Pat, ast::Expr, ast::Expr)> {
    let res = match (pat, pat2) {
        (ast::Pat::WildcardPat(_), _) => return None,
        (pat, ast::Pat::WildcardPat(_)) => (pat, expr, expr2),
        (pat, _) if is_empty_expr(&expr2) => (pat, expr, expr2),
        (_, pat) if is_empty_expr(&expr) => (pat, expr2, expr),
        (pat, pat2) => match (binds_name(sema, &pat), binds_name(sema, &pat2)) {
            (true, true) => return None,
            (true, false) => (pat, expr, expr2),
            (false, true) => (pat2, expr2, expr),
            _ if is_sad_pat(sema, &pat) => (pat2, expr2, expr),
            (false, false) => (pat, expr, expr2),
        },
    };
    Some(res)
}

fn is_empty_expr(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::BlockExpr(expr) => match expr.stmt_list() {
            Some(it) => it.statements().next().is_none() && it.tail_expr().is_none(),
            None => true,
        },
        ast::Expr::TupleExpr(expr) => expr.fields().next().is_none(),
        _ => false,
    }
}

fn binds_name(sema: &hir::Semantics<'_, RootDatabase>, pat: &ast::Pat) -> bool {
    let binds_name_v = |pat| binds_name(sema, &pat);
    match pat {
        ast::Pat::IdentPat(pat) => !matches!(
            pat.name().and_then(|name| NameClass::classify(sema, &name)),
            Some(NameClass::ConstReference(_))
        ),
        ast::Pat::MacroPat(_) => true,
        ast::Pat::OrPat(pat) => pat.pats().any(binds_name_v),
        ast::Pat::SlicePat(pat) => pat.pats().any(binds_name_v),
        ast::Pat::TuplePat(it) => it.fields().any(binds_name_v),
        ast::Pat::TupleStructPat(it) => it.fields().any(binds_name_v),
        ast::Pat::RecordPat(it) => it
            .record_pat_field_list()
            .map_or(false, |rpfl| rpfl.fields().flat_map(|rpf| rpf.pat()).any(binds_name_v)),
        ast::Pat::RefPat(pat) => pat.pat().map_or(false, binds_name_v),
        ast::Pat::BoxPat(pat) => pat.pat().map_or(false, binds_name_v),
        ast::Pat::ParenPat(pat) => pat.pat().map_or(false, binds_name_v),
        _ => false,
    }
}

fn is_sad_pat(sema: &hir::Semantics<'_, RootDatabase>, pat: &ast::Pat) -> bool {
    sema.type_of_pat(pat)
        .and_then(|ty| TryEnum::from_ty(sema, &ty.adjusted()))
        .map_or(false, |it| does_pat_match_variant(pat, &it.sad_pattern()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn test_if_let_with_match_unapplicable_for_simple_ifs() {
        check_assist_not_applicable(
            replace_if_let_with_match,
            r#"
fn main() {
    if $0true {} else if false {} else {}
}
"#,
        )
    }

    #[test]
    fn test_if_with_match_no_else() {
        check_assist(
            replace_if_let_with_match,
            r#"
pub fn foo(foo: bool) {
    if foo$0 {
        self.foo();
    }
}
"#,
            r#"
pub fn foo(foo: bool) {
    match foo {
        true => {
            self.foo();
        }
        false => (),
    }
}
"#,
        )
    }

    #[test]
    fn test_if_with_match_with_else() {
        check_assist(
            replace_if_let_with_match,
            r#"
pub fn foo(foo: bool) {
    if foo$0 {
        self.foo();
    } else {
        self.bar();
    }
}
"#,
            r#"
pub fn foo(foo: bool) {
    match foo {
        true => {
            self.foo();
        }
        false => {
            self.bar();
        }
    }
}
"#,
        )
    }

    #[test]
    fn test_if_let_with_match_no_else() {
        check_assist(
            replace_if_let_with_match,
            r#"
impl VariantData {
    pub fn foo(&self) {
        if $0let VariantData::Struct(..) = *self {
            self.foo();
        }
    }
}
"#,
            r#"
impl VariantData {
    pub fn foo(&self) {
        match *self {
            VariantData::Struct(..) => {
                self.foo();
            }
            _ => (),
        }
    }
}
"#,
        )
    }

    #[test]
    fn test_if_let_with_match_available_range_left() {
        check_assist_not_applicable(
            replace_if_let_with_match,
            r#"
impl VariantData {
    pub fn foo(&self) {
        $0 if let VariantData::Struct(..) = *self {
            self.foo();
        }
    }
}
"#,
        )
    }

    #[test]
    fn test_if_let_with_match_available_range_right() {
        check_assist_not_applicable(
            replace_if_let_with_match,
            r#"
impl VariantData {
    pub fn foo(&self) {
        if let VariantData::Struct(..) = *self {$0
            self.foo();
        }
    }
}
"#,
        )
    }

    #[test]
    fn test_if_let_with_match_let_chain() {
        check_assist_not_applicable(
            replace_if_let_with_match,
            r#"
fn main() {
    if $0let true = true && let Some(1) = None {}
}
"#,
        )
    }

    #[test]
    fn test_if_let_with_match_basic() {
        check_assist(
            replace_if_let_with_match,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if $0let VariantData::Struct(..) = *self {
            true
        } else if let VariantData::Tuple(..) = *self {
            false
        } else if cond() {
            true
        } else {
            bar(
                123
            )
        }
    }
}
"#,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        match *self {
            VariantData::Struct(..) => true,
            VariantData::Tuple(..) => false,
            _ if cond() => true,
            _ => {
                    bar(
                        123
                    )
                }
        }
    }
}
"#,
        )
    }

    #[test]
    fn test_if_let_with_match_on_tail_if_let() {
        check_assist(
            replace_if_let_with_match,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else if let$0 VariantData::Tuple(..) = *self {
            false
        } else {
            false
        }
    }
}
"#,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
    match *self {
            VariantData::Tuple(..) => false,
            _ => false,
        }
}
    }
}
"#,
        )
    }

    #[test]
    fn special_case_option() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: option
fn foo(x: Option<i32>) {
    $0if let Some(x) = x {
        println!("{}", x)
    } else {
        println!("none")
    }
}
"#,
            r#"
fn foo(x: Option<i32>) {
    match x {
        Some(x) => println!("{}", x),
        None => println!("none"),
    }
}
"#,
        );
    }

    #[test]
    fn special_case_inverted_option() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: option
fn foo(x: Option<i32>) {
    $0if let None = x {
        println!("none")
    } else {
        println!("some")
    }
}
"#,
            r#"
fn foo(x: Option<i32>) {
    match x {
        None => println!("none"),
        Some(_) => println!("some"),
    }
}
"#,
        );
    }

    #[test]
    fn special_case_result() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn foo(x: Result<i32, ()>) {
    $0if let Ok(x) = x {
        println!("{}", x)
    } else {
        println!("none")
    }
}
"#,
            r#"
fn foo(x: Result<i32, ()>) {
    match x {
        Ok(x) => println!("{}", x),
        Err(_) => println!("none"),
    }
}
"#,
        );
    }

    #[test]
    fn special_case_inverted_result() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn foo(x: Result<i32, ()>) {
    $0if let Err(x) = x {
        println!("{}", x)
    } else {
        println!("ok")
    }
}
"#,
            r#"
fn foo(x: Result<i32, ()>) {
    match x {
        Err(x) => println!("{}", x),
        Ok(_) => println!("ok"),
    }
}
"#,
        );
    }

    #[test]
    fn nested_indent() {
        check_assist(
            replace_if_let_with_match,
            r#"
fn main() {
    if true {
        $0if let Ok(rel_path) = path.strip_prefix(root_path) {
            let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
            Some((*id, rel_path))
        } else {
            None
        }
    }
}
"#,
            r#"
fn main() {
    if true {
        match path.strip_prefix(root_path) {
            Ok(rel_path) => {
                let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
                Some((*id, rel_path))
            }
            _ => None,
        }
    }
}
"#,
        )
    }

    #[test]
    fn nested_type() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn foo(x: Result<i32, ()>) {
    let bar: Result<_, ()> = Ok(Some(1));
    $0if let Ok(Some(_)) = bar {
        ()
    } else {
        ()
    }
}
"#,
            r#"
fn foo(x: Result<i32, ()>) {
    let bar: Result<_, ()> = Ok(Some(1));
    match bar {
        Ok(Some(_)) => (),
        _ => (),
    }
}
"#,
        );
    }

    #[test]
    fn test_replace_match_with_if_let_unwraps_simple_expressions() {
        check_assist(
            replace_match_with_if_let,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        $0match *self {
            VariantData::Struct(..) => true,
            _ => false,
        }
    }
}           "#,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
}           "#,
        )
    }

    #[test]
    fn test_replace_match_with_if_let_doesnt_unwrap_multiline_expressions() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn foo() {
    $0match a {
        VariantData::Struct(..) => {
            bar(
                123
            )
        }
        _ => false,
    }
}           "#,
            r#"
fn foo() {
    if let VariantData::Struct(..) = a {
        bar(
            123
        )
    } else {
        false
    }
}           "#,
        )
    }

    #[test]
    fn replace_match_with_if_let_target() {
        check_assist_target(
            replace_match_with_if_let,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        $0match *self {
            VariantData::Struct(..) => true,
            _ => false,
        }
    }
}           "#,
            r#"match *self {
            VariantData::Struct(..) => true,
            _ => false,
        }"#,
        );
    }

    #[test]
    fn special_case_option_match_to_if_let() {
        check_assist(
            replace_match_with_if_let,
            r#"
//- minicore: option
fn foo(x: Option<i32>) {
    $0match x {
        Some(x) => println!("{}", x),
        None => println!("none"),
    }
}
"#,
            r#"
fn foo(x: Option<i32>) {
    if let Some(x) = x {
        println!("{}", x)
    } else {
        println!("none")
    }
}
"#,
        );
    }

    #[test]
    fn special_case_result_match_to_if_let() {
        check_assist(
            replace_match_with_if_let,
            r#"
//- minicore: result
fn foo(x: Result<i32, ()>) {
    $0match x {
        Ok(x) => println!("{}", x),
        Err(_) => println!("none"),
    }
}
"#,
            r#"
fn foo(x: Result<i32, ()>) {
    if let Ok(x) = x {
        println!("{}", x)
    } else {
        println!("none")
    }
}
"#,
        );
    }

    #[test]
    fn nested_indent_match_to_if_let() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    if true {
        $0match path.strip_prefix(root_path) {
            Ok(rel_path) => {
                let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
                Some((*id, rel_path))
            }
            _ => None,
        }
    }
}
"#,
            r#"
fn main() {
    if true {
        if let Ok(rel_path) = path.strip_prefix(root_path) {
            let rel_path = RelativePathBuf::from_path(rel_path).ok()?;
            Some((*id, rel_path))
        } else {
            None
        }
    }
}
"#,
        )
    }

    #[test]
    fn replace_match_with_if_let_empty_wildcard_expr() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    $0match path.strip_prefix(root_path) {
        Ok(rel_path) => println!("{}", rel_path),
        _ => (),
    }
}
"#,
            r#"
fn main() {
    if let Ok(rel_path) = path.strip_prefix(root_path) {
        println!("{}", rel_path)
    }
}
"#,
        )
    }

    #[test]
    fn replace_match_with_if_let_number_body() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    $0match Ok(()) {
        Ok(()) => {},
        Err(_) => 0,
    }
}
"#,
            r#"
fn main() {
    if let Err(_) = Ok(()) {
        0
    }
}
"#,
        )
    }

    #[test]
    fn replace_match_with_if_let_exhaustive() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn print_source(def_source: ModuleSource) {
    match def_so$0urce {
        ModuleSource::SourceFile(..) => { println!("source file"); }
        ModuleSource::Module(..) => { println!("module"); }
    }
}
"#,
            r#"
fn print_source(def_source: ModuleSource) {
    if let ModuleSource::SourceFile(..) = def_source { println!("source file"); } else { println!("module"); }
}
"#,
        )
    }

    #[test]
    fn replace_match_with_if_let_prefer_name_bind() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn foo() {
    match $0Foo(0) {
        Foo(_) => (),
        Bar(bar) => println!("bar {}", bar),
    }
}
"#,
            r#"
fn foo() {
    if let Bar(bar) = Foo(0) {
        println!("bar {}", bar)
    }
}
"#,
        );
        check_assist(
            replace_match_with_if_let,
            r#"
fn foo() {
    match $0Foo(0) {
        Bar(bar) => println!("bar {}", bar),
        Foo(_) => (),
    }
}
"#,
            r#"
fn foo() {
    if let Bar(bar) = Foo(0) {
        println!("bar {}", bar)
    }
}
"#,
        );
    }

    #[test]
    fn replace_match_with_if_let_prefer_nonempty_body() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn foo() {
    match $0Ok(0) {
        Ok(value) => {},
        Err(err) => eprintln!("{}", err),
    }
}
"#,
            r#"
fn foo() {
    if let Err(err) = Ok(0) {
        eprintln!("{}", err)
    }
}
"#,
        );
        check_assist(
            replace_match_with_if_let,
            r#"
fn foo() {
    match $0Ok(0) {
        Err(err) => eprintln!("{}", err),
        Ok(value) => {},
    }
}
"#,
            r#"
fn foo() {
    if let Err(err) = Ok(0) {
        eprintln!("{}", err)
    }
}
"#,
        );
    }

    #[test]
    fn replace_match_with_if_let_rejects_double_name_bindings() {
        check_assist_not_applicable(
            replace_match_with_if_let,
            r#"
fn foo() {
    match $0Foo(0) {
        Foo(foo) => println!("bar {}", foo),
        Bar(bar) => println!("bar {}", bar),
    }
}
"#,
        );
    }

    #[test]
    fn test_replace_match_with_if_let_keeps_unsafe_block() {
        check_assist(
            replace_match_with_if_let,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        $0match *self {
            VariantData::Struct(..) => true,
            _ => unsafe { unreachable_unchecked() },
        }
    }
}           "#,
            r#"
impl VariantData {
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
            unsafe { unreachable_unchecked() }
        }
    }
}           "#,
        )
    }

    #[test]
    fn test_replace_match_with_if_let_forces_else() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    match$0 0 {
        0 => (),
        _ => code(),
    }
}
"#,
            r#"
fn main() {
    if let 0 = 0 {
        ()
    } else {
        code()
    }
}
"#,
        )
    }

    #[test]
    fn test_replace_match_with_if_bool() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    match$0 b {
        true => (),
        _ => code(),
    }
}
"#,
            r#"
fn main() {
    if b {
        ()
    } else {
        code()
    }
}
"#,
        );
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    match$0 b {
        false => code(),
        true => (),
    }
}
"#,
            r#"
fn main() {
    if !b {
        code()
    }
}
"#,
        );
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    match$0 b {
        false => (),
        true => code(),
    }
}
"#,
            r#"
fn main() {
    if b {
        code()
    }
}
"#,
        )
    }
}
