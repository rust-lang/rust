use crate::assist_context::{AssistContext, Assists};
use hir::HirDisplay;
use ide_db::{
    assists::{AssistId, AssistKind},
    defs::NameRefClass,
};
use syntax::{
    ast::{self, edit::IndentLevel},
    AstNode,
};

// Assist: generate_constant
//
// Generate a named constant.
//
// ```
// struct S { i: usize }
// impl S { pub fn new(n: usize) {} }
// fn main() {
//     let v = S::new(CAPA$0CITY);
// }
// ```
// ->
// ```
// struct S { i: usize }
// impl S { pub fn new(n: usize) {} }
// fn main() {
//     const CAPACITY: usize = $0;
//     let v = S::new(CAPACITY);
// }
// ```

pub(crate) fn generate_constant(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let constant_token = ctx.find_node_at_offset::<ast::NameRef>()?;
    let expr = constant_token.syntax().ancestors().find_map(ast::Expr::cast)?;
    let statement = expr.syntax().ancestors().find_map(ast::Stmt::cast)?;
    let ty = ctx.sema.type_of_expr(&expr)?;
    let scope = ctx.sema.scope(statement.syntax());
    let module = scope.module()?;
    let type_name = ty.original().display_source_code(ctx.db(), module.into()).ok()?;
    let indent = IndentLevel::from_node(statement.syntax());
    if constant_token.to_string().chars().any(|it| !(it.is_uppercase() || it == '_')) {
        cov_mark::hit!(not_constant_name);
        return None;
    }
    if NameRefClass::classify(&ctx.sema, &constant_token).is_some() {
        cov_mark::hit!(already_defined);
        return None;
    }
    let target = statement.syntax().parent()?.text_range();
    acc.add(
        AssistId("generate_constant", AssistKind::QuickFix),
        "Generate constant",
        target,
        |builder| {
            builder.insert(
                statement.syntax().text_range().start(),
                format!("const {}: {} = $0;\n{}", constant_token, type_name, indent),
            );
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_trivial() {
        check_assist(
            generate_constant,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    let v = S::new(CAPA$0CITY);
}"#,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    const CAPACITY: usize = $0;
    let v = S::new(CAPACITY);
}"#,
        );
    }
    #[test]
    fn test_wont_apply_when_defined() {
        cov_mark::check!(already_defined);
        check_assist_not_applicable(
            generate_constant,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    const CAPACITY: usize = 10;
    let v = S::new(CAPAC$0ITY);
}"#,
        );
    }
    #[test]
    fn test_wont_apply_when_maybe_not_constant() {
        cov_mark::check!(not_constant_name);
        check_assist_not_applicable(
            generate_constant,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    let v = S::new(capa$0city);
}"#,
        );
    }
}
