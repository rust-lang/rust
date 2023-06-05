use crate::assist_context::{AssistContext, Assists};
use hir::{HasVisibility, HirDisplay, Module};
use ide_db::{
    assists::{AssistId, AssistKind},
    base_db::{FileId, Upcast},
    defs::{Definition, NameRefClass},
};
use syntax::{
    ast::{self, edit::IndentLevel, NameRef},
    AstNode, Direction, SyntaxKind, TextSize,
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

pub(crate) fn generate_constant(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let constant_token = ctx.find_node_at_offset::<ast::NameRef>()?;
    if constant_token.to_string().chars().any(|it| !(it.is_uppercase() || it == '_')) {
        cov_mark::hit!(not_constant_name);
        return None;
    }
    if NameRefClass::classify(&ctx.sema, &constant_token).is_some() {
        cov_mark::hit!(already_defined);
        return None;
    }
    let expr = constant_token.syntax().ancestors().find_map(ast::Expr::cast)?;
    let statement = expr.syntax().ancestors().find_map(ast::Stmt::cast)?;
    let ty = ctx.sema.type_of_expr(&expr)?;
    let scope = ctx.sema.scope(statement.syntax())?;
    let constant_module = scope.module();
    let type_name =
        ty.original().display_source_code(ctx.db(), constant_module.into(), false).ok()?;
    let target = statement.syntax().parent()?.text_range();
    let path = constant_token.syntax().ancestors().find_map(ast::Path::cast)?;

    let name_refs = path.segments().map(|s| s.name_ref());
    let mut outer_exists = false;
    let mut not_exist_name_ref = Vec::new();
    let mut current_module = constant_module;
    for name_ref in name_refs {
        let name_ref_value = name_ref?;
        let name_ref_class = NameRefClass::classify(&ctx.sema, &name_ref_value);
        match name_ref_class {
            Some(NameRefClass::Definition(Definition::Module(m))) => {
                if !m.visibility(ctx.sema.db).is_visible_from(ctx.sema.db, constant_module.into()) {
                    return None;
                }
                outer_exists = true;
                current_module = m;
            }
            Some(_) => {
                return None;
            }
            None => {
                not_exist_name_ref.push(name_ref_value);
            }
        }
    }
    let (offset, indent, file_id, post_string) =
        target_data_for_generate_constant(ctx, current_module, constant_module).unwrap_or_else(
            || {
                let indent = IndentLevel::from_node(statement.syntax());
                (statement.syntax().text_range().start(), indent, None, format!("\n{indent}"))
            },
        );

    let text = get_text_for_generate_constant(not_exist_name_ref, indent, outer_exists, type_name)?;
    acc.add(
        AssistId("generate_constant", AssistKind::QuickFix),
        "Generate constant",
        target,
        |builder| {
            if let Some(file_id) = file_id {
                builder.edit_file(file_id);
            }
            builder.insert(offset, format!("{text}{post_string}"));
        },
    )
}

fn get_text_for_generate_constant(
    mut not_exist_name_ref: Vec<NameRef>,
    indent: IndentLevel,
    outer_exists: bool,
    type_name: String,
) -> Option<String> {
    let constant_token = not_exist_name_ref.pop()?;
    let vis = if not_exist_name_ref.len() == 0 && !outer_exists { "" } else { "\npub " };
    let mut text = format!("{vis}const {constant_token}: {type_name} = $0;");
    while let Some(name_ref) = not_exist_name_ref.pop() {
        let vis = if not_exist_name_ref.len() == 0 && !outer_exists { "" } else { "\npub " };
        text = text.replace('\n', "\n    ");
        text = format!("{vis}mod {name_ref} {{{text}\n}}");
    }
    Some(text.replace('\n', &format!("\n{indent}")))
}

fn target_data_for_generate_constant(
    ctx: &AssistContext<'_>,
    current_module: Module,
    constant_module: Module,
) -> Option<(TextSize, IndentLevel, Option<FileId>, String)> {
    if current_module == constant_module {
        // insert in current file
        return None;
    }
    let in_file_source = current_module.definition_source(ctx.sema.db);
    let file_id = in_file_source.file_id.original_file(ctx.sema.db.upcast());
    match in_file_source.value {
        hir::ModuleSource::Module(module_node) => {
            let indent = IndentLevel::from_node(module_node.syntax());
            let l_curly_token = module_node.item_list()?.l_curly_token()?;
            let offset = l_curly_token.text_range().end();

            let siblings_has_newline = l_curly_token
                .siblings_with_tokens(Direction::Next)
                .find(|it| it.kind() == SyntaxKind::WHITESPACE && it.to_string().contains('\n'))
                .is_some();
            let post_string =
                if siblings_has_newline { format!("{indent}") } else { format!("\n{indent}") };
            Some((offset, indent + 1, Some(file_id), post_string))
        }
        _ => Some((TextSize::from(0), 0.into(), Some(file_id), "\n".into())),
    }
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

    #[test]
    fn test_constant_with_path() {
        check_assist(
            generate_constant,
            r#"mod foo {}
fn bar() -> i32 {
    foo::A_CON$0STANT
}"#,
            r#"mod foo {
    pub const A_CONSTANT: i32 = $0;
}
fn bar() -> i32 {
    foo::A_CONSTANT
}"#,
        );
    }

    #[test]
    fn test_constant_with_longer_path() {
        check_assist(
            generate_constant,
            r#"mod foo {
    pub mod goo {}
}
fn bar() -> i32 {
    foo::goo::A_CON$0STANT
}"#,
            r#"mod foo {
    pub mod goo {
        pub const A_CONSTANT: i32 = $0;
    }
}
fn bar() -> i32 {
    foo::goo::A_CONSTANT
}"#,
        );
    }

    #[test]
    fn test_constant_with_not_exist_longer_path() {
        check_assist(
            generate_constant,
            r#"fn bar() -> i32 {
    foo::goo::A_CON$0STANT
}"#,
            r#"mod foo {
    pub mod goo {
        pub const A_CONSTANT: i32 = $0;
    }
}
fn bar() -> i32 {
    foo::goo::A_CONSTANT
}"#,
        );
    }
}
