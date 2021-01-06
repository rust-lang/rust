use ast::edit::IndentLevel;
use ide_db::base_db::AnchoredPathBuf;
use syntax::{
    ast::{self, edit::AstNodeEdit, NameOwner},
    AstNode, TextRange,
};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: move_module_to_file
//
// Moves inline module's contents to a separate file.
//
// ```
// mod $0foo {
//     fn t() {}
// }
// ```
// ->
// ```
// mod foo;
// ```
pub(crate) fn move_module_to_file(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let module_ast = ctx.find_node_at_offset::<ast::Module>()?;
    let module_items = module_ast.item_list()?;

    let l_curly_offset = module_items.syntax().text_range().start();
    if l_curly_offset <= ctx.offset() {
        mark::hit!(available_before_curly);
        return None;
    }
    let target = TextRange::new(module_ast.syntax().text_range().start(), l_curly_offset);

    let module_name = module_ast.name()?;

    let module_def = ctx.sema.to_def(&module_ast)?;
    let parent_module = module_def.parent(ctx.db())?;

    acc.add(
        AssistId("move_module_to_file", AssistKind::RefactorExtract),
        "Extract module to file",
        target,
        |builder| {
            let path = {
                let dir = match parent_module.name(ctx.db()) {
                    Some(name) if !parent_module.is_mod_rs(ctx.db()) => format!("{}/", name),
                    _ => String::new(),
                };
                format!("./{}{}.rs", dir, module_name)
            };
            let contents = {
                let items = module_items.dedent(IndentLevel(1)).to_string();
                let mut items =
                    items.trim_start_matches('{').trim_end_matches('}').trim().to_string();
                if !items.is_empty() {
                    items.push('\n');
                }
                items
            };

            builder.replace(module_ast.syntax().text_range(), format!("mod {};", module_name));

            let dst = AnchoredPathBuf { anchor: ctx.frange.file_id, path };
            builder.create_file(dst, contents);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn extract_from_root() {
        check_assist(
            move_module_to_file,
            r#"
mod $0tests {
    #[test] fn t() {}
}
"#,
            r#"
//- /main.rs
mod tests;
//- /tests.rs
#[test] fn t() {}
"#,
        );
    }

    #[test]
    fn extract_from_submodule() {
        check_assist(
            move_module_to_file,
            r#"
//- /main.rs
mod submod;
//- /submod.rs
$0mod inner {
    fn f() {}
}
fn g() {}
"#,
            r#"
//- /submod.rs
mod inner;
fn g() {}
//- /submod/inner.rs
fn f() {}
"#,
        );
    }

    #[test]
    fn extract_from_mod_rs() {
        check_assist(
            move_module_to_file,
            r#"
//- /main.rs
mod submodule;
//- /submodule/mod.rs
mod inner$0 {
    fn f() {}
}
fn g() {}
"#,
            r#"
//- /submodule/mod.rs
mod inner;
fn g() {}
//- /submodule/inner.rs
fn f() {}
"#,
        );
    }

    #[test]
    fn available_before_curly() {
        mark::check!(available_before_curly);
        check_assist_not_applicable(move_module_to_file, r#"mod m { $0 }"#);
    }
}
