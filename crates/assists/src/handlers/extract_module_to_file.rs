use ast::edit::IndentLevel;
use ide_db::base_db::{AnchoredPathBuf, SourceDatabaseExt};
use syntax::{
    ast::{self, edit::AstNodeEdit, NameOwner},
    AstNode,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: extract_module_to_file
//
// This assist extract module to file.
//
// ```
// mod foo {<|>
//     fn t() {}
// }
// ```
// ->
// ```
// mod foo;
// ```
pub(crate) fn extract_module_to_file(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let assist_id = AssistId("extract_module_to_file", AssistKind::RefactorExtract);
    let assist_label = "Extract module to file";
    let db = ctx.db();
    let module_ast = ctx.find_node_at_offset::<ast::Module>()?;
    let module_items = module_ast.item_list()?;
    let dedent_module_items_text = module_items.dedent(IndentLevel(1)).to_string();
    let module_name = module_ast.name()?;
    let target = module_ast.syntax().text_range();
    let anchor_file_id = ctx.frange.file_id;
    let sr = db.file_source_root(anchor_file_id);
    let sr = db.source_root(sr);
    let file_path = sr.path_for_file(&anchor_file_id)?;
    let (file_name, file_ext) = file_path.name_and_extension()?;
    acc.add(assist_id, assist_label, target, |builder| {
        builder.replace(target, format!("mod {};", module_name));
        let path = if is_main_or_lib(file_name) {
            format!("./{}.{}", module_name, file_ext.unwrap())
        } else {
            format!("./{}/{}.{}", file_name, module_name, file_ext.unwrap())
        };
        let dst = AnchoredPathBuf { anchor: anchor_file_id, path };
        let contents = update_module_items_string(dedent_module_items_text);
        builder.create_file(dst, contents);
    })
}
fn is_main_or_lib(file_name: &str) -> bool {
    file_name == "main".to_string() || file_name == "lib".to_string()
}
fn update_module_items_string(items_str: String) -> String {
    let mut items_string_lines: Vec<&str> = items_str.lines().collect();
    items_string_lines.pop(); // Delete last line
    items_string_lines.reverse();
    items_string_lines.pop(); // Delete first line
    items_string_lines.reverse();

    let string = items_string_lines.join("\n");
    format!("{}", string)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn extract_module_to_file_with_basic_module() {
        check_assist(
            extract_module_to_file,
            r#"
//- /foo.rs crate:foo
mod tests {<|>
    #[test] fn t() {}
}
"#,
            r#"
//- /foo.rs
mod tests;
//- /foo/tests.rs
#[test] fn t() {}"#,
        )
    }

    #[test]
    fn extract_module_to_file_with_file_path() {
        check_assist(
            extract_module_to_file,
            r#"
//- /src/foo.rs crate:foo
mod bar {<|>
    fn f() {

    }
}
fn main() {
    println!("Hello, world!");
}
"#,
            r#"
//- /src/foo.rs
mod bar;
fn main() {
    println!("Hello, world!");
}
//- /src/foo/bar.rs
fn f() {

}"#,
        )
    }

    #[test]
    fn extract_module_to_file_with_main_filw() {
        check_assist(
            extract_module_to_file,
            r#"
//- /main.rs
mod foo {<|>
    fn f() {

    }
}
fn main() {
    println!("Hello, world!");
}
"#,
            r#"
//- /main.rs
mod foo;
fn main() {
    println!("Hello, world!");
}
//- /foo.rs
fn f() {

}"#,
        )
    }

    #[test]
    fn extract_module_to_file_with_lib_file() {
        check_assist(
            extract_module_to_file,
            r#"
//- /lib.rs
mod foo {<|>
    fn f() {

    }
}
fn main() {
    println!("Hello, world!");
}
"#,
            r#"
//- /lib.rs
mod foo;
fn main() {
    println!("Hello, world!");
}
//- /foo.rs
fn f() {

}"#,
        )
    }
}
