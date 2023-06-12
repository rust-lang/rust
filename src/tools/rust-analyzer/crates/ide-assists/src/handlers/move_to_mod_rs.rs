use ide_db::{
    assists::{AssistId, AssistKind},
    base_db::AnchoredPathBuf,
};
use syntax::{ast, AstNode};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::trimmed_text_range,
};

// Assist: move_to_mod_rs
//
// Moves xxx.rs to xxx/mod.rs.
//
// ```
// //- /main.rs
// mod a;
// //- /a.rs
// $0fn t() {}$0
// ```
// ->
// ```
// fn t() {}
// ```
pub(crate) fn move_to_mod_rs(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let source_file = ctx.find_node_at_offset::<ast::SourceFile>()?;
    let module = ctx.sema.to_module_def(ctx.file_id())?;
    // Enable this assist if the user select all "meaningful" content in the source file
    let trimmed_selected_range = trimmed_text_range(&source_file, ctx.selection_trimmed());
    let trimmed_file_range = trimmed_text_range(&source_file, source_file.syntax().text_range());
    if module.is_mod_rs(ctx.db()) {
        cov_mark::hit!(already_mod_rs);
        return None;
    }
    if trimmed_selected_range != trimmed_file_range {
        cov_mark::hit!(not_all_selected);
        return None;
    }

    let target = source_file.syntax().text_range();
    let module_name = module.name(ctx.db())?.display(ctx.db()).to_string();
    let path = format!("./{module_name}/mod.rs");
    let dst = AnchoredPathBuf { anchor: ctx.file_id(), path };
    acc.add(
        AssistId("move_to_mod_rs", AssistKind::Refactor),
        format!("Convert {module_name}.rs to {module_name}/mod.rs"),
        target,
        |builder| {
            builder.move_file(ctx.file_id(), dst);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn trivial() {
        check_assist(
            move_to_mod_rs,
            r#"
//- /main.rs
mod a;
//- /a.rs
$0fn t() {}
$0"#,
            r#"
//- /a/mod.rs
fn t() {}
"#,
        );
    }

    #[test]
    fn must_select_all_file() {
        cov_mark::check!(not_all_selected);
        check_assist_not_applicable(
            move_to_mod_rs,
            r#"
//- /main.rs
mod a;
//- /a.rs
fn t() {}$0
"#,
        );
        cov_mark::check!(not_all_selected);
        check_assist_not_applicable(
            move_to_mod_rs,
            r#"
//- /main.rs
mod a;
//- /a.rs
$0fn$0 t() {}
"#,
        );
    }

    #[test]
    fn cannot_promote_mod_rs() {
        cov_mark::check!(already_mod_rs);
        check_assist_not_applicable(
            move_to_mod_rs,
            r#"//- /main.rs
mod a;
//- /a/mod.rs
$0fn t() {}$0
"#,
        );
    }

    #[test]
    fn cannot_promote_main_and_lib_rs() {
        check_assist_not_applicable(
            move_to_mod_rs,
            r#"//- /main.rs
$0fn t() {}$0
"#,
        );
        check_assist_not_applicable(
            move_to_mod_rs,
            r#"//- /lib.rs
$0fn t() {}$0
"#,
        );
    }

    #[test]
    fn works_in_mod() {
        // note: /a/b.rs remains untouched
        check_assist(
            move_to_mod_rs,
            r#"//- /main.rs
mod a;
//- /a.rs
$0mod b;
fn t() {}$0
//- /a/b.rs
fn t1() {}
"#,
            r#"
//- /a/mod.rs
mod b;
fn t() {}
"#,
        );
    }
}
