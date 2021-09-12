use ide_db::{
    assists::{AssistId, AssistKind},
    base_db::AnchoredPathBuf,
};
use syntax::{
    ast::{self},
    AstNode, TextRange,
};

use crate::assist_context::{AssistContext, Assists};

// Assist: promote_mod_file
//
// Moves inline module's contents to a separate file.
//
// ```
// // a.rs
// $0fn t() {}
// ```
// ->
// ```
// // /a/mod.rs
// fn t() {}
// ```
pub(crate) fn promote_mod_file(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let source_file = ctx.find_node_at_offset::<ast::SourceFile>()?;
    let module = ctx.sema.to_module_def(ctx.frange.file_id)?;
    if module.is_mod_rs(ctx.db()) {
        return None;
    }
    let target = TextRange::new(
        source_file.syntax().text_range().start(),
        source_file.syntax().text_range().end(),
    );
    let path = format!("./{}/mod.rs", module.name(ctx.db())?.to_string());
    let dst = AnchoredPathBuf { anchor: ctx.frange.file_id, path };
    acc.add(
        AssistId("promote_mod_file", AssistKind::Refactor),
        "Promote Module to directory",
        target,
        |builder| {
            builder.move_file(ctx.frange.file_id, dst);
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
            promote_mod_file,
            r#"
//- /main.rs
mod a;
//- /a.rs
$0fn t() {}
"#,
            r#"
//- /a/mod.rs
fn t() {}
"#,
        );
    }

    #[test]
    fn cursor_can_be_putted_anywhere() {
        check_assist(
            promote_mod_file,
            r#"
//- /main.rs
mod a;
//- /a.rs
fn t() {}$0
"#,
            r#"
//- /a/mod.rs
fn t() {}
"#,
        );
        check_assist(
            promote_mod_file,
            r#"
//- /main.rs
mod a;
//- /a.rs
fn t()$0 {}
"#,
            r#"
//- /a/mod.rs
fn t() {}
"#,
        );
        check_assist(
            promote_mod_file,
            r#"
//- /main.rs
mod a;
//- /a.rs
fn t($0) {}
"#,
            r#"
//- /a/mod.rs
fn t() {}
"#,
        );
    }

    #[test]
    fn cannot_promote_mod_rs() {
        check_assist_not_applicable(
            promote_mod_file,
            r#"//- /main.rs
mod a;
//- /a/mod.rs
$0fn t() {}
"#,
        );
    }

    #[test]
    fn cannot_promote_main_and_lib_rs() {
        check_assist_not_applicable(
            promote_mod_file,
            r#"//- /main.rs
$0fn t() {}
"#,
        );
        check_assist_not_applicable(
            promote_mod_file,
            r#"//- /lib.rs
$0fn t() {}
"#,
        );
    }

    #[test]
    fn works_in_mod() {
        // note: /a/b.rs remains untouched
        check_assist(
            promote_mod_file,
            r#"//- /main.rs
mod a;
//- /a.rs
mod b;
$0fn t() {}
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
