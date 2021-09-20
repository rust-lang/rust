use ide_db::{
    assists::{AssistId, AssistKind},
    base_db::AnchoredPathBuf,
};
use syntax::{
    ast::{self, Whitespace},
    AstNode, AstToken, SourceFile, TextRange, TextSize,
};

use crate::assist_context::{AssistContext, Assists};

/// Trim(remove leading and trailing whitespace) `initial_range` in `source_file`, return the trimmed range.
fn trimmed_text_range(source_file: &SourceFile, initial_range: TextRange) -> TextRange {
    let mut trimmed_range = initial_range;
    while source_file
        .syntax()
        .token_at_offset(trimmed_range.start())
        .find_map(Whitespace::cast)
        .is_some()
        && trimmed_range.start() < trimmed_range.end()
    {
        let start = trimmed_range.start() + TextSize::from(1);
        trimmed_range = TextRange::new(start, trimmed_range.end());
    }
    while source_file
        .syntax()
        .token_at_offset(trimmed_range.end())
        .find_map(Whitespace::cast)
        .is_some()
        && trimmed_range.start() < trimmed_range.end()
    {
        let end = trimmed_range.end() - TextSize::from(1);
        trimmed_range = TextRange::new(trimmed_range.start(), end);
    }
    trimmed_range
}

// Assist: promote_mod_file
//
// Moves inline module's contents to a separate file.
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
pub(crate) fn promote_mod_file(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let source_file = ctx.find_node_at_offset::<ast::SourceFile>()?;
    let module = ctx.sema.to_module_def(ctx.frange.file_id)?;
    // Enable this assist if the user select all "meaningful" content in the source file
    let trimmed_selected_range = trimmed_text_range(&source_file, ctx.frange.range);
    let trimmed_file_range = trimmed_text_range(&source_file, source_file.syntax().text_range());
    if module.is_mod_rs(ctx.db()) || trimmed_selected_range != trimmed_file_range {
        return None;
    }

    let target = TextRange::new(
        source_file.syntax().text_range().start(),
        source_file.syntax().text_range().end(),
    );
    let module_name = module.name(ctx.db())?.to_string();
    let path = format!("./{}/mod.rs", module_name);
    let dst = AnchoredPathBuf { anchor: ctx.frange.file_id, path };
    acc.add(
        AssistId("promote_mod_file", AssistKind::Refactor),
        format!("Turn {}.rs to {}/mod.rs", module_name, module_name),
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
$0"#,
            r#"
//- /a/mod.rs
fn t() {}
"#,
        );
    }

    #[test]
    fn must_select_all_file() {
        check_assist_not_applicable(
            promote_mod_file,
            r#"
//- /main.rs
mod a;
//- /a.rs
fn t() {}$0
"#,
        );
        check_assist_not_applicable(
            promote_mod_file,
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
