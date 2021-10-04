//! Diagnostic emitted for files that aren't part of any crate.

use hir::db::DefDatabase;
use ide_db::{
    base_db::{FileId, FileLoader, SourceDatabase, SourceDatabaseExt},
    source_change::SourceChange,
    RootDatabase,
};
use syntax::{
    ast::{self, HasModuleItem, HasName},
    AstNode, TextRange, TextSize,
};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext, Severity};

// Diagnostic: unlinked-file
//
// This diagnostic is shown for files that are not included in any crate, or files that are part of
// crates rust-analyzer failed to discover. The file will not have IDE features available.
pub(crate) fn unlinked_file(ctx: &DiagnosticsContext, acc: &mut Vec<Diagnostic>, file_id: FileId) {
    // Limit diagnostic to the first few characters in the file. This matches how VS Code
    // renders it with the full span, but on other editors, and is less invasive.
    let range = ctx.sema.db.parse(file_id).syntax_node().text_range();
    // FIXME: This is wrong if one of the first three characters is not ascii: `//Ы`.
    let range = range.intersect(TextRange::up_to(TextSize::of("..."))).unwrap_or(range);

    acc.push(
        Diagnostic::new("unlinked-file", "file not included in module tree", range)
            .severity(Severity::WeakWarning)
            .with_fixes(fixes(ctx, file_id)),
    );
}

fn fixes(ctx: &DiagnosticsContext, file_id: FileId) -> Option<Vec<Assist>> {
    // If there's an existing module that could add `mod` or `pub mod` items to include the unlinked file,
    // suggest that as a fix.

    let source_root = ctx.sema.db.source_root(ctx.sema.db.file_source_root(file_id));
    let our_path = source_root.path_for_file(&file_id)?;
    let (mut module_name, _) = our_path.name_and_extension()?;

    // Candidates to look for:
    // - `mod.rs`, `main.rs` and `lib.rs` in the same folder
    // - `$dir.rs` in the parent folder, where `$dir` is the directory containing `self.file_id`
    let parent = our_path.parent()?;
    let paths = {
        let parent = if module_name == "mod" {
            // for mod.rs we need to actually look up one higher
            // and take the parent as our to be module name
            let (name, _) = parent.name_and_extension()?;
            module_name = name;
            parent.parent()?
        } else {
            parent
        };
        let mut paths =
            vec![parent.join("mod.rs")?, parent.join("lib.rs")?, parent.join("main.rs")?];

        // `submod/bla.rs` -> `submod.rs`
        let parent_mod = (|| {
            let (name, _) = parent.name_and_extension()?;
            parent.parent()?.join(&format!("{}.rs", name))
        })();
        paths.extend(parent_mod);
        paths
    };

    for &parent_id in paths.iter().filter_map(|path| source_root.file_for_path(path)) {
        for &krate in ctx.sema.db.relevant_crates(parent_id).iter() {
            let crate_def_map = ctx.sema.db.crate_def_map(krate);
            for (_, module) in crate_def_map.modules() {
                if module.origin.is_inline() {
                    // We don't handle inline `mod parent {}`s, they use different paths.
                    continue;
                }

                if module.origin.file_id() == Some(parent_id) {
                    return make_fixes(ctx.sema.db, parent_id, module_name, file_id);
                }
            }
        }
    }

    None
}

fn make_fixes(
    db: &RootDatabase,
    parent_file_id: FileId,
    new_mod_name: &str,
    added_file_id: FileId,
) -> Option<Vec<Assist>> {
    fn is_outline_mod(item: &ast::Item) -> bool {
        matches!(item, ast::Item::Module(m) if m.item_list().is_none())
    }

    let mod_decl = format!("mod {};", new_mod_name);
    let pub_mod_decl = format!("pub mod {};", new_mod_name);

    let ast: ast::SourceFile = db.parse(parent_file_id).tree();

    let mut mod_decl_builder = TextEdit::builder();
    let mut pub_mod_decl_builder = TextEdit::builder();

    // If there's an existing `mod m;` statement matching the new one, don't emit a fix (it's
    // probably `#[cfg]`d out).
    for item in ast.items() {
        if let ast::Item::Module(m) = item {
            if let Some(name) = m.name() {
                if m.item_list().is_none() && name.to_string() == new_mod_name {
                    cov_mark::hit!(unlinked_file_skip_fix_when_mod_already_exists);
                    return None;
                }
            }
        }
    }

    // If there are existing `mod m;` items, append after them (after the first group of them, rather).
    match ast
        .items()
        .skip_while(|item| !is_outline_mod(item))
        .take_while(|item| is_outline_mod(item))
        .last()
    {
        Some(last) => {
            cov_mark::hit!(unlinked_file_append_to_existing_mods);
            let offset = last.syntax().text_range().end();
            mod_decl_builder.insert(offset, format!("\n{}", mod_decl));
            pub_mod_decl_builder.insert(offset, format!("\n{}", pub_mod_decl));
        }
        None => {
            // Prepend before the first item in the file.
            match ast.items().next() {
                Some(item) => {
                    cov_mark::hit!(unlinked_file_prepend_before_first_item);
                    let offset = item.syntax().text_range().start();
                    mod_decl_builder.insert(offset, format!("{}\n\n", mod_decl));
                    pub_mod_decl_builder.insert(offset, format!("{}\n\n", pub_mod_decl));
                }
                None => {
                    // No items in the file, so just append at the end.
                    cov_mark::hit!(unlinked_file_empty_file);
                    let offset = ast.syntax().text_range().end();
                    mod_decl_builder.insert(offset, format!("{}\n", mod_decl));
                    pub_mod_decl_builder.insert(offset, format!("{}\n", pub_mod_decl));
                }
            }
        }
    }

    let trigger_range = db.parse(added_file_id).tree().syntax().text_range();
    Some(vec![
        fix(
            "add_mod_declaration",
            &format!("Insert `{}`", mod_decl),
            SourceChange::from_text_edit(parent_file_id, mod_decl_builder.finish()),
            trigger_range,
        ),
        fix(
            "add_pub_mod_declaration",
            &format!("Insert `{}`", pub_mod_decl),
            SourceChange::from_text_edit(parent_file_id, pub_mod_decl_builder.finish()),
            trigger_range,
        ),
    ])
}

#[cfg(test)]
mod tests {

    use crate::tests::{check_diagnostics, check_fix, check_fixes, check_no_fix};

    #[test]
    fn unlinked_file_prepend_first_item() {
        cov_mark::check!(unlinked_file_prepend_before_first_item);
        // Only tests the first one for `pub mod` since the rest are the same
        check_fixes(
            r#"
//- /main.rs
fn f() {}
//- /foo.rs
$0
"#,
            vec![
                r#"
mod foo;

fn f() {}
"#,
                r#"
pub mod foo;

fn f() {}
"#,
            ],
        );
    }

    #[test]
    fn unlinked_file_append_mod() {
        cov_mark::check!(unlinked_file_append_to_existing_mods);
        check_fix(
            r#"
//- /main.rs
//! Comment on top

mod preexisting;

mod preexisting2;

struct S;

mod preexisting_bottom;)
//- /foo.rs
$0
"#,
            r#"
//! Comment on top

mod preexisting;

mod preexisting2;
mod foo;

struct S;

mod preexisting_bottom;)
"#,
        );
    }

    #[test]
    fn unlinked_file_insert_in_empty_file() {
        cov_mark::check!(unlinked_file_empty_file);
        check_fix(
            r#"
//- /main.rs
//- /foo.rs
$0
"#,
            r#"
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_insert_in_empty_file_mod_file() {
        check_fix(
            r#"
//- /main.rs
//- /foo/mod.rs
$0
"#,
            r#"
mod foo;
"#,
        );
        check_fix(
            r#"
//- /main.rs
mod bar;
//- /bar.rs
// bar module
//- /bar/foo/mod.rs
$0
"#,
            r#"
// bar module
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_old_style_modrs() {
        check_fix(
            r#"
//- /main.rs
mod submod;
//- /submod/mod.rs
// in mod.rs
//- /submod/foo.rs
$0
"#,
            r#"
// in mod.rs
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_new_style_mod() {
        check_fix(
            r#"
//- /main.rs
mod submod;
//- /submod.rs
//- /submod/foo.rs
$0
"#,
            r#"
mod foo;
"#,
        );
    }

    #[test]
    fn unlinked_file_with_cfg_off() {
        cov_mark::check!(unlinked_file_skip_fix_when_mod_already_exists);
        check_no_fix(
            r#"
//- /main.rs
#[cfg(never)]
mod foo;

//- /foo.rs
$0
"#,
        );
    }

    #[test]
    fn unlinked_file_with_cfg_on() {
        check_diagnostics(
            r#"
//- /main.rs
#[cfg(not(never))]
mod foo;

//- /foo.rs
"#,
        );
    }
}
