//! Diagnostic emitted for files that aren't part of any crate.

use std::iter;

use hir::{db::DefDatabase, InFile, ModuleSource};
use ide_db::{
    base_db::{FileId, FileLoader, SourceDatabase, SourceDatabaseExt},
    source_change::SourceChange,
    RootDatabase,
};
use syntax::{
    ast::{self, edit::IndentLevel, HasModuleItem, HasName},
    AstNode, TextRange, TextSize,
};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext, Severity};

// Diagnostic: unlinked-file
//
// This diagnostic is shown for files that are not included in any crate, or files that are part of
// crates rust-analyzer failed to discover. The file will not have IDE features available.
pub(crate) fn unlinked_file(
    ctx: &DiagnosticsContext<'_>,
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
) {
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

fn fixes(ctx: &DiagnosticsContext<'_>, file_id: FileId) -> Option<Vec<Assist>> {
    // If there's an existing module that could add `mod` or `pub mod` items to include the unlinked file,
    // suggest that as a fix.

    let source_root = ctx.sema.db.source_root(ctx.sema.db.file_source_root(file_id));
    let our_path = source_root.path_for_file(&file_id)?;
    let parent = our_path.parent()?;
    let (module_name, _) = our_path.name_and_extension()?;
    let (parent, module_name) = match module_name {
        // for mod.rs we need to actually look up one higher
        // and take the parent as our to be module name
        "mod" => {
            let (name, _) = parent.name_and_extension()?;
            (parent.parent()?, name.to_owned())
        }
        _ => (parent, module_name.to_owned()),
    };

    // check crate roots, i.e. main.rs, lib.rs, ...
    'crates: for &krate in &*ctx.sema.db.relevant_crates(file_id) {
        let crate_def_map = ctx.sema.db.crate_def_map(krate);

        let root_module = &crate_def_map[crate_def_map.root()];
        let Some(root_file_id) = root_module.origin.file_id() else { continue };
        let Some(crate_root_path) = source_root.path_for_file(&root_file_id) else { continue };
        let Some(rel) = parent.strip_prefix(&crate_root_path.parent()?) else { continue };

        // try resolving the relative difference of the paths as inline modules
        let mut current = root_module;
        for ele in rel.as_ref().components() {
            let seg = match ele {
                std::path::Component::Normal(seg) => seg.to_str()?,
                std::path::Component::RootDir => continue,
                // shouldn't occur
                _ => continue 'crates,
            };
            match current.children.iter().find(|(name, _)| name.to_smol_str() == seg) {
                Some((_, &child)) => current = &crate_def_map[child],
                None => continue 'crates,
            }
            if !current.origin.is_inline() {
                continue 'crates;
            }
        }

        let InFile { file_id: parent_file_id, value: source } =
            current.definition_source(ctx.sema.db);
        let parent_file_id = parent_file_id.file_id()?;
        return make_fixes(ctx.sema.db, parent_file_id, source, &module_name, file_id);
    }

    // if we aren't adding to a crate root, walk backwards such that we support `#[path = ...]` overrides if possible

    // build all parent paths of the form `../module_name/mod.rs` and `../module_name.rs`
    let paths = iter::successors(Some(parent.clone()), |prev| prev.parent()).filter_map(|path| {
        let parent = path.parent()?;
        let (name, _) = path.name_and_extension()?;
        Some(([parent.join(&format!("{name}.rs"))?, path.join("mod.rs")?], name.to_owned()))
    });
    let mut stack = vec![];
    let &parent_id =
        paths.inspect(|(_, name)| stack.push(name.clone())).find_map(|(paths, _)| {
            paths.into_iter().find_map(|path| source_root.file_for_path(&path))
        })?;
    stack.pop();
    'crates: for &krate in ctx.sema.db.relevant_crates(parent_id).iter() {
        let crate_def_map = ctx.sema.db.crate_def_map(krate);
        let Some((_, module)) =
            crate_def_map.modules()
            .find(|(_, module)| module.origin.file_id() == Some(parent_id) && !module.origin.is_inline())
        else { continue };

        if stack.is_empty() {
            return make_fixes(
                ctx.sema.db,
                parent_id,
                module.definition_source(ctx.sema.db).value,
                &module_name,
                file_id,
            );
        } else {
            // direct parent file is missing,
            // try finding a parent that has an inline tree from here on
            let mut current = module;
            for s in stack.iter().rev() {
                match module.children.iter().find(|(name, _)| name.to_smol_str() == s) {
                    Some((_, child)) => {
                        current = &crate_def_map[*child];
                    }
                    None => continue 'crates,
                }
                if !current.origin.is_inline() {
                    continue 'crates;
                }
            }
            let InFile { file_id: parent_file_id, value: source } =
                current.definition_source(ctx.sema.db);
            let parent_file_id = parent_file_id.file_id()?;
            return make_fixes(ctx.sema.db, parent_file_id, source, &module_name, file_id);
        }
    }

    None
}

fn make_fixes(
    db: &RootDatabase,
    parent_file_id: FileId,
    source: ModuleSource,
    new_mod_name: &str,
    added_file_id: FileId,
) -> Option<Vec<Assist>> {
    fn is_outline_mod(item: &ast::Item) -> bool {
        matches!(item, ast::Item::Module(m) if m.item_list().is_none())
    }

    let mod_decl = format!("mod {new_mod_name};");
    let pub_mod_decl = format!("pub mod {new_mod_name};");

    let mut mod_decl_builder = TextEdit::builder();
    let mut pub_mod_decl_builder = TextEdit::builder();

    let mut items = match &source {
        ModuleSource::SourceFile(it) => it.items(),
        ModuleSource::Module(it) => it.item_list()?.items(),
        ModuleSource::BlockExpr(_) => return None,
    };

    // If there's an existing `mod m;` statement matching the new one, don't emit a fix (it's
    // probably `#[cfg]`d out).
    for item in items.clone() {
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
    match items.clone().skip_while(|item| !is_outline_mod(item)).take_while(is_outline_mod).last() {
        Some(last) => {
            cov_mark::hit!(unlinked_file_append_to_existing_mods);
            let offset = last.syntax().text_range().end();
            let indent = IndentLevel::from_node(last.syntax());
            mod_decl_builder.insert(offset, format!("\n{indent}{mod_decl}"));
            pub_mod_decl_builder.insert(offset, format!("\n{indent}{pub_mod_decl}"));
        }
        None => {
            // Prepend before the first item in the file.
            match items.next() {
                Some(first) => {
                    cov_mark::hit!(unlinked_file_prepend_before_first_item);
                    let offset = first.syntax().text_range().start();
                    let indent = IndentLevel::from_node(first.syntax());
                    mod_decl_builder.insert(offset, format!("{mod_decl}\n\n{indent}"));
                    pub_mod_decl_builder.insert(offset, format!("{pub_mod_decl}\n\n{indent}"));
                }
                None => {
                    // No items in the file, so just append at the end.
                    cov_mark::hit!(unlinked_file_empty_file);
                    let mut indent = IndentLevel::from(0);
                    let offset = match &source {
                        ModuleSource::SourceFile(it) => it.syntax().text_range().end(),
                        ModuleSource::Module(it) => {
                            indent = IndentLevel::from_node(it.syntax()) + 1;
                            it.item_list()?.r_curly_token()?.text_range().start()
                        }
                        ModuleSource::BlockExpr(it) => {
                            it.stmt_list()?.r_curly_token()?.text_range().start()
                        }
                    };
                    mod_decl_builder.insert(offset, format!("{indent}{mod_decl}\n"));
                    pub_mod_decl_builder.insert(offset, format!("{indent}{pub_mod_decl}\n"));
                }
            }
        }
    }

    let trigger_range = db.parse(added_file_id).tree().syntax().text_range();
    Some(vec![
        fix(
            "add_mod_declaration",
            &format!("Insert `{mod_decl}`"),
            SourceChange::from_text_edit(parent_file_id, mod_decl_builder.finish()),
            trigger_range,
        ),
        fix(
            "add_pub_mod_declaration",
            &format!("Insert `{pub_mod_decl}`"),
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

    #[test]
    fn unlinked_file_insert_into_inline_simple() {
        check_fix(
            r#"
//- /main.rs
mod bar;
//- /bar.rs
mod foo {
}
//- /bar/foo/baz.rs
$0
"#,
            r#"
mod foo {
    mod baz;
}
"#,
        );
    }

    #[test]
    fn unlinked_file_insert_into_inline_simple_modrs() {
        check_fix(
            r#"
//- /main.rs
mod bar;
//- /bar.rs
mod baz {
}
//- /bar/baz/foo/mod.rs
$0
"#,
            r#"
mod baz {
    mod foo;
}
"#,
        );
    }

    #[test]
    fn unlinked_file_insert_into_inline_simple_modrs_main() {
        check_fix(
            r#"
//- /main.rs
mod bar {
}
//- /bar/foo/mod.rs
$0
"#,
            r#"
mod bar {
    mod foo;
}
"#,
        );
    }
}
