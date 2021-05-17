//! Diagnostic emitted for files that aren't part of any crate.

use hir::{
    db::DefDatabase,
    diagnostics::{Diagnostic, DiagnosticCode},
    InFile,
};
use ide_assists::AssistResolveStrategy;
use ide_db::{
    base_db::{FileId, FileLoader, SourceDatabase, SourceDatabaseExt},
    source_change::SourceChange,
    RootDatabase,
};
use syntax::{
    ast::{self, ModuleItemOwner, NameOwner},
    AstNode, SyntaxNodePtr,
};
use text_edit::TextEdit;

use crate::{
    diagnostics::{fix, fixes::DiagnosticWithFixes},
    Assist,
};

// Diagnostic: unlinked-file
//
// This diagnostic is shown for files that are not included in any crate, or files that are part of
// crates rust-analyzer failed to discover. The file will not have IDE features available.
#[derive(Debug)]
pub(crate) struct UnlinkedFile {
    pub(crate) file_id: FileId,
    pub(crate) node: SyntaxNodePtr,
}

impl Diagnostic for UnlinkedFile {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unlinked-file")
    }

    fn message(&self) -> String {
        "file not included in module tree".to_string()
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file_id.into(), self.node.clone())
    }

    fn as_any(&self) -> &(dyn std::any::Any + Send + 'static) {
        self
    }
}

impl DiagnosticWithFixes for UnlinkedFile {
    fn fixes(
        &self,
        sema: &hir::Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        // If there's an existing module that could add `mod` or `pub mod` items to include the unlinked file,
        // suggest that as a fix.

        let source_root = sema.db.source_root(sema.db.file_source_root(self.file_id));
        let our_path = source_root.path_for_file(&self.file_id)?;
        let module_name = our_path.name_and_extension()?.0;

        // Candidates to look for:
        // - `mod.rs` in the same folder
        //   - we also check `main.rs` and `lib.rs`
        // - `$dir.rs` in the parent folder, where `$dir` is the directory containing `self.file_id`
        let parent = our_path.parent()?;
        let mut paths =
            vec![parent.join("mod.rs")?, parent.join("lib.rs")?, parent.join("main.rs")?];

        // `submod/bla.rs` -> `submod.rs`
        if let Some(newmod) = (|| {
            let name = parent.name_and_extension()?.0;
            parent.parent()?.join(&format!("{}.rs", name))
        })() {
            paths.push(newmod);
        }

        for path in &paths {
            if let Some(parent_id) = source_root.file_for_path(path) {
                for krate in sema.db.relevant_crates(*parent_id).iter() {
                    let crate_def_map = sema.db.crate_def_map(*krate);
                    for (_, module) in crate_def_map.modules() {
                        if module.origin.is_inline() {
                            // We don't handle inline `mod parent {}`s, they use different paths.
                            continue;
                        }

                        if module.origin.file_id() == Some(*parent_id) {
                            return make_fixes(sema.db, *parent_id, module_name, self.file_id);
                        }
                    }
                }
            }
        }

        None
    }
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
