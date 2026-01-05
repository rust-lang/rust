//! Reports references in code that the IDE layer cannot resolve.
use hir::{AnyDiagnostic, Crate, Module, Semantics, db::HirDatabase, sym};
use ide::{AnalysisHost, RootDatabase, TextRange};
use ide_db::{FxHashSet, LineIndexDatabase as _, base_db::SourceDatabase, defs::NameRefClass};
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use parser::SyntaxKind;
use syntax::{AstNode, WalkEvent, ast};
use vfs::FileId;

use crate::cli::flags;

impl flags::UnresolvedReferences {
    pub fn run(self) -> anyhow::Result<()> {
        const STACK_SIZE: usize = 1024 * 1024 * 8;

        let handle = stdx::thread::Builder::new(
            stdx::thread::ThreadIntent::LatencySensitive,
            "BIG_STACK_THREAD",
        )
        .stack_size(STACK_SIZE)
        .spawn(|| self.run_())
        .unwrap();

        handle.join()
    }

    fn run_(self) -> anyhow::Result<()> {
        let root =
            vfs::AbsPathBuf::assert_utf8(std::env::current_dir()?.join(&self.path)).normalize();
        let config = crate::config::Config::new(
            root,
            lsp_types::ClientCapabilities::default(),
            vec![],
            None,
        );
        let cargo_config = config.cargo(None);
        let with_proc_macro_server = if let Some(p) = &self.proc_macro_srv {
            let path = vfs::AbsPathBuf::assert_utf8(std::env::current_dir()?.join(p));
            ProcMacroServerChoice::Explicit(path)
        } else {
            ProcMacroServerChoice::Sysroot
        };
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: !self.disable_build_scripts,
            with_proc_macro_server,
            prefill_caches: false,
            proc_macro_processes: config.proc_macro_num_processes(),
        };
        let (db, vfs, _proc_macro) =
            load_workspace_at(&self.path, &cargo_config, &load_cargo_config, &|_| {})?;
        let host = AnalysisHost::with_database(db);
        let db = host.raw_database();
        let sema = Semantics::new(db);

        let mut visited_files = FxHashSet::default();

        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source_file_id(db).original_file(db);
            let source_root = db.file_source_root(file_id.file_id(db)).source_root_id(db);
            let source_root = db.source_root(source_root).source_root(db);
            !source_root.is_library
        });

        for module in work {
            let file_id = module.definition_source_file_id(db).original_file(db);
            let file_id = file_id.file_id(db);
            if !visited_files.contains(&file_id) {
                let crate_name = module
                    .krate(db)
                    .display_name(db)
                    .as_deref()
                    .unwrap_or(&sym::unknown)
                    .to_owned();
                let file_path = vfs.file_path(file_id);
                eprintln!("processing crate: {crate_name}, module: {file_path}",);

                let line_index = db.line_index(file_id);
                let file_text = db.file_text(file_id);

                for range in find_unresolved_references(db, &sema, file_id, &module) {
                    let line_col = line_index.line_col(range.start());
                    let line = line_col.line + 1;
                    let col = line_col.col + 1;
                    let text = &file_text.text(db)[range];
                    println!("{file_path}:{line}:{col}: {text}");
                }

                visited_files.insert(file_id);
            }
        }

        eprintln!();
        eprintln!("scan complete");

        Ok(())
    }
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module(db)).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}

fn find_unresolved_references(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    file_id: FileId,
    module: &Module,
) -> Vec<TextRange> {
    let mut unresolved_references = all_unresolved_references(sema, file_id);

    // remove unresolved references which are within inactive code
    let mut diagnostics = Vec::new();
    module.diagnostics(db, &mut diagnostics, false);
    for diagnostic in diagnostics {
        let AnyDiagnostic::InactiveCode(inactive_code) = diagnostic else {
            continue;
        };

        let node = inactive_code.node;
        let range = node.map(|it| it.text_range()).original_node_file_range_rooted(db);

        if range.file_id.file_id(db) != file_id {
            continue;
        }

        unresolved_references.retain(|r| !range.range.contains_range(*r));
    }

    unresolved_references
}

fn all_unresolved_references(
    sema: &Semantics<'_, RootDatabase>,
    file_id: FileId,
) -> Vec<TextRange> {
    let file_id = sema.attach_first_edition(file_id);
    let file = sema.parse(file_id);
    let root = file.syntax();

    let mut unresolved_references = Vec::new();
    for event in root.preorder() {
        let WalkEvent::Enter(syntax) = event else {
            continue;
        };
        let Some(name_ref) = ast::NameRef::cast(syntax) else {
            continue;
        };
        let Some(descended_name_ref) = name_ref.syntax().first_token().and_then(|tok| {
            sema.descend_into_macros_single_exact(tok).parent().and_then(ast::NameRef::cast)
        }) else {
            continue;
        };

        // if we can classify the name_ref, it's not unresolved
        if NameRefClass::classify(sema, &descended_name_ref).is_some() {
            continue;
        }

        // if we couldn't classify it, but it's in an attr, ignore it. See #10935
        if descended_name_ref.syntax().ancestors().any(|it| it.kind() == SyntaxKind::ATTR) {
            continue;
        }

        // otherwise, it's unresolved
        unresolved_references.push(name_ref.syntax().text_range());
    }
    unresolved_references
}
