//! Analyze all modules in a project for diagnostics. Exits with a non-zero
//! status code if any errors are found.

use project_model::{CargoConfig, RustLibSource};
use rustc_hash::FxHashSet;

use hir::{db::HirDatabase, Crate, HirFileIdExt, Module};
use ide::{AnalysisHost, AssistResolveStrategy, Diagnostic, DiagnosticsConfig, Severity};
use ide_db::{base_db::SourceRootDatabase, LineIndexDatabase};
use load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice};

use crate::cli::flags;

impl flags::Diagnostics {
    pub fn run(self) -> anyhow::Result<()> {
        const STACK_SIZE: usize = 1024 * 1024 * 8;

        let handle = stdx::thread::Builder::new(stdx::thread::ThreadIntent::LatencySensitive)
            .name("BIG_STACK_THREAD".into())
            .stack_size(STACK_SIZE)
            .spawn(|| self.run_())
            .unwrap();

        handle.join()
    }
    fn run_(self) -> anyhow::Result<()> {
        let cargo_config = CargoConfig {
            sysroot: Some(RustLibSource::Discover),
            all_targets: true,
            ..Default::default()
        };
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
        };
        let (db, _vfs, _proc_macro) =
            load_workspace_at(&self.path, &cargo_config, &load_cargo_config, &|_| {})?;
        let host = AnalysisHost::with_database(db);
        let db = host.raw_database();
        let analysis = host.analysis();

        let mut found_error = false;
        let mut visited_files = FxHashSet::default();

        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source_file_id(db).original_file(db);
            let source_root = db.file_source_root(file_id.into());
            let source_root = db.source_root(source_root);
            !source_root.is_library
        });

        for module in work {
            let file_id = module.definition_source_file_id(db).original_file(db);
            if !visited_files.contains(&file_id) {
                let crate_name =
                    module.krate().display_name(db).as_deref().unwrap_or("unknown").to_owned();
                println!(
                    "processing crate: {crate_name}, module: {}",
                    _vfs.file_path(file_id.into())
                );
                for diagnostic in analysis
                    .full_diagnostics(
                        &DiagnosticsConfig::test_sample(),
                        AssistResolveStrategy::None,
                        file_id.into(),
                    )
                    .unwrap()
                {
                    if matches!(diagnostic.severity, Severity::Error) {
                        found_error = true;
                    }

                    let Diagnostic { code, message, range, severity, .. } = diagnostic;
                    let line_index = db.line_index(range.file_id);
                    let start = line_index.line_col(range.range.start());
                    let end = line_index.line_col(range.range.end());
                    println!("{severity:?} {code:?} from {start:?} to {end:?}: {message}");
                }

                visited_files.insert(file_id);
            }
        }

        println!();
        println!("diagnostic scan complete");

        if found_error {
            println!();
            anyhow::bail!("diagnostic error detected")
        }

        Ok(())
    }
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module()).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}
