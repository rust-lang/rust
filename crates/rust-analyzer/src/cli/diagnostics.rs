//! Analyze all modules in a project for diagnostics. Exits with a non-zero
//! status code if any errors are found.

use project_model::{CargoConfig, RustLibSource};
use rustc_hash::FxHashSet;

use hir::{db::HirDatabase, Crate, Module};
use ide::{AssistResolveStrategy, DiagnosticsConfig, Severity};
use ide_db::base_db::SourceDatabaseExt;
use load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice};

use crate::cli::flags;

impl flags::Diagnostics {
    pub fn run(self) -> anyhow::Result<()> {
        let mut cargo_config = CargoConfig::default();
        cargo_config.sysroot = Some(RustLibSource::Discover);
        let with_proc_macro_server = if let Some(p) = &self.proc_macro_srv {
            let path = vfs::AbsPathBuf::assert(std::env::current_dir()?.join(&p));
            ProcMacroServerChoice::Explicit(path)
        } else {
            ProcMacroServerChoice::Sysroot
        };
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: !self.disable_build_scripts,
            with_proc_macro_server,
            prefill_caches: false,
        };
        let (host, _vfs, _proc_macro) =
            load_workspace_at(&self.path, &cargo_config, &load_cargo_config, &|_| {})?;
        let db = host.raw_database();
        let analysis = host.analysis();

        let mut found_error = false;
        let mut visited_files = FxHashSet::default();

        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source_file_id(db).original_file(db);
            let source_root = db.file_source_root(file_id);
            let source_root = db.source_root(source_root);
            !source_root.is_library
        });

        for module in work {
            let file_id = module.definition_source_file_id(db).original_file(db);
            if !visited_files.contains(&file_id) {
                let crate_name =
                    module.krate().display_name(db).as_deref().unwrap_or("unknown").to_string();
                println!("processing crate: {crate_name}, module: {}", _vfs.file_path(file_id));
                for diagnostic in analysis
                    .diagnostics(
                        &DiagnosticsConfig::test_sample(),
                        AssistResolveStrategy::None,
                        file_id,
                    )
                    .unwrap()
                {
                    if matches!(diagnostic.severity, Severity::Error) {
                        found_error = true;
                    }

                    println!("{diagnostic:?}");
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
        Crate::all(db).into_iter().map(|krate| krate.root_module(db)).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}
