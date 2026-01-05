//! Analyze all modules in a project for diagnostics. Exits with a non-zero
//! status code if any errors are found.

use project_model::{CargoConfig, RustLibSource};
use rustc_hash::FxHashSet;

use hir::{Crate, Module, db::HirDatabase, sym};
use ide::{AnalysisHost, AssistResolveStrategy, Diagnostic, DiagnosticsConfig, Severity};
use ide_db::{LineIndexDatabase, base_db::SourceDatabase};
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};

use crate::cli::{flags, progress_report::ProgressReport};

impl flags::Diagnostics {
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
            proc_macro_processes: 1,
        };
        let (db, _vfs, _proc_macro) =
            load_workspace_at(&self.path, &cargo_config, &load_cargo_config, &|_| {})?;
        let host = AnalysisHost::with_database(db);
        let db = host.raw_database();
        let analysis = host.analysis();

        let mut found_error = false;
        let mut visited_files = FxHashSet::default();
        let min_severity = self.severity.unwrap_or(flags::Severity::Weak);

        let work = all_modules(db)
            .into_iter()
            .filter(|module| {
                let file_id = module.definition_source_file_id(db).original_file(db);
                let source_root = db.file_source_root(file_id.file_id(db)).source_root_id(db);
                let source_root = db.source_root(source_root).source_root(db);
                !source_root.is_library
            })
            .collect::<Vec<_>>();

        let mut bar = ProgressReport::new(work.len());
        for module in work {
            let file_id = module.definition_source_file_id(db).original_file(db);
            if !visited_files.contains(&file_id) {
                let message = format!("processing {}", _vfs.file_path(file_id.file_id(db)));
                bar.set_message(move || message.clone());
                let crate_name = module
                    .krate(db)
                    .display_name(db)
                    .as_deref()
                    .unwrap_or(&sym::unknown)
                    .to_owned();
                for diagnostic in analysis
                    .full_diagnostics(
                        &DiagnosticsConfig::test_sample(),
                        AssistResolveStrategy::None,
                        file_id.file_id(db),
                    )
                    .unwrap()
                {
                    let severity = match diagnostic.severity {
                        Severity::Error => flags::Severity::Error,
                        Severity::Warning => flags::Severity::Warning,
                        Severity::WeakWarning => flags::Severity::Weak,
                        Severity::Allow => continue,
                    };
                    if severity < min_severity {
                        continue;
                    }

                    if matches!(diagnostic.severity, Severity::Error) {
                        found_error = true;
                    }

                    let Diagnostic { code, message, range, severity, .. } = diagnostic;
                    let line_index = db.line_index(range.file_id);
                    let start = line_index.line_col(range.range.start());
                    let end = line_index.line_col(range.range.end());
                    bar.println(format!(
                        "at crate {crate_name}, file {}: {severity:?} {code:?} from {start:?} to {end:?}: {message}",
                        _vfs.file_path(file_id.file_id(db))
                    ));
                }

                visited_files.insert(file_id);
            }
            bar.inc(1);
        }
        bar.finish_and_clear();

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
