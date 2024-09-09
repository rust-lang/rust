//! Run all tests in a project, similar to `cargo test`, but using the mir interpreter.

use hir::{Crate, Module};
use hir_ty::db::HirDatabase;
use ide_db::{base_db::SourceRootDatabase, LineIndexDatabase};
use profile::StopWatch;
use project_model::{CargoConfig, RustLibSource};
use syntax::TextRange;

use load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice};

use crate::cli::{flags, full_name_of_item, Result};

impl flags::RunTests {
    pub fn run(self) -> Result<()> {
        let cargo_config = CargoConfig {
            sysroot: Some(RustLibSource::Discover),
            all_targets: true,
            set_test: true,
            ..Default::default()
        };
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
        };
        let (ref db, _vfs, _proc_macro) =
            load_workspace_at(&self.path, &cargo_config, &load_cargo_config, &|_| {})?;

        let tests = all_modules(db)
            .into_iter()
            .flat_map(|x| x.declarations(db))
            .filter_map(|x| match x {
                hir::ModuleDef::Function(f) => Some(f),
                _ => None,
            })
            .filter(|x| x.is_test(db));
        let span_formatter = |file_id, text_range: TextRange| {
            let line_col = match db.line_index(file_id).try_line_col(text_range.start()) {
                None => " (unknown line col)".to_owned(),
                Some(x) => format!("#{}:{}", x.line + 1, x.col),
            };
            let path = &db
                .source_root(db.file_source_root(file_id))
                .path_for_file(&file_id)
                .map(|x| x.to_string());
            let path = path.as_deref().unwrap_or("<unknown file>");
            format!("file://{path}{line_col}")
        };
        let mut pass_count = 0;
        let mut ignore_count = 0;
        let mut fail_count = 0;
        let mut sw_all = StopWatch::start();
        for test in tests {
            let full_name = full_name_of_item(db, test.module(db), test.name(db));
            println!("test {full_name}");
            if test.is_ignore(db) {
                println!("ignored");
                ignore_count += 1;
                continue;
            }
            let mut sw_one = StopWatch::start();
            let result = test.eval(db, span_formatter);
            if result.trim() == "pass" {
                pass_count += 1;
            } else {
                fail_count += 1;
            }
            println!("{result}");
            eprintln!("{:<20} {}", format!("test {}", full_name), sw_one.elapsed());
        }
        println!("{pass_count} passed, {fail_count} failed, {ignore_count} ignored");
        eprintln!("{:<20} {}", "All tests", sw_all.elapsed());
        Ok(())
    }
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> = Crate::all(db)
        .into_iter()
        .filter(|x| x.origin(db).is_local())
        .map(|krate| krate.root_module())
        .collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}
