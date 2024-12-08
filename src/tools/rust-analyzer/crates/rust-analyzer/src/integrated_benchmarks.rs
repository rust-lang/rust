//! Fully integrated benchmarks for rust-analyzer, which load real cargo
//! projects.
//!
//! The benchmark here is used to debug specific performance regressions. If you
//! notice that, eg, completion is slow in some specific case, you can  modify
//! code here exercise this specific completion, and thus have a fast
//! edit/compile/test cycle.
//!
//! Note that "rust-analyzer: Run" action does not allow running a single test
//! in release mode in VS Code. There's however "rust-analyzer: Copy Run Command Line"
//! which you can use to paste the command in terminal and add `--release` manually.

use hir::ChangeWithProcMacros;
use ide::{
    AnalysisHost, CallableSnippets, CompletionConfig, CompletionFieldsToResolve, DiagnosticsConfig,
    FilePosition, TextSize,
};
use ide_db::{
    imports::insert_use::{ImportGranularity, InsertUseConfig},
    SnippetCap,
};
use project_model::CargoConfig;
use test_utils::project_root;
use vfs::{AbsPathBuf, VfsPath};

use load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice};

#[test]
fn integrated_highlighting_benchmark() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }

    // Load rust-analyzer itself.
    let workspace_to_load = project_root();
    let file = "./crates/rust-analyzer/src/config.rs";

    let cargo_config = CargoConfig {
        sysroot: Some(project_model::RustLibSource::Discover),
        all_targets: true,
        set_test: true,
        ..CargoConfig::default()
    };
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: false,
    };

    let (db, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(
            workspace_to_load.as_std_path(),
            &cargo_config,
            &load_cargo_config,
            &|_| {},
        )
        .unwrap()
    };
    let mut host = AnalysisHost::with_database(db);

    let file_id = {
        let file = workspace_to_load.join(file);
        let path = VfsPath::from(AbsPathBuf::assert(file));
        vfs.file_id(&path).unwrap_or_else(|| panic!("can't find virtual file for {path}"))
    };

    {
        let _it = stdx::timeit("initial");
        let analysis = host.analysis();
        analysis.highlight_as_html(file_id, false).unwrap();
    }

    {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        text = text.replace(
            "self.data.cargo_buildScripts_rebuildOnSave",
            "self. data. cargo_buildScripts_rebuildOnSave",
        );
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
    }

    let _g = crate::tracing::hprof::init("*>10");

    {
        let _it = stdx::timeit("after change");
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        analysis.highlight_as_html(file_id, false).unwrap();
    }
}

#[test]
fn integrated_completion_benchmark() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }

    // Load rust-analyzer itself.
    let workspace_to_load = project_root();
    let file = "./crates/hir/src/lib.rs";

    let cargo_config = CargoConfig {
        sysroot: Some(project_model::RustLibSource::Discover),
        all_targets: true,
        set_test: true,
        ..CargoConfig::default()
    };
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: true,
    };

    let (db, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(
            workspace_to_load.as_std_path(),
            &cargo_config,
            &load_cargo_config,
            &|_| {},
        )
        .unwrap()
    };
    let mut host = AnalysisHost::with_database(db);

    let file_id = {
        let file = workspace_to_load.join(file);
        let path = VfsPath::from(AbsPathBuf::assert(file));
        vfs.file_id(&path).unwrap_or_else(|| panic!("can't find virtual file for {path}"))
    };

    // kick off parsing and index population

    let completion_offset = {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        let completion_offset =
            patch(&mut text, "db.struct_data(self.id)", "sel;\ndb.struct_data(self.id)")
                + "sel".len();
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
        completion_offset
    };

    {
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_the_fly: true,
            enable_self_on_the_fly: true,
            enable_private_editable: true,
            enable_term_search: true,
            term_search_fuel: 200,
            full_function_signatures: false,
            callable: Some(CallableSnippets::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: hir::PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            prefer_no_std: false,
            prefer_prelude: true,
            prefer_absolute: false,
            snippets: Vec::new(),
            limit: None,
            add_semicolon_to_unit: true,
            fields_to_resolve: CompletionFieldsToResolve::empty(),
        };
        let position =
            FilePosition { file_id, offset: TextSize::try_from(completion_offset).unwrap() };
        analysis.completions(&config, position, None).unwrap();
    }

    let _g = crate::tracing::hprof::init("*>10");

    let completion_offset = {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        let completion_offset =
            patch(&mut text, "sel;\ndb.struct_data(self.id)", ";sel;\ndb.struct_data(self.id)")
                + ";sel".len();
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
        completion_offset
    };

    {
        let _p = tracing::info_span!("unqualified path completion").entered();
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_the_fly: true,
            enable_self_on_the_fly: true,
            enable_private_editable: true,
            enable_term_search: true,
            term_search_fuel: 200,
            full_function_signatures: false,
            callable: Some(CallableSnippets::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: hir::PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            prefer_no_std: false,
            prefer_prelude: true,
            prefer_absolute: false,
            snippets: Vec::new(),
            limit: None,
            add_semicolon_to_unit: true,
            fields_to_resolve: CompletionFieldsToResolve::empty(),
        };
        let position =
            FilePosition { file_id, offset: TextSize::try_from(completion_offset).unwrap() };
        analysis.completions(&config, position, None).unwrap();
    }

    let completion_offset = {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        let completion_offset =
            patch(&mut text, "sel;\ndb.struct_data(self.id)", "self.;\ndb.struct_data(self.id)")
                + "self.".len();
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
        completion_offset
    };

    {
        let _p = tracing::info_span!("dot completion").entered();
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_the_fly: true,
            enable_self_on_the_fly: true,
            enable_private_editable: true,
            enable_term_search: true,
            term_search_fuel: 200,
            full_function_signatures: false,
            callable: Some(CallableSnippets::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: hir::PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            prefer_no_std: false,
            prefer_prelude: true,
            prefer_absolute: false,
            snippets: Vec::new(),
            limit: None,
            add_semicolon_to_unit: true,
            fields_to_resolve: CompletionFieldsToResolve::empty(),
        };
        let position =
            FilePosition { file_id, offset: TextSize::try_from(completion_offset).unwrap() };
        analysis.completions(&config, position, None).unwrap();
    }
}

#[test]
fn integrated_diagnostics_benchmark() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }

    // Load rust-analyzer itself.
    let workspace_to_load = project_root();
    let file = "./crates/hir/src/lib.rs";

    let cargo_config = CargoConfig {
        sysroot: Some(project_model::RustLibSource::Discover),
        all_targets: true,
        set_test: true,
        ..CargoConfig::default()
    };
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: true,
    };

    let (db, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(
            workspace_to_load.as_std_path(),
            &cargo_config,
            &load_cargo_config,
            &|_| {},
        )
        .unwrap()
    };
    let mut host = AnalysisHost::with_database(db);

    let file_id = {
        let file = workspace_to_load.join(file);
        let path = VfsPath::from(AbsPathBuf::assert(file));
        vfs.file_id(&path).unwrap_or_else(|| panic!("can't find virtual file for {path}"))
    };

    let diagnostics_config = DiagnosticsConfig {
        enabled: false,
        proc_macros_enabled: true,
        proc_attr_macros_enabled: true,
        disable_experimental: true,
        disabled: Default::default(),
        expr_fill_default: Default::default(),
        style_lints: false,
        snippet_cap: SnippetCap::new(true),
        insert_use: InsertUseConfig {
            granularity: ImportGranularity::Crate,
            enforce_granularity: false,
            prefix_kind: hir::PrefixKind::ByCrate,
            group: true,
            skip_glob_imports: true,
        },
        prefer_no_std: false,
        prefer_prelude: false,
        prefer_absolute: false,
        term_search_fuel: 400,
        term_search_borrowck: true,
    };
    host.analysis()
        .full_diagnostics(&diagnostics_config, ide::AssistResolveStrategy::None, file_id)
        .unwrap();

    let _g = crate::tracing::hprof::init("*");

    {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        patch(&mut text, "db.struct_data(self.id)", "();\ndb.struct_data(self.id)");
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
    };

    {
        let _p = tracing::info_span!("diagnostics").entered();
        let _span = profile::cpu_span();
        host.analysis()
            .full_diagnostics(&diagnostics_config, ide::AssistResolveStrategy::None, file_id)
            .unwrap();
    }
}

fn patch(what: &mut String, from: &str, to: &str) -> usize {
    let idx = what.find(from).unwrap();
    *what = what.replacen(from, to, 1);
    idx
}
