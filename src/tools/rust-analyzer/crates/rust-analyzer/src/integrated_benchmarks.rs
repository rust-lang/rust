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

use ide::{CallableSnippets, Change, CompletionConfig, FilePosition, TextSize};
use ide_db::{
    imports::insert_use::{ImportGranularity, InsertUseConfig},
    SnippetCap,
};
use project_model::CargoConfig;
use test_utils::project_root;
use triomphe::Arc;
use vfs::{AbsPathBuf, VfsPath};

use crate::cli::load_cargo::{load_workspace_at, LoadCargoConfig, ProcMacroServerChoice};

#[test]
fn integrated_highlighting_benchmark() {
    if std::env::var("RUN_SLOW_BENCHES").is_err() {
        return;
    }

    // Load rust-analyzer itself.
    let workspace_to_load = project_root();
    let file = "./crates/ide-db/src/apply_change.rs";

    let cargo_config = CargoConfig::default();
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::None,
        prefill_caches: false,
    };

    let (mut host, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(&workspace_to_load, &cargo_config, &load_cargo_config, &|_| {}).unwrap()
    };

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

    profile::init_from("*>100");
    // let _s = profile::heartbeat_span();

    {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        text.push_str("\npub fn _dummy() {}\n");
        let mut change = Change::new();
        change.change_file(file_id, Some(Arc::from(text)));
        host.apply_change(change);
    }

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

    let cargo_config = CargoConfig::default();
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::None,
        prefill_caches: true,
    };

    let (mut host, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(&workspace_to_load, &cargo_config, &load_cargo_config, &|_| {}).unwrap()
    };

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

    profile::init_from("*>5");
    // let _s = profile::heartbeat_span();

    let completion_offset = {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        let completion_offset =
            patch(&mut text, "db.struct_data(self.id)", "sel;\ndb.struct_data(self.id)")
                + "sel".len();
        let mut change = Change::new();
        change.change_file(file_id, Some(Arc::from(text)));
        host.apply_change(change);
        completion_offset
    };

    {
        let _p = profile::span("unqualified path completion");
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_the_fly: true,
            enable_self_on_the_fly: true,
            enable_private_editable: true,
            callable: Some(CallableSnippets::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: hir::PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            snippets: Vec::new(),
            prefer_no_std: false,
            limit: None,
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
        let mut change = Change::new();
        change.change_file(file_id, Some(Arc::from(text)));
        host.apply_change(change);
        completion_offset
    };

    {
        let _p = profile::span("dot completion");
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_the_fly: true,
            enable_self_on_the_fly: true,
            enable_private_editable: true,
            callable: Some(CallableSnippets::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: hir::PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            snippets: Vec::new(),
            prefer_no_std: false,
            limit: None,
        };
        let position =
            FilePosition { file_id, offset: TextSize::try_from(completion_offset).unwrap() };
        analysis.completions(&config, position, None).unwrap();
    }
}

fn patch(what: &mut String, from: &str, to: &str) -> usize {
    let idx = what.find(from).unwrap();
    *what = what.replacen(from, to, 1);
    idx
}
