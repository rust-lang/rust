//! Fully integrated benchmarks for rust-analyzer, which load real cargo
//! projects.
//!
//! The benchmark here is used to debug specific performance regressions. If you
//! notice that, eg, completion is slow in some specific case, you can  modify
//! code here exercise this specific completion, and thus have a fast
//! edit/compile/test cycle.
//!
//! Note that "Rust Analyzer: Run" action does not allow running a single test
//! in release mode in VS Code. There's however "Rust Analyzer: Copy Run Command Line"
//! which you can use to paste the command in terminal and add `--release` manually.

use std::sync::Arc;

use ide::Change;
use test_utils::project_root;
use vfs::{AbsPathBuf, VfsPath};

use crate::cli::load_cargo::{load_workspace_at, LoadCargoConfig};

#[test]
fn benchmark_integrated_highlighting() {
    // Don't run slow benchmark by default
    if true {
        return;
    }

    // Load rust-analyzer itself.
    let workspace_to_load = project_root();
    let file = "./crates/ide_db/src/apply_change.rs";

    let cargo_config = Default::default();
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        wrap_rustc: false,
        with_proc_macro: false,
    };

    let (mut host, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(&workspace_to_load, &cargo_config, &load_cargo_config, &|_| {}).unwrap()
    };

    let file_id = {
        let file = workspace_to_load.join(file);
        let path = VfsPath::from(AbsPathBuf::assert(file));
        vfs.file_id(&path).unwrap_or_else(|| panic!("can't find virtual file for {}", path))
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
        change.change_file(file_id, Some(Arc::new(text)));
        host.apply_change(change);
    }

    {
        let _it = stdx::timeit("after change");
        let _span = profile::cpu_span();
        let analysis = host.analysis();
        analysis.highlight_as_html(file_id, false).unwrap();
    }
}
