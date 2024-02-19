use std::path::PathBuf;

use project_model::{CargoWorkspace, ProjectWorkspace, Sysroot, WorkspaceBuildScripts};
use rust_analyzer::ws_to_crate_graph;
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;
use vfs::{AbsPathBuf, FileId};

fn load_cargo_with_fake_sysroot(file: &str) -> ProjectWorkspace {
    let meta = get_test_json_file(file);
    let cargo_workspace = CargoWorkspace::new(meta);
    ProjectWorkspace::Cargo {
        cargo: cargo_workspace,
        build_scripts: WorkspaceBuildScripts::default(),
        sysroot: Ok(get_fake_sysroot()),
        rustc: Err(None),
        rustc_cfg: Vec::new(),
        cfg_overrides: Default::default(),
        toolchain: None,
        target_layout: Err("target_data_layout not loaded".into()),
        cargo_config_extra_env: Default::default(),
    }
}

fn get_test_json_file<T: DeserializeOwned>(file: &str) -> T {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let file = base.join("tests/test_data").join(file);
    let data = std::fs::read_to_string(file).unwrap();
    let mut json = data.parse::<serde_json::Value>().unwrap();
    fixup_paths(&mut json);
    return serde_json::from_value(json).unwrap();

    fn fixup_paths(val: &mut serde_json::Value) {
        match val {
            serde_json::Value::String(s) => replace_root(s, true),
            serde_json::Value::Array(vals) => vals.iter_mut().for_each(fixup_paths),
            serde_json::Value::Object(kvals) => kvals.values_mut().for_each(fixup_paths),
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            }
        }
    }
}

fn replace_root(s: &mut String, direction: bool) {
    if direction {
        let root = if cfg!(windows) { r#"C:\\ROOT\"# } else { "/ROOT/" };
        *s = s.replace("$ROOT$", root)
    } else {
        let root = if cfg!(windows) { r#"C:\\\\ROOT\\"# } else { "/ROOT/" };
        *s = s.replace(root, "$ROOT$")
    }
}

fn get_fake_sysroot_path() -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base.join("../project-model/test_data/fake-sysroot")
}

fn get_fake_sysroot() -> Sysroot {
    let sysroot_path = get_fake_sysroot_path();
    // there's no `libexec/` directory with a `proc-macro-srv` binary in that
    // fake sysroot, so we give them both the same path:
    let sysroot_dir = AbsPathBuf::assert(sysroot_path);
    let sysroot_src_dir = sysroot_dir.clone();
    Sysroot::load(sysroot_dir, Some(Ok(sysroot_src_dir)), false)
}

#[test]
fn test_deduplicate_origin_dev() {
    let path_map = &mut FxHashMap::default();
    let ws = load_cargo_with_fake_sysroot("deduplication_crate_graph_A.json");
    let ws2 = load_cargo_with_fake_sysroot("deduplication_crate_graph_B.json");

    let (crate_graph, ..) = ws_to_crate_graph(&[ws, ws2], &Default::default(), |path| {
        let len = path_map.len();
        Some(*path_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
    });

    let mut crates_named_p2 = vec![];
    for id in crate_graph.iter() {
        let krate = &crate_graph[id];
        if let Some(name) = krate.display_name.as_ref() {
            if name.to_string() == "p2" {
                crates_named_p2.push(krate);
            }
        }
    }

    assert!(crates_named_p2.len() == 1);
    let p2 = crates_named_p2[0];
    assert!(p2.origin.is_local());
}

#[test]
fn test_deduplicate_origin_dev_rev() {
    let path_map = &mut FxHashMap::default();
    let ws = load_cargo_with_fake_sysroot("deduplication_crate_graph_B.json");
    let ws2 = load_cargo_with_fake_sysroot("deduplication_crate_graph_A.json");

    let (crate_graph, ..) = ws_to_crate_graph(&[ws, ws2], &Default::default(), |path| {
        let len = path_map.len();
        Some(*path_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
    });

    let mut crates_named_p2 = vec![];
    for id in crate_graph.iter() {
        let krate = &crate_graph[id];
        if let Some(name) = krate.display_name.as_ref() {
            if name.to_string() == "p2" {
                crates_named_p2.push(krate);
            }
        }
    }

    assert!(crates_named_p2.len() == 1);
    let p2 = crates_named_p2[0];
    assert!(p2.origin.is_local());
}
