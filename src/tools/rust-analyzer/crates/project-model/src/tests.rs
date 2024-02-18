use std::{
    ops::Deref,
    path::{Path, PathBuf},
};

use base_db::{CrateGraph, FileId, ProcMacroPaths};
use cfg::{CfgAtom, CfgDiff};
use expect_test::{expect_file, ExpectFile};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;

use crate::{
    CargoWorkspace, CfgOverrides, ProjectJson, ProjectJsonData, ProjectWorkspace, Sysroot,
    WorkspaceBuildScripts,
};

fn load_cargo(file: &str) -> (CrateGraph, ProcMacroPaths) {
    load_cargo_with_overrides(file, CfgOverrides::default())
}

fn load_cargo_with_overrides(
    file: &str,
    cfg_overrides: CfgOverrides,
) -> (CrateGraph, ProcMacroPaths) {
    let meta = get_test_json_file(file);
    let cargo_workspace = CargoWorkspace::new(meta);
    let project_workspace = ProjectWorkspace::Cargo {
        cargo: cargo_workspace,
        build_scripts: WorkspaceBuildScripts::default(),
        sysroot: Err(None),
        rustc: Err(None),
        rustc_cfg: Vec::new(),
        cfg_overrides,
        toolchain: None,
        target_layout: Err("target_data_layout not loaded".into()),
    };
    to_crate_graph(project_workspace)
}

fn load_cargo_with_fake_sysroot(
    file_map: &mut FxHashMap<AbsPathBuf, FileId>,
    file: &str,
) -> (CrateGraph, ProcMacroPaths) {
    let meta = get_test_json_file(file);
    let cargo_workspace = CargoWorkspace::new(meta);
    let project_workspace = ProjectWorkspace::Cargo {
        cargo: cargo_workspace,
        build_scripts: WorkspaceBuildScripts::default(),
        sysroot: Ok(get_fake_sysroot()),
        rustc: Err(None),
        rustc_cfg: Vec::new(),
        cfg_overrides: Default::default(),
        toolchain: None,
        target_layout: Err("target_data_layout not loaded".into()),
    };
    project_workspace.to_crate_graph(
        &mut {
            |path| {
                let len = file_map.len();
                Some(*file_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
            }
        },
        &Default::default(),
    )
}

fn load_rust_project(file: &str) -> (CrateGraph, ProcMacroPaths) {
    let data = get_test_json_file(file);
    let project = rooted_project_json(data);
    let sysroot = Ok(get_fake_sysroot());
    let project_workspace =
        ProjectWorkspace::Json { project, sysroot, rustc_cfg: Vec::new(), toolchain: None };
    to_crate_graph(project_workspace)
}

fn get_test_json_file<T: DeserializeOwned>(file: &str) -> T {
    let file = get_test_path(file);
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

fn replace_fake_sys_root(s: &mut String) {
    let fake_sysroot_path = get_test_path("fake-sysroot");
    let fake_sysroot_path = if cfg!(windows) {
        let normalized_path =
            fake_sysroot_path.to_str().expect("expected str").replace('\\', r#"\\"#);
        format!(r#"{}\\"#, normalized_path)
    } else {
        format!("{}/", fake_sysroot_path.to_str().expect("expected str"))
    };
    *s = s.replace(&fake_sysroot_path, "$FAKESYSROOT$")
}

fn get_test_path(file: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base.join("test_data").join(file)
}

fn get_fake_sysroot() -> Sysroot {
    let sysroot_path = get_test_path("fake-sysroot");
    // there's no `libexec/` directory with a `proc-macro-srv` binary in that
    // fake sysroot, so we give them both the same path:
    let sysroot_dir = AbsPathBuf::assert(sysroot_path);
    let sysroot_src_dir = sysroot_dir.clone();
    Sysroot::load(sysroot_dir, sysroot_src_dir, false)
}

fn rooted_project_json(data: ProjectJsonData) -> ProjectJson {
    let mut root = "$ROOT$".to_owned();
    replace_root(&mut root, true);
    let path = Path::new(&root);
    let base = AbsPath::assert(path);
    ProjectJson::new(base, data)
}

fn to_crate_graph(project_workspace: ProjectWorkspace) -> (CrateGraph, ProcMacroPaths) {
    project_workspace.to_crate_graph(
        &mut {
            let mut counter = 0;
            move |_path| {
                counter += 1;
                Some(FileId::from_raw(counter))
            }
        },
        &Default::default(),
    )
}

fn check_crate_graph(crate_graph: CrateGraph, expect: ExpectFile) {
    let mut crate_graph = format!("{crate_graph:#?}");
    replace_root(&mut crate_graph, false);
    replace_fake_sys_root(&mut crate_graph);
    expect.assert_eq(&crate_graph);
}

#[test]
fn cargo_hello_world_project_model_with_wildcard_overrides() {
    let cfg_overrides = CfgOverrides {
        global: CfgDiff::new(Vec::new(), vec![CfgAtom::Flag("test".into())]).unwrap(),
        selective: Default::default(),
    };
    let (crate_graph, _proc_macros) =
        load_cargo_with_overrides("hello-world-metadata.json", cfg_overrides);
    check_crate_graph(
        crate_graph,
        expect_file![
            "../test_data/output/cargo_hello_world_project_model_with_wildcard_overrides.txt"
        ],
    )
}

#[test]
fn cargo_hello_world_project_model_with_selective_overrides() {
    let cfg_overrides = CfgOverrides {
        global: Default::default(),
        selective: std::iter::once((
            "libc".to_owned(),
            CfgDiff::new(Vec::new(), vec![CfgAtom::Flag("test".into())]).unwrap(),
        ))
        .collect(),
    };
    let (crate_graph, _proc_macros) =
        load_cargo_with_overrides("hello-world-metadata.json", cfg_overrides);
    check_crate_graph(
        crate_graph,
        expect_file![
            "../test_data/output/cargo_hello_world_project_model_with_selective_overrides.txt"
        ],
    )
}

#[test]
fn cargo_hello_world_project_model() {
    let (crate_graph, _proc_macros) = load_cargo("hello-world-metadata.json");
    check_crate_graph(
        crate_graph,
        expect_file!["../test_data/output/cargo_hello_world_project_model.txt"],
    )
}

#[test]
fn rust_project_hello_world_project_model() {
    let (crate_graph, _proc_macros) = load_rust_project("hello-world-project.json");
    check_crate_graph(
        crate_graph,
        expect_file!["../test_data/output/rust_project_hello_world_project_model.txt"],
    );
}

#[test]
fn rust_project_is_proc_macro_has_proc_macro_dep() {
    let (crate_graph, _proc_macros) = load_rust_project("is-proc-macro-project.json");
    // Since the project only defines one crate (outside the sysroot crates),
    // it should be the one with the biggest Id.
    let crate_id = crate_graph.iter().max().unwrap();
    let crate_data = &crate_graph[crate_id];
    // Assert that the project crate with `is_proc_macro` has a dependency
    // on the proc_macro sysroot crate.
    crate_data.dependencies.iter().find(|&dep| dep.name.deref() == "proc_macro").unwrap();
}

#[test]
fn crate_graph_dedup_identical() {
    let (mut crate_graph, proc_macros) =
        load_cargo_with_fake_sysroot(&mut Default::default(), "regex-metadata.json");
    crate_graph.sort_deps();

    let (d_crate_graph, mut d_proc_macros) = (crate_graph.clone(), proc_macros.clone());

    crate_graph.extend(d_crate_graph.clone(), &mut d_proc_macros, |_| ());
    assert!(crate_graph.iter().eq(d_crate_graph.iter()));
    assert_eq!(proc_macros, d_proc_macros);
}

#[test]
fn crate_graph_dedup() {
    let path_map = &mut Default::default();
    let (mut crate_graph, _proc_macros) =
        load_cargo_with_fake_sysroot(path_map, "ripgrep-metadata.json");
    assert_eq!(crate_graph.iter().count(), 81);
    crate_graph.sort_deps();
    let (regex_crate_graph, mut regex_proc_macros) =
        load_cargo_with_fake_sysroot(path_map, "regex-metadata.json");
    assert_eq!(regex_crate_graph.iter().count(), 60);

    crate_graph.extend(regex_crate_graph, &mut regex_proc_macros, |_| ());
    assert_eq!(crate_graph.iter().count(), 118);
}

#[test]
fn test_deduplicate_origin_dev() {
    let path_map = &mut Default::default();
    let (mut crate_graph, _proc_macros) =
        load_cargo_with_fake_sysroot(path_map, "deduplication_crate_graph_A.json");
    crate_graph.sort_deps();
    let (crate_graph_1, mut _proc_macros_2) =
        load_cargo_with_fake_sysroot(path_map, "deduplication_crate_graph_B.json");

    crate_graph.extend(crate_graph_1, &mut _proc_macros_2, |_| ());

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
    let path_map = &mut Default::default();
    let (mut crate_graph, _proc_macros) =
        load_cargo_with_fake_sysroot(path_map, "deduplication_crate_graph_B.json");
    crate_graph.sort_deps();
    let (crate_graph_1, mut _proc_macros_2) =
        load_cargo_with_fake_sysroot(path_map, "deduplication_crate_graph_A.json");

    crate_graph.extend(crate_graph_1, &mut _proc_macros_2, |_| ());

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
fn smoke_test_real_sysroot_cargo() {
    if std::env::var("SYSROOT_CARGO_METADATA").is_err() {
        return;
    }
    let file_map = &mut FxHashMap::<AbsPathBuf, FileId>::default();
    let meta = get_test_json_file("hello-world-metadata.json");

    let cargo_workspace = CargoWorkspace::new(meta);
    let sysroot = Ok(Sysroot::discover(
        AbsPath::assert(Path::new(env!("CARGO_MANIFEST_DIR"))),
        &Default::default(),
        true,
    )
    .unwrap());

    let project_workspace = ProjectWorkspace::Cargo {
        cargo: cargo_workspace,
        build_scripts: WorkspaceBuildScripts::default(),
        sysroot,
        rustc: Err(None),
        rustc_cfg: Vec::new(),
        cfg_overrides: Default::default(),
        toolchain: None,
        target_layout: Err("target_data_layout not loaded".into()),
    };
    project_workspace.to_crate_graph(
        &mut {
            |path| {
                let len = file_map.len();
                Some(*file_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
            }
        },
        &Default::default(),
    );
}
