use std::ops::Deref;

use base_db::{CrateGraph, ProcMacroPaths};
use cargo_metadata::Metadata;
use cfg::{CfgAtom, CfgDiff};
use expect_test::{expect_file, ExpectFile};
use intern::sym;
use paths::{AbsPath, AbsPathBuf, Utf8Path, Utf8PathBuf};
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;
use span::FileId;
use triomphe::Arc;

use crate::{
    sysroot::SysrootMode, workspace::ProjectWorkspaceKind, CargoWorkspace, CfgOverrides,
    ManifestPath, ProjectJson, ProjectJsonData, ProjectWorkspace, Sysroot, WorkspaceBuildScripts,
};

fn load_cargo(file: &str) -> (CrateGraph, ProcMacroPaths) {
    load_cargo_with_overrides(file, CfgOverrides::default())
}

fn load_cargo_with_overrides(
    file: &str,
    cfg_overrides: CfgOverrides,
) -> (CrateGraph, ProcMacroPaths) {
    let meta: Metadata = get_test_json_file(file);
    let manifest_path =
        ManifestPath::try_from(AbsPathBuf::try_from(meta.workspace_root.clone()).unwrap()).unwrap();
    let cargo_workspace = CargoWorkspace::new(meta, manifest_path);
    let project_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::Cargo {
            cargo: cargo_workspace,
            build_scripts: WorkspaceBuildScripts::default(),
            rustc: Err(None),
            cargo_config_extra_env: Default::default(),
            error: None,
            set_test: true,
        },
        cfg_overrides,
        sysroot: Sysroot::empty(),
        rustc_cfg: Vec::new(),
        toolchain: None,
        target_layout: Err("target_data_layout not loaded".into()),
    };
    to_crate_graph(project_workspace)
}

fn load_rust_project(file: &str) -> (CrateGraph, ProcMacroPaths) {
    let data = get_test_json_file(file);
    let project = rooted_project_json(data);
    let sysroot = get_fake_sysroot();
    let project_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::Json(project),
        sysroot,
        rustc_cfg: Vec::new(),
        toolchain: None,
        target_layout: Err(Arc::from("test has no data layout")),
        cfg_overrides: Default::default(),
    };
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

fn replace_cargo(s: &mut String) {
    let path = toolchain::Tool::Cargo.path().to_string().escape_debug().collect::<String>();
    *s = s.replace(&path, "$CARGO$");
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
        let normalized_path = fake_sysroot_path.as_str().replace('\\', r#"\\"#);
        format!(r#"{normalized_path}\\"#)
    } else {
        format!("{}/", fake_sysroot_path.as_str())
    };
    *s = s.replace(&fake_sysroot_path, "$FAKESYSROOT$")
}

fn get_test_path(file: &str) -> Utf8PathBuf {
    let base = Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base.join("test_data").join(file)
}

fn get_fake_sysroot() -> Sysroot {
    let sysroot_path = get_test_path("fake-sysroot");
    // there's no `libexec/` directory with a `proc-macro-srv` binary in that
    // fake sysroot, so we give them both the same path:
    let sysroot_dir = AbsPathBuf::assert(sysroot_path);
    let sysroot_src_dir = sysroot_dir.clone();
    Sysroot::load(Some(sysroot_dir), Some(sysroot_src_dir))
}

fn rooted_project_json(data: ProjectJsonData) -> ProjectJson {
    let mut root = "$ROOT$".to_owned();
    replace_root(&mut root, true);
    let path = Utf8Path::new(&root);
    let base = AbsPath::assert(path);
    ProjectJson::new(None, base, data)
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
    replace_cargo(&mut crate_graph);
    replace_fake_sys_root(&mut crate_graph);
    expect.assert_eq(&crate_graph);
}

#[test]
fn cargo_hello_world_project_model_with_wildcard_overrides() {
    let cfg_overrides = CfgOverrides {
        global: CfgDiff::new(Vec::new(), vec![CfgAtom::Flag(sym::test.clone())]).unwrap(),
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
            CfgDiff::new(Vec::new(), vec![CfgAtom::Flag(sym::test.clone())]).unwrap(),
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
fn rust_project_cfg_groups() {
    let (crate_graph, _proc_macros) = load_rust_project("cfg-groups.json");
    check_crate_graph(crate_graph, expect_file!["../test_data/output/rust_project_cfg_groups.txt"]);
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
fn smoke_test_real_sysroot_cargo() {
    let file_map = &mut FxHashMap::<AbsPathBuf, FileId>::default();
    let meta: Metadata = get_test_json_file("hello-world-metadata.json");
    let manifest_path =
        ManifestPath::try_from(AbsPathBuf::try_from(meta.workspace_root.clone()).unwrap()).unwrap();
    let cargo_workspace = CargoWorkspace::new(meta, manifest_path);
    let sysroot = Sysroot::discover(
        AbsPath::assert(Utf8Path::new(env!("CARGO_MANIFEST_DIR"))),
        &Default::default(),
    );
    assert!(matches!(sysroot.mode(), SysrootMode::Workspace(_)));
    let project_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::Cargo {
            cargo: cargo_workspace,
            build_scripts: WorkspaceBuildScripts::default(),
            rustc: Err(None),
            cargo_config_extra_env: Default::default(),
            error: None,
            set_test: true,
        },
        sysroot,
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
